"""
classify_wikiart.py

Classifies the first N samples from huggan/wikiart (default: 20) using
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free via OpenRouter API.

For each image, determines:
  - has_human  : whether any human (person, figure, portrait) appears anywhere
  - has_animal : whether any animal (mammal, bird, fish, insect, etc.) appears
  - has_flower : whether any flower (blossom, petal, bouquet) appears

Results are saved to results.html and images/ folder.

Usage:
    python classify_wikiart.py [--limit 20] [--timeout 60] [--max-retries 3]
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
MODEL = "nemotron_3_nano_omni"
IMAGES_DIR = Path("images")
RESULTS_HTML = Path("results.html")
RESULTS_JSON = Path("results.json")

# JSON schema for structured / schema-guided decoding
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": (
                "Whether any human figure (person, portrait, figure, deity in human form) "
                "is visible anywhere in the image — in the foreground, center, or background."
            ),
        },
        "has_animal": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": (
                "Whether any animal (mammal, bird, reptile, fish, insect, mythological creature, etc.) "
                "is visible anywhere in the image — in the foreground, center, or background."
            ),
        },
        "has_flower": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": (
                "Whether any flower (blossom, petal, bouquet, floral motif) "
                "is visible anywhere in the image — in the foreground, center, or background."
            ),
        },
        "reason": {
            "type": "string",
            "description": (
                "A brief explanation describing what is visible in the foreground, center, and background "
                "of the image, and how that led to each classification decision."
            ),
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an expert art analyst specializing in visual content classification.

Carefully examine the artwork and classify whether the following subjects are present.
Consider ALL regions of the image — foreground, center, and background.

Definitions:
- has_human  : "yes" if any human figure (person, portrait, figure, deity depicted in human form, 
               human body part) is visible anywhere — in the foreground, center, or background; 
               otherwise "no".
- has_animal : "yes" if any animal (mammal, bird, reptile, fish, insect, or any real/mythological 
               creature that is not human) is visible anywhere — in the foreground, center, or background; 
               otherwise "no".
- has_flower : "yes" if any flower (blossoms, petals, bouquet, floral decoration, or floral motif) 
               is visible anywhere — in the foreground, center, or background; otherwise "no".

You MUST respond with valid JSON that strictly follows the provided schema.
Each of has_human, has_animal, has_flower must be exactly "yes" or "no".
In the "reason" field, describe what you observe in the foreground, center, and background, 
then explain each classification decision concisely."""

USER_PROMPT = """Analyze this artwork and classify whether it contains humans, animals, and/or flowers.

Look carefully at:
- The FOREGROUND (objects/subjects closest to the viewer)
- The CENTER (main subject or focal point of the composition)
- The BACKGROUND (distant elements, sky, landscape, architectural details)

Then fill in the JSON:
{
  "has_human":  "yes" or "no"   — Is any human figure visible anywhere in the image?
  "has_animal": "yes" or "no"   — Is any animal visible anywhere in the image?
  "has_flower": "yes" or "no"   — Is any flower visible anywhere in the image?
  "reason":     <string>        — Describe foreground/center/background content and justify each label.
}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(img: Image.Image, idx: int) -> Path:
    IMAGES_DIR.mkdir(exist_ok=True)
    path = IMAGES_DIR / f"sample_{idx:03d}.jpg"
    img.convert("RGB").save(path, "JPEG")
    return path


def _validate_output(parsed: dict) -> dict:
    required = ["has_human", "has_animal", "has_flower", "reason"]
    for k in required:
        if k not in parsed:
            raise ValueError(f"missing required field: {k}")
    for k in ["has_human", "has_animal", "has_flower"]:
        if parsed[k] not in {"yes", "no"}:
            raise ValueError(f"invalid label in {k}: {parsed[k]}")
    if not isinstance(parsed["reason"], str):
        raise ValueError("reason must be a string")
    return parsed


def classify_image_once(client: OpenAI, img: Image.Image, timeout: float = 60.0) -> dict:
    """Call OpenRouter API once with schema-guided decoding and parse result."""
    b64 = pil_to_base64(img)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={
            "reasoning": {"enabled": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "wikiart_classification",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        },
        timeout=timeout,
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    return _validate_output(parsed)


def classify_image_with_retry(
    client: OpenAI,
    img: Image.Image,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Retry on timeout, JSON parse failure, or invalid schema output."""
    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            return classify_image_once(client, img, timeout=timeout)
        except Exception as exc:
            msg = str(exc).lower()
            is_retryable = (
                isinstance(exc, json.JSONDecodeError)
                or "timeout" in msg
                or "timed out" in msg
                or "missing required field" in msg
                or "invalid label" in msg
            )
            last_error = exc
            if not is_retryable or attempt > max_retries:
                raise
            sleep_s = retry_delay * attempt
            print(
                f"    retry {attempt}/{max_retries} after error: {exc} "
                f"(sleep {sleep_s:.1f}s)"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"classification failed after retries: {last_error}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>WikiArt Classification Results</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; color: #333; }}
  .summary {{ display: flex; gap: 16px; justify-content: center; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{ background: #fff; border-radius: 8px; padding: 12px 24px; box-shadow: 0 2px 6px rgba(0,0,0,.1); text-align: center; }}
  .stat .num {{ font-size: 2em; font-weight: bold; color: #4a90e2; }}
  .stat .label {{ color: #666; font-size: .9em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.12); overflow: hidden; }}
  .card img {{ width: 100%; height: 200px; object-fit: cover; }}
  .card-body {{ padding: 12px; }}
  .card-title {{ font-weight: bold; margin-bottom: 8px; color: #333; }}
  .badges {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
  .badge {{ padding: 3px 10px; border-radius: 12px; font-size: .8em; font-weight: 600; }}
  .badge-yes {{ background: #d4edda; color: #155724; }}
  .badge-no  {{ background: #f8d7da; color: #721c24; }}
  .badge-err {{ background: #fff3cd; color: #856404; }}
  .reason {{ font-size: .82em; color: #555; line-height: 1.45; }}
  .error {{ color: #c0392b; font-size: .85em; }}
</style>
</head>
<body>
<h1>WikiArt Classification Results</h1>
<div class="summary">
  <div class="stat"><div class="num">{total}</div><div class="label">Total Samples</div></div>
  <div class="stat"><div class="num">{n_human}</div><div class="label">Has Human</div></div>
  <div class="stat"><div class="num">{n_animal}</div><div class="label">Has Animal</div></div>
  <div class="stat"><div class="num">{n_flower}</div><div class="label">Has Flower</div></div>
  <div class="stat"><div class="num">{n_error}</div><div class="label">Errors</div></div>
</div>
<div class="grid">
{cards}
</div>
</body>
</html>"""

CARD_TEMPLATE = """  <div class="card">
    <img src="{img_src}" alt="Sample {idx}"/>
    <div class="card-body">
      <div class="card-title">Sample {idx}</div>
      <div class="badges">
        <span class="badge {human_cls}">Human: {has_human}</span>
        <span class="badge {animal_cls}">Animal: {has_animal}</span>
        <span class="badge {flower_cls}">Flower: {has_flower}</span>
      </div>
      <div class="reason">{reason}</div>
    </div>
  </div>"""

ERROR_CARD_TEMPLATE = """  <div class="card">
    <img src="{img_src}" alt="Sample {idx}"/>
    <div class="card-body">
      <div class="card-title">Sample {idx}</div>
      <div class="error">Error: {error}</div>
    </div>
  </div>"""


def badge_cls(val: str) -> str:
    if val == "yes":
        return "badge-yes"
    if val == "no":
        return "badge-no"
    return "badge-err"


def build_html(results: list[dict]) -> str:
    cards_html = []
    n_human = n_animal = n_flower = n_error = 0
    for r in results:
        img_src = r.get("img_path", "")
        idx = r["idx"]
        if r.get("error"):
            n_error += 1
            cards_html.append(
                ERROR_CARD_TEMPLATE.format(
                    img_src=img_src, idx=idx, error=r["error"]
                )
            )
        else:
            hh = r.get("has_human", "?")
            ha = r.get("has_animal", "?")
            hf = r.get("has_flower", "?")
            if hh == "yes":
                n_human += 1
            if ha == "yes":
                n_animal += 1
            if hf == "yes":
                n_flower += 1
            cards_html.append(
                CARD_TEMPLATE.format(
                    img_src=img_src,
                    idx=idx,
                    has_human=hh,
                    has_animal=ha,
                    has_flower=hf,
                    human_cls=badge_cls(hh),
                    animal_cls=badge_cls(ha),
                    flower_cls=badge_cls(hf),
                    reason=r.get("reason", ""),
                )
            )
    return HTML_TEMPLATE.format(
        total=len(results),
        n_human=n_human,
        n_animal=n_animal,
        n_flower=n_flower,
        n_error=n_error,
        cards="\n".join(cards_html),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify WikiArt images with MLLM")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples (default: 20)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for timeout/invalid JSON")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Base retry delay in seconds")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_TOKEN")
    if not api_key:
        raise RuntimeError("OPENROUTER_TOKEN not found in environment / .env")

    client = OpenAI(
        # base_url="https://openrouter.ai/api/v1",
        base_url="http://192.168.0.106:8000/v1",
        api_key=api_key,
    )

    print(f"Loading huggan/wikiart (first {args.limit} samples)…")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    samples = list(ds.take(args.limit))
    print(f"Loaded {len(samples)} samples.")

    results = []

    for idx, sample in enumerate(samples):
        img: Image.Image = sample["image"]
        img_path = save_image(img, idx)
        print(f"  [{idx+1}/{len(samples)}] classifying…")
        t0 = time.time()
        try:
            parsed = classify_image_with_retry(
                client,
                img,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            elapsed = time.time() - t0
            print(
                f"  [{idx+1}/{len(samples)}] done in {elapsed:.1f}s — "
                f"human={parsed['has_human']} animal={parsed['has_animal']} flower={parsed['has_flower']}"
            )
            results.append({
                "idx": idx,
                "img_path": str(img_path),
                "has_human": parsed["has_human"],
                "has_animal": parsed["has_animal"],
                "has_flower": parsed["has_flower"],
                "reason": parsed["reason"],
            })
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(samples)}] ERROR after {elapsed:.1f}s: {exc}")
            results.append({"idx": idx, "img_path": str(img_path), "error": str(exc)})

    # Write JSON
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON saved → {RESULTS_JSON}")

    # Write HTML
    html = build_html(results)
    RESULTS_HTML.write_text(html, encoding="utf-8")
    print(f"HTML saved → {RESULTS_HTML}")


if __name__ == "__main__":
    main()
