import argparse
import base64
import html
import io
import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from PIL import Image


DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LIMIT = 20

# JSON Schema for structured model output
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the image visibly contains a human or person figure.",
        },
        "has_animal": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the image visibly contains any animal (excluding humans).",
        },
        "has_flower": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the image visibly contains a flower.",
        },
        "evidence": {
            "type": "string",
            "description": (
                "One concise sentence explaining the yes/no decisions. "
                "Include spatial positions such as center, top-left, foreground, or background when relevant."
            ),
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "evidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a careful art analyst. "
    "Your task is to detect the visible presence of specific objects in artwork images. "
    "Definitions:\n"
    "- has_human: 'yes' if the image shows any human being or person, even partially. "
    "'no' for purely abstract or non-figurative art without people.\n"
    "- has_animal: 'yes' if the image shows any non-human animal (mammal, bird, fish, insect, etc.). "
    "'no' otherwise.\n"
    "- has_flower: 'yes' if the image shows any flower (real or stylised). "
    "'no' otherwise.\n"
    "Always output valid JSON matching the required schema."
)

USER_PROMPT = (
    "Examine this artwork carefully and fill in the JSON schema.\n"
    "- has_human: does this artwork visibly contain a human or person?\n"
    "- has_animal: does this artwork visibly contain a non-human animal?\n"
    "- has_flower: does this artwork visibly contain a flower?\n"
    "For evidence, write one concise sentence with spatial references "
    "(e.g. center, top-left, foreground, background) where applicable."
)


def load_env(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def to_jpeg_bytes(image: Image.Image, max_size: int) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()


def _extract_from_reasoning(reasoning: str) -> dict | None:
    result = {}
    for key in ("has_human", "has_animal", "has_flower"):
        m = re.search(rf'{key}[\s:=]*["\']?(yes|no)["\']?', reasoning, re.IGNORECASE)
        if m:
            result[key] = m.group(1).lower()
    if len(result) == 3:
        result["evidence"] = "Extracted from model reasoning output."
        return result
    return None


def _parse_json_tolerant(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise
        # Try progressively shorter substrings from the end
        chunk = text[start:]
        for i in range(len(chunk), 0, -1):
            try:
                return json.loads(chunk[:i])
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("No valid JSON found", text, 0)


def classify_image(client: OpenAI, data_url: str, model: str, timeout: float, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "wikiart_classification",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
                extra_headers={"X-Title": "HUFS-LAI-MM WikiArt Assignment2"},
                timeout=timeout,
            )
            message = response.choices[0].message
            content = message.content
            if isinstance(content, dict):
                return content
            if content is None and getattr(message, "parsed", None) is not None:
                return message.parsed
            if content is None:
                # Reasoning model put the answer only in reasoning_details
                reasoning = getattr(message, "reasoning", "") or ""
                for rd in getattr(message, "reasoning_details", []) or []:
                    reasoning += rd.get("text", "")
                extracted = _extract_from_reasoning(reasoning)
                if extracted:
                    return extracted
                raise RuntimeError(f"Empty response from model: {message}")
            return _parse_json_tolerant(content)
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait)
            else:
                raise


def iter_samples(limit: int):
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    for idx, item in enumerate(dataset.take(limit)):
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        yield {
            "index": idx,
            "artist": item.get("artist", ""),
            "genre": item.get("genre", ""),
            "style": item.get("style", ""),
            "image": image,
        }


def build_html(results: list, output_path: str, model: str):
    rows = []
    for row in sorted(results, key=lambda r: r["index"]):
        labels = row.get("labels", {})
        error = row.get("error", "")
        if error:
            cells = (
                "<td class='error'>—</td>" * 3
                + f"<td class='error'>{html.escape(error)}</td>"
            )
        else:
            cells = (
                f"<td class=\"{labels['has_human']}\">{labels['has_human']}</td>"
                f"<td class=\"{labels['has_animal']}\">{labels['has_animal']}</td>"
                f"<td class=\"{labels['has_flower']}\">{labels['has_flower']}</td>"
                f"<td>{html.escape(labels.get('evidence', ''))}</td>"
            )
        rows.append(f"""
        <tr>
          <td>{row['index']}</td>
          <td><img src="{html.escape(row['image_path'])}" alt="sample {row['index']}"></td>
          <td>
            <div><strong>Artist:</strong> {html.escape(str(row['artist']))}</div>
            <div><strong>Genre:</strong> {html.escape(str(row['genre']))}</div>
            <div><strong>Style:</strong> {html.escape(str(row['style']))}</div>
          </td>
          {cells}
        </tr>""")

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WikiArt Classification — 202400220</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    p.meta {{ color: #555; margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f0f0; position: sticky; top: 0; }}
    img {{ max-width: 180px; max-height: 180px; object-fit: contain; }}
    .yes {{ background: #d4edda; font-weight: bold; text-align: center; color: #155724; }}
    .no  {{ background: #f8d7da; font-weight: bold; text-align: center; color: #721c24; }}
    .error {{ background: #fff3cd; color: #664d03; }}
  </style>
</head>
<body>
  <h1>WikiArt Classification Results</h1>
  <p class="meta">Model: {html.escape(model)} &nbsp;|&nbsp; Samples: {len(results)} &nbsp;|&nbsp; Student: Yeonjoo Yoo (202400220)</p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Image</th><th>Metadata</th>
        <th>Human</th><th>Animal</th><th>Flower</th><th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    Path(output_path).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Classify WikiArt images via OpenRouter")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of images to classify")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds to wait between API calls")
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--resume", action="store_true", help="Retry only failed samples from existing results.json")
    args = parser.parse_args()

    load_env()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not token:
        raise RuntimeError("OPENROUTER_TOKEN not found. Add it to .env or set as environment variable.")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=token)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.output_json).exists():
        results = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
        failed = [r for r in results if r.get("error")]
        print(f"Resuming: retrying {len(failed)} failed samples from saved images...", flush=True)
        for i, row in enumerate(failed):
            idx = row["index"]
            img_path = Path(row["image_path"])
            if not img_path.exists():
                print(f"[{idx}] Saved image not found, skipping", flush=True)
                continue
            print(f"[{i+1}/{len(failed)}] Retrying sample {idx}...", flush=True)
            t_start = time.perf_counter()
            jpeg = img_path.read_bytes()
            error = ""
            labels = {}
            try:
                labels = classify_image(client, to_data_url(jpeg), args.model, args.timeout)
            except Exception as exc:
                error = str(exc)
                print(f"  ERROR: {error}", flush=True)
            elapsed = time.perf_counter() - t_start
            if not error:
                print(f"  done in {elapsed:.1f}s — human={labels['has_human']} animal={labels['has_animal']} flower={labels['has_flower']}", flush=True)
            for r in results:
                if r["index"] == idx:
                    r["labels"] = labels
                    r["error"] = error
                    r["elapsed_seconds"] = round(elapsed, 3)
                    break
            if i < len(failed) - 1:
                time.sleep(args.sleep)
    else:
        print(f"Loading {args.limit} samples from huggan/wikiart ...", flush=True)
        t0 = time.perf_counter()
        samples = list(iter_samples(args.limit))
        print(f"Loaded {len(samples)} samples in {time.perf_counter() - t0:.1f}s", flush=True)

        results = []
        for sample in samples:
            idx = sample["index"]
            print(f"[{idx+1}/{len(samples)}] Classifying sample {idx}...", flush=True)
            t_start = time.perf_counter()

            jpeg = to_jpeg_bytes(sample["image"], args.max_size)
            img_path = image_dir / f"sample_{idx:03d}.jpg"
            img_path.write_bytes(jpeg)

            error = ""
            labels = {}
            try:
                labels = classify_image(client, to_data_url(jpeg), args.model, args.timeout)
            except Exception as exc:
                error = str(exc)
                print(f"  ERROR: {error}", flush=True)

            elapsed = time.perf_counter() - t_start
            if not error:
                print(f"  done in {elapsed:.1f}s — human={labels['has_human']} animal={labels['has_animal']} flower={labels['has_flower']}", flush=True)

            results.append({
                "index": idx,
                "artist": sample["artist"],
                "genre": sample["genre"],
                "style": sample["style"],
                "image_path": img_path.as_posix(),
                "labels": labels,
                "error": error,
                "elapsed_seconds": round(elapsed, 3),
            })

            if idx < len(samples) - 1:
                time.sleep(args.sleep)

    build_html(results, args.output_html, args.model)
    Path(args.output_json).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ok = sum(1 for r in results if not r["error"])
    print(f"\nDone. {ok}/{len(results)} classified successfully.")
    print(f"Output: {args.output_html}, {args.output_json}")


if __name__ == "__main__":
    main()
