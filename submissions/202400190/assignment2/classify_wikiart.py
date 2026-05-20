import argparse
import base64
import html
import io
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from PIL import Image


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {"type": "string", "enum": ["yes", "no"]},
        "has_animal": {"type": "string", "enum": ["yes", "no"]},
        "has_flower": {"type": "string", "enum": ["yes", "no"]},
        "evidence": {
            "type": "string",
            "description": (
                "One concise sentence explaining the classification. "
                "Include location terms such as center, top-left, foreground, or background."
            ),
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "evidence"],
    "additionalProperties": False,
}

PROMPT = (
    "Examine this artwork and determine whether each of the following is visibly present:\n"
    "- has_human: a human figure or person (yes/no)\n"
    "- has_animal: any animal (yes/no)\n"
    "- has_flower: any flower (yes/no)\n\n"
    "Focus only on visible content, not the artwork's style, medium, or genre. "
    "In the evidence field, write one sentence explaining your decisions and mention "
    "approximate locations (e.g., center, top-left, foreground, background) where relevant."
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def to_jpeg_bytes(image: Image.Image, max_size: int) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()


def classify(client: OpenAI, data_url: str, model: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "wikiart_classification",
                "strict": True,
                "schema": SCHEMA,
            },
        },
        extra_body={
            "reasoning": {"effort": "none", "exclude": True},
            "plugins": [{"id": "response-healing"}],
        },
        extra_headers={"X-Title": "HUFS-LAI-MM WikiArt Assignment"},
    )
    if not response.choices:
        raise RuntimeError(f"Empty choices in response: {response}")
    message = response.choices[0].message
    content = message.content
    if isinstance(content, dict):
        return content
    if content is None and hasattr(message, "parsed") and message.parsed is not None:
        return message.parsed
    if content is None:
        raise RuntimeError(f"Model returned empty content: {message}")
    return json.loads(content)


def fetch_samples(n: int):
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    for idx, item in enumerate(dataset.take(n)):
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


def process(sample: dict, client: OpenAI, model: str, image_dir: Path, max_size: int, retries: int) -> dict:
    idx = sample["index"]
    jpeg = to_jpeg_bytes(sample["image"], max_size)
    img_path = image_dir / f"sample_{idx:03d}.jpg"
    img_path.write_bytes(jpeg)

    labels, error = {}, ""
    for attempt in range(1, retries + 1):
        try:
            labels = classify(client, to_data_url(jpeg), model)
            break
        except Exception as exc:
            error = str(exc)
            if attempt < retries:
                print(f"  sample {idx} attempt {attempt} failed, retrying: {exc}", flush=True)
                time.sleep(2)

    return {
        "index": idx,
        "artist": sample["artist"],
        "genre": sample["genre"],
        "style": sample["style"],
        "image_path": str(img_path),
        "labels": labels,
        "error": error,
    }


def build_html(results: list, model: str) -> str:
    cards = []
    for r in sorted(results, key=lambda x: x["index"]):
        labels = r["labels"]
        error = r.get("error", "")
        if error or not labels:
            badges = "<span class='badge err'>error</span>"
            evidence = f"<p class='evidence err'>{html.escape(error)}</p>"
        else:
            def badge(name, val):
                return f"<span class='badge {val}'>{name}: {val}</span>"
            badges = badge("Human", labels["has_human"]) + badge("Animal", labels["has_animal"]) + badge("Flower", labels["has_flower"])
            evidence = f"<p class='evidence'>{html.escape(labels.get('evidence', ''))}</p>"

        cards.append(f"""
        <div class="card">
          <div class="card-img">
            <img src="{html.escape(r['image_path'])}" alt="sample {r['index']}">
            <span class="index-badge">#{r['index']}</span>
          </div>
          <div class="card-body">
            <div class="meta"></div>
            <div class="badges">{badges}</div>
            {evidence}
          </div>
        </div>""")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WikiArt Classification Results</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    body {{ font-family: sans-serif; margin: 0; padding: 24px; background: #121212; color: #e0e0e0; }}
    header {{ margin-bottom: 24px; }}
    header h1 {{ margin: 0 0 4px; font-size: 1.5rem; color: #fff; }}
    header p {{ margin: 0; font-size: 0.85rem; color: #888; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: #1e1e1e; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 2px 8px rgba(0,0,0,0.4); }}
    .card-img {{ position: relative; background: #111; display: flex; justify-content: center; align-items: center; min-height: 180px; }}
    .card-img img {{ max-width: 100%; max-height: 220px; object-fit: contain; display: block; }}
    .index-badge {{ position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.6); color: #fff; font-size: 0.75rem; padding: 2px 7px; border-radius: 999px; }}
    .card-body {{ padding: 12px; display: flex; flex-direction: column; gap: 8px; flex: 1; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 4px; }}
    .meta span {{ font-size: 0.75rem; background: #2a2a2a; color: #aaa; padding: 2px 8px; border-radius: 999px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .badge {{ font-size: 0.8rem; font-weight: bold; padding: 3px 10px; border-radius: 999px; }}
    .badge.yes {{ background: #1a4731; color: #6ee7a0; }}
    .badge.no  {{ background: #4a1a1a; color: #f28b82; }}
    .badge.err {{ background: #3a3000; color: #fdd663; }}
    .evidence {{ font-size: 0.82rem; color: #bbb; margin: 0; line-height: 1.5; }}
    .evidence.err {{ color: #fdd663; }}
  </style>
</head>
<body>
  <header>
    <h1>WikiArt Classification Results</h1>
    <p>Model: {html.escape(model)}</p>
  </header>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>"""


def retry_failed(client: OpenAI, model: str, retries: int, sleep: float, json_path: Path, html_path: Path):
    if not json_path.exists():
        raise SystemExit(f"{json_path} not found. Run without --retry-failed first.")
    results = json.loads(json_path.read_text(encoding="utf-8"))
    failed = [r for r in results if r.get("error")]
    if not failed:
        print("No failed samples found.")
        return
    print(f"Retrying {len(failed)} failed sample(s)...\n", flush=True)
    lookup = {r["index"]: r for r in results}
    for r in failed:
        idx = r["index"]
        img_path = Path(r["image_path"])
        if not img_path.exists():
            print(f"  sample {idx}: image file missing, skipping", flush=True)
            continue
        jpeg = img_path.read_bytes()
        t0 = time.perf_counter()
        print(f"retrying sample {idx}...", flush=True)
        labels, error = {}, ""
        for attempt in range(1, retries + 1):
            try:
                labels = classify(client, to_data_url(jpeg), model)
                break
            except Exception as exc:
                error = str(exc)
                if attempt < retries:
                    print(f"  attempt {attempt} failed, retrying: {exc}", flush=True)
                    time.sleep(2)
        elapsed = time.perf_counter() - t0
        if error:
            print(f"  -> still failed in {elapsed:.1f}s: {error}", flush=True)
        else:
            lbl = labels
            print(f"  -> done in {elapsed:.1f}s | human={lbl['has_human']} animal={lbl['has_animal']} flower={lbl['has_flower']}", flush=True)
        lookup[idx] = {**r, "labels": labels, "error": error}
        time.sleep(sleep)

    results = sorted(lookup.values(), key=lambda x: x["index"])
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(build_html(results, model), encoding="utf-8")
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    print(f"\nUpdated files. {total - errors}/{total} succeeded.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--retry-failed", action="store_true", help="재시도: results.json의 error 항목만 다시 분류")
    args = parser.parse_args()

    load_env()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not token:
        raise SystemExit("OPENROUTER_TOKEN not set. Add it to .env or export it.")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=token, max_retries=0, timeout=args.timeout)

    if args.retry_failed:
        retry_failed(client, args.model, args.retries, args.sleep,
                     Path(args.output_json), Path(args.output_html))
        return

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.limit} samples from huggan/wikiart...", flush=True)
    samples = list(fetch_samples(args.limit))
    print(f"Loaded {len(samples)} samples. Starting classification...\n", flush=True)

    results = []
    for sample in samples:
        t0 = time.perf_counter()
        print(f"[{sample['index'] + 1}/{len(samples)}] classifying sample {sample['index']}...", flush=True)
        result = process(sample, client, args.model, image_dir, args.max_size, args.retries)
        elapsed = time.perf_counter() - t0
        if result["error"]:
            print(f"  -> failed in {elapsed:.1f}s: {result['error']}", flush=True)
        else:
            lbl = result["labels"]
            print(f"  -> done in {elapsed:.1f}s | human={lbl['has_human']} animal={lbl['has_animal']} flower={lbl['has_flower']}", flush=True)
        results.append(result)
        time.sleep(args.sleep)

    html_out = Path(args.output_html)
    html_out.write_text(build_html(results, args.model), encoding="utf-8")
    print(f"\nWrote {html_out}")

    if args.output_json:
        json_out = Path(args.output_json)
        json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {json_out}")

    total = len(results)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone: {total - errors}/{total} succeeded.")


if __name__ == "__main__":
    main()
