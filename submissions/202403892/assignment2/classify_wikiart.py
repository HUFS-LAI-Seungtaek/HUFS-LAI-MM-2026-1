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


DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether a human figure, person, or portrait is clearly visible in the artwork.",
        },
        "has_animal": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether any animal (bird, mammal, fish, insect, or mythical creature) is clearly visible.",
        },
        "has_flower": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether any flower or floral element is clearly visible in the artwork.",
        },
        "evidence": {
            "type": "string",
            "description": (
                "One concise sentence explaining all three labels. "
                "Mention approximate positions such as center, top-left, foreground, or background when relevant."
            ),
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "evidence"],
    "additionalProperties": False,
}

PROMPT = (
    "Examine this artwork carefully and answer three questions about visible content:\n\n"
    "1. has_human — Is a human being (figure, portrait, or body part) clearly visible? Answer yes or no.\n"
    "2. has_animal — Is any animal (bird, mammal, fish, insect, reptile, or mythical creature) clearly visible? Answer yes or no.\n"
    "3. has_flower — Is any flower or floral element clearly visible? Answer yes or no.\n\n"
    "Focus only on what is depicted in the image, not the art style or medium. "
    "In the evidence field, write one concise sentence justifying all three answers. "
    "When relevant, mention approximate position using terms like center, top-left, foreground, background, etc. "
    "Return only the JSON object required by the schema."
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


def resize_to_jpeg(image: Image.Image, max_size: int) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("utf-8")


def classify_image(client: OpenAI, data_url: str, model: str, timeout: float) -> dict:
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
        timeout=timeout,
    )
    content = response.choices[0].message.content
    if isinstance(content, dict):
        return content
    if content is None:
        raise RuntimeError(f"Model returned empty content. Raw: {response.choices[0].message}")
    return json.loads(content)


def get_samples(limit: int) -> list:
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    samples = []
    for i, item in enumerate(dataset.take(limit)):
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        samples.append(
            {
                "index": i,
                "artist": item.get("artist", ""),
                "genre": item.get("genre", ""),
                "style": item.get("style", ""),
                "image": image,
            }
        )
    return samples


def write_html(results: list, output_path: str, model: str) -> None:
    rows = []
    for row in sorted(results, key=lambda r: r["index"]):
        labels = row.get("labels", {})
        error = row.get("error", "")

        if error:
            label_cells = "<td class='err' colspan='3'>error</td>"
            ev_cell = f"<td class='err'>{html.escape(error)}</td>"
        else:

            def cell(key):
                val = labels.get(key, "?")
                return f"<td class='{val}'>{val}</td>"

            label_cells = cell("has_human") + cell("has_animal") + cell("has_flower")
            ev_cell = f"<td>{html.escape(labels.get('evidence', ''))}</td>"

        img_src = html.escape(row["image_path"])
        rows.append(
            f"""
    <tr>
      <td class="idx">{row["index"]}</td>
      <td><img src="{img_src}" alt="sample {row['index']}"></td>
      <td>
        <div><b>Artist:</b> {html.escape(str(row["artist"]))}</div>
        <div><b>Genre:</b> {html.escape(str(row["genre"]))}</div>
        <div><b>Style:</b> {html.escape(str(row["style"]))}</div>
        <div class="time">{row.get("elapsed_seconds", 0):.1f}s</div>
      </td>
      {label_cells}
      {ev_cell}
    </tr>"""
        )

    success = sum(1 for r in results if not r.get("error"))
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WikiArt Classification — 202403892</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: Georgia, serif; margin: 0; padding: 24px; background: #f5f5f0; color: #333; }}
    h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
    .meta {{ color: #777; font-size: 0.88em; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 6px rgba(0,0,0,.08); border-radius: 6px; overflow: hidden; }}
    th {{ background: #34495e; color: white; padding: 10px 8px; text-align: left; position: sticky; top: 0; font-family: Arial, sans-serif; font-size: 0.9em; }}
    td {{ border: 1px solid #e8e8e8; padding: 8px; vertical-align: top; font-size: 0.88em; }}
    td.idx {{ text-align: center; color: #aaa; font-size: 0.8em; width: 36px; }}
    img {{ max-width: 150px; max-height: 150px; display: block; border-radius: 3px; }}
    .yes {{ background: #d4edda; color: #155724; font-weight: bold; text-align: center; font-family: Arial, sans-serif; }}
    .no {{ background: #f8d7da; color: #721c24; font-weight: bold; text-align: center; font-family: Arial, sans-serif; }}
    .err {{ background: #fff3cd; color: #664d03; font-size: 0.82em; }}
    .time {{ color: #bbb; font-size: 0.8em; margin-top: 6px; font-family: Arial, sans-serif; }}
    b {{ color: #555; font-family: Arial, sans-serif; }}
  </style>
</head>
<body>
  <h1>WikiArt Classification Results</h1>
  <div class="meta">
    Model: {html.escape(model)} &nbsp;|&nbsp;
    Samples: {len(results)} &nbsp;|&nbsp;
    Successful: {success} / {len(results)}
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Image</th>
        <th>Metadata</th>
        <th>Human</th>
        <th>Animal</th>
        <th>Flower</th>
        <th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>"""
    Path(output_path).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="WikiArt image classifier via OpenRouter API")
    parser.add_argument("--limit", type=int, default=20, help="Number of images to classify")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-size", type=int, default=320, help="Max image dimension in pixels")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds to wait between API calls")
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--image-dir", default="images")
    args = parser.parse_args()

    load_env()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not token:
        raise RuntimeError("OPENROUTER_TOKEN is missing. Put it in .env or set as environment variable.")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=token)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.limit} samples from huggan/wikiart...", flush=True)
    t0 = time.perf_counter()
    samples = get_samples(args.limit)
    print(f"Loaded {len(samples)} samples in {time.perf_counter() - t0:.1f}s", flush=True)

    results = []
    for i, sample in enumerate(samples):
        idx = sample["index"]
        print(f"[{idx:02d}] Classifying ({i+1}/{len(samples)})...", end=" ", flush=True)
        t1 = time.perf_counter()

        jpeg = resize_to_jpeg(sample["image"], args.max_size)
        img_path = image_dir / f"sample_{idx:03d}.jpg"
        img_path.write_bytes(jpeg)
        rel_path = f"{args.image_dir}/sample_{idx:03d}.jpg"

        error = ""
        labels = {}
        try:
            labels = classify_image(client, to_data_url(jpeg), args.model, args.timeout)
        except Exception as exc:
            error = str(exc)

        elapsed = time.perf_counter() - t1
        if error:
            print(f"ERROR ({elapsed:.1f}s): {error}", flush=True)
        else:
            h = labels.get("has_human", "?")
            a = labels.get("has_animal", "?")
            f = labels.get("has_flower", "?")
            print(f"done ({elapsed:.1f}s) - human:{h} animal:{a} flower:{f}", flush=True)

        results.append(
            {
                "index": idx,
                "artist": sample["artist"],
                "genre": sample["genre"],
                "style": sample["style"],
                "image_path": rel_path,
                "labels": labels,
                "error": error,
                "elapsed_seconds": round(elapsed, 3),
            }
        )

        if args.delay > 0 and i < len(samples) - 1:
            time.sleep(args.delay)

    write_html(results, args.output_html, args.model)
    Path(args.output_json).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    success = sum(1 for r in results if not r.get("error"))
    print(f"\nResults: {success}/{len(results)} succeeded")
    print(f"Wrote: {args.output_html}, {args.output_json}, {args.image_dir}/")


if __name__ == "__main__":
    main()
