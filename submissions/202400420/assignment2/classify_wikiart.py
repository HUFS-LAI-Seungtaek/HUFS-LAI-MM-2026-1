"""WikiArt image classifier using OpenRouter API with schema-guided decoding."""

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

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the artwork visibly contains a human or person figure.",
        },
        "has_animal": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the artwork visibly contains any animal (mammal, bird, fish, insect, etc.).",
        },
        "has_flower": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Whether the artwork visibly contains a flower or flowers.",
        },
        "evidence": {
            "type": "string",
            "description": (
                "One concise sentence explaining each yes/no decision. "
                "Include location cues like center, top-left, foreground, or background where helpful."
            ),
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "evidence"],
    "additionalProperties": False,
}

PROMPT = (
    "Examine this painting and answer three yes/no questions about its visible content:\n"
    "1. has_human: Does it depict any human or person (including partial figures, faces, or crowds)?\n"
    "2. has_animal: Does it depict any animal (mammal, bird, fish, insect, reptile, or mythological creature)?\n"
    "3. has_flower: Does it depict any flower or flowers?\n\n"
    "Important:\n"
    "- Judge only what is visibly present, not the painting style or genre.\n"
    "- A partial body part or silhouette counts as a human.\n"
    "- Return the JSON object specified by the schema.\n"
    "- In evidence, briefly explain your yes/no choices and mention approximate positions "
    "(e.g. center, top-left, foreground) when relevant."
)


def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def image_to_jpeg_bytes(image: Image.Image, max_size: int) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def classify_image(client: OpenAI, jpeg_bytes: bytes, model: str, timeout: float) -> dict:
    encoded = base64.b64encode(jpeg_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

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

    message = response.choices[0].message
    content = message.content
    if isinstance(content, dict):
        return content
    if content is None and hasattr(message, "parsed") and message.parsed is not None:
        return message.parsed
    if content is None:
        raise RuntimeError(f"Empty content from model. Raw: {message}")
    return json.loads(content)


def get_samples(limit: int):
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


def write_html(results: list, output_path: str, model: str) -> None:
    rows = []
    for row in sorted(results, key=lambda r: r["index"]):
        labels = row["labels"]
        error = row.get("error", "")
        img_src = row["image_path"]

        if error:
            label_cells = "<td class='error'>error</td><td class='error'>error</td><td class='error'>error</td>"
            evidence_cell = f"<td class='error'>{html.escape(error)}</td>"
        else:
            label_cells = (
                f"<td class=\"{labels['has_human']}\">{labels['has_human']}</td>"
                f"<td class=\"{labels['has_animal']}\">{labels['has_animal']}</td>"
                f"<td class=\"{labels['has_flower']}\">{labels['has_flower']}</td>"
            )
            evidence_cell = f"<td>{html.escape(labels.get('evidence', ''))}</td>"

        rows.append(
            f"""
            <tr>
              <td>{row["index"]}</td>
              <td><img src="{html.escape(img_src)}" alt="sample {row["index"]}"></td>
              <td>
                <div><strong>Artist:</strong> {html.escape(str(row["artist"]))}</div>
                <div><strong>Genre:</strong> {html.escape(str(row["genre"]))}</div>
                <div><strong>Style:</strong> {html.escape(str(row["style"]))}</div>
              </td>
              {label_cells}
              {evidence_cell}
            </tr>"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WikiArt Classification Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
    h1 {{ margin-bottom: 4px; }}
    p.meta {{ color: #555; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f0f0; position: sticky; top: 0; z-index: 1; }}
    img {{ max-width: 160px; max-height: 160px; display: block; }}
    .yes {{ background: #d4edda; font-weight: bold; text-align: center; color: #155724; }}
    .no  {{ background: #f8d7da; font-weight: bold; text-align: center; color: #721c24; }}
    .error {{ background: #fff3cd; color: #664d03; }}
  </style>
</head>
<body>
  <h1>WikiArt Classification Results</h1>
  <p class="meta">Model: {html.escape(model)} &nbsp;|&nbsp; {len(results)} samples</p>
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
    Path(output_path).write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify WikiArt images with OpenRouter.")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to classify")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between requests")
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts on failure")
    args = parser.parse_args()

    load_env()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not token:
        raise RuntimeError("Set OPENROUTER_TOKEN in .env or environment.")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=token)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.limit} samples from WikiArt...", flush=True)
    t0 = time.perf_counter()
    samples = list(get_samples(args.limit))
    print(f"Loaded {len(samples)} samples in {time.perf_counter() - t0:.1f}s", flush=True)

    results = []
    for sample in samples:
        idx = sample["index"]
        print(f"[{idx + 1}/{len(samples)}] classifying sample {idx}...", flush=True, end=" ")

        jpeg_bytes = image_to_jpeg_bytes(sample["image"], max_size=args.max_size)
        image_path = image_dir / f"sample_{idx:03d}.jpg"
        image_path.write_bytes(jpeg_bytes)

        labels, error = {}, ""
        for attempt in range(args.retries + 1):
            try:
                t_start = time.perf_counter()
                labels = classify_image(client, jpeg_bytes, args.model, args.timeout)
                elapsed = time.perf_counter() - t_start
                print(f"done ({elapsed:.1f}s)", flush=True)
                break
            except Exception as exc:
                error = str(exc)
                if attempt < args.retries:
                    print(f"retry {attempt + 1}...", flush=True, end=" ")
                    time.sleep(2)
                else:
                    print(f"failed: {error}", flush=True)

        results.append({
            "index": idx,
            "artist": sample["artist"],
            "genre": sample["genre"],
            "style": sample["style"],
            "image_path": image_path.as_posix(),
            "labels": labels,
            "error": error,
        })

        if idx < len(samples) - 1:
            time.sleep(args.sleep)

    write_html(results, args.output_html, args.model)
    Path(args.output_json).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    success = sum(1 for r in results if not r["error"])
    print(f"\nDone. {success}/{len(results)} succeeded.")
    print(f"Wrote {args.output_html} and {args.output_json}")


if __name__ == "__main__":
    main()
