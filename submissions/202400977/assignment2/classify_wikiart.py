import argparse
import base64
import html
import json
import os
import time
from io import BytesIO
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from tqdm import tqdm


MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "has_human": {"type": "boolean"},
        "has_animal": {"type": "boolean"},
        "has_flower": {"type": "boolean"},
        "brief_reason": {
            "type": "string",
            "description": "One concise sentence explaining the visible evidence.",
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "brief_reason"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify WikiArt samples with OpenRouter schema-guided decoding."
    )
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between API calls.")
    return parser.parse_args()


def ensure_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def image_to_data_url(image_path):
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def save_dataset_samples(output_dir, sample_count, seed_offset):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset("huggan/wikiart", split="train", streaming=True)
    samples = []

    for index, item in enumerate(stream.skip(seed_offset).take(sample_count)):
        image = ensure_rgb(item["image"])
        filename = f"wikiart_{index + 1:02d}.jpg"
        path = images_dir / filename
        image.thumbnail((900, 900))
        image.save(path, format="JPEG", quality=92)

        metadata = {
            key: value
            for key, value in item.items()
            if key != "image" and isinstance(value, (str, int, float, bool))
        }
        samples.append({"index": index + 1, "path": path, "metadata": metadata})

    return samples


def classify_image(api_key, image_path, retries=3):
    prompt = (
        "Look only at the visible image. Classify whether it contains any human figure, "
        "any animal, and any flower or floral motif. Return JSON that exactly matches "
        "the provided schema."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "reasoning": {"exclude": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "wikiart_visual_labels",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "HUFS-LAI-MM-2026-1 Assignment 2",
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"].get("content")
            if not content:
                raise RuntimeError(f"Empty model content: {json.dumps(body)[:500]}")
            return parse_schema_json(content)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"OpenRouter classification failed for {image_path}: {last_error}")


def parse_schema_json(content):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])

    return {
        "has_human": bool(parsed["has_human"]),
        "has_animal": bool(parsed["has_animal"]),
        "has_flower": bool(parsed["has_flower"]),
        "brief_reason": str(parsed["brief_reason"]),
    }


def render_html(output_dir, rows):
    html_path = output_dir / "results.html"
    cards = []
    for row in rows:
        metadata = "<br>".join(
            f"<strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}"
            for key, value in row["metadata"].items()
        )
        result = row["result"]
        cards.append(
            f"""
            <section class="card">
              <img src="images/{html.escape(row['path'].name)}" alt="WikiArt sample {row['index']}">
              <div class="content">
                <h2>Sample {row['index']}</h2>
                <dl>
                  <dt>has_human</dt><dd>{result['has_human']}</dd>
                  <dt>has_animal</dt><dd>{result['has_animal']}</dd>
                  <dt>has_flower</dt><dd>{result['has_flower']}</dd>
                </dl>
                <p>{html.escape(result['brief_reason'])}</p>
                <p class="meta">{metadata}</p>
              </div>
            </section>
            """
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WikiArt Visual Classification</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ margin-top: 0; color: #52606d; }}
    .card {{ display: grid; grid-template-columns: 220px 1fr; gap: 18px; padding: 18px 0; border-top: 1px solid #d9e2ec; page-break-inside: avoid; }}
    img {{ width: 220px; max-height: 220px; object-fit: contain; background: #f5f7fa; }}
    h2 {{ margin: 0 0 8px; font-size: 20px; }}
    dl {{ display: grid; grid-template-columns: 110px 1fr; gap: 6px 12px; margin: 0 0 10px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; }}
    .meta {{ color: #616e7c; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>WikiArt Visual Classification</h1>
  <p class="subtitle">Dataset: huggan/wikiart | Model: {MODEL}</p>
  {''.join(cards)}
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def render_pdf(output_dir, rows):
    pdf_path = output_dir / "results.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("WikiArt Visual Classification", styles["Title"]),
        Paragraph(f"Dataset: huggan/wikiart<br/>Model: {MODEL}", styles["BodyText"]),
        Spacer(1, 0.2 * inch),
    ]

    for row in rows:
        result = row["result"]
        labels = (
            f"<b>has_human:</b> {result['has_human']}<br/>"
            f"<b>has_animal:</b> {result['has_animal']}<br/>"
            f"<b>has_flower:</b> {result['has_flower']}<br/><br/>"
            f"{html.escape(result['brief_reason'])}"
        )
        image = PdfImage(str(row["path"]), width=1.8 * inch, height=1.8 * inch)
        table = Table(
            [[image, Paragraph(f"<b>Sample {row['index']}</b><br/>{labels}", styles["BodyText"])]],
            colWidths=[2.0 * inch, 4.8 * inch],
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend([table, Spacer(1, 0.16 * inch)])

    doc.build(story)
    return pdf_path


def main():
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running this script.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = save_dataset_samples(args.output_dir, args.sample_count, args.seed_offset)

    rows = []
    for sample in tqdm(samples, desc="Classifying WikiArt samples"):
        result = classify_image(api_key, sample["path"])
        rows.append({**sample, "result": result})
        time.sleep(args.sleep)

    html_path = render_html(args.output_dir, rows)
    pdf_path = render_pdf(args.output_dir, rows)
    print(f"Saved {html_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
