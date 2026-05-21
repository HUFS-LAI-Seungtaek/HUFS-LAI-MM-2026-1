import argparse
import base64
import html
import io
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from PIL import Image


MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {"type": "string", "enum": ["yes", "no"]},
        "has_animal": {"type": "string", "enum": ["yes", "no"]},
        "has_flower": {"type": "string", "enum": ["yes", "no"]},
        "evidence": {
            "type": "string",
            "description": (
                "One short sentence explaining the visual evidence, including location "
                "phrases such as center, top-left, foreground, background, or lower-right."
            ),
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "evidence"],
    "additionalProperties": False,
}


def find_env_file() -> Path | None:
    """Find .env from the current directory or one of its parents."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def load_env() -> None:
    env_file = find_env_file()
    if env_file is None:
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def token_from_env() -> str:
    load_env()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not token:
        raise RuntimeError(
            "OPENROUTER_TOKEN is missing. Create a .env file with OPENROUTER_TOKEN=..."
        )
    return token


def image_to_jpeg_bytes(image: Image.Image, max_size: int) -> bytes:
    rgb = image.convert("RGB")
    rgb.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def image_to_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def stream_wikiart_samples(limit: int):
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    features = dataset.features

    def label_name(field: str, value):
        feature = features.get(field)
        if hasattr(feature, "int2str") and isinstance(value, int):
            return feature.int2str(value)
        return value

    for index, item in enumerate(dataset.take(limit)):
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        yield {
            "index": index,
            "artist": label_name("artist", item.get("artist", "")),
            "genre": label_name("genre", item.get("genre", "")),
            "style": label_name("style", item.get("style", "")),
            "image": image,
        }


def classify_image(client: OpenAI, model: str, data_url: str, timeout: float) -> dict:
    prompt = (
        "Classify the visible content of this WikiArt image. "
        "has_human means at least one visible person, body, face, or human-like figure. "
        "has_animal means at least one visible non-human animal, including birds, horses, dogs, "
        "fish, insects, or mythical animal-like creatures. "
        "has_flower means at least one visible flower, blossom, floral bouquet, or flower pattern. "
        "For every field, answer yes only when the object is visually present in the image. "
        "Return only a JSON object matching the schema. "
        "The evidence must be exactly one concise sentence and include a location expression "
        "such as center, top-left, foreground, background, upper-right, or lower-left."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "wikiart_presence_classification",
                "strict": True,
                "schema": RESULT_SCHEMA,
            },
        },
        extra_headers={
            "HTTP-Referer": "https://github.com/",
            "X-Title": "HUFS-LAI-MM-2026-1 Assignment 2",
        },
        timeout=timeout,
    )

    content = response.choices[0].message.content
    if isinstance(content, dict):
        return content
    if not content:
        raise RuntimeError("OpenRouter returned an empty response.")
    return json.loads(content)


def clean_error(exc: Exception) -> str:
    message = str(exc)
    if "free-models-per-day" in message:
        return "OpenRouter free-models-per-day rate limit exceeded."
    if "free-models-per-min" in message:
        return "OpenRouter free-models-per-min rate limit exceeded."
    if "Timed out after" in message:
        return message
    return message.split("\n", 1)[0][:300]


def classify_worker(connection, token: str, model: str, data_url: str, timeout: float) -> None:
    try:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=token)
        connection.send({"classification": classify_image(client, model, data_url, timeout), "error": ""})
    except Exception as exc:
        connection.send({"classification": {}, "error": clean_error(exc)})
    finally:
        connection.close()


def classify_image_in_process(token: str, model: str, data_url: str, timeout: float) -> tuple[dict, str]:
    parent_connection, child_connection = mp.Pipe(duplex=False)
    process = mp.Process(
        target=classify_worker,
        args=(child_connection, token, model, data_url, timeout),
    )
    process.start()
    child_connection.close()
    process.join(timeout + 5)

    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        process.close()
        parent_connection.close()
        return {}, f"Timed out after {timeout:.1f} seconds"

    try:
        result = parent_connection.recv() if parent_connection.poll() else {"classification": {}, "error": "No response"}
    except EOFError:
        result = {"classification": {}, "error": "No response"}
    parent_connection.close()
    process.close()
    return result["classification"], result["error"]


def save_json(results: list[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def save_html(results: list[dict], output_path: Path, model: str) -> None:
    rows = []
    for result in results:
        labels = result.get("classification", {})
        image_path = html.escape(result["image_path"])
        evidence = html.escape(labels.get("evidence", result.get("error", "")))
        rows.append(
            f"""
            <tr>
              <td>{result["index"]}</td>
              <td><img src="{image_path}" alt="WikiArt sample {result["index"]}"></td>
              <td>
                <div><b>Artist</b>: {html.escape(str(result.get("artist", "")))}</div>
                <div><b>Genre</b>: {html.escape(str(result.get("genre", "")))}</div>
                <div><b>Style</b>: {html.escape(str(result.get("style", "")))}</div>
              </td>
              <td class="{html.escape(labels.get("has_human", "error"))}">{html.escape(labels.get("has_human", "error"))}</td>
              <td class="{html.escape(labels.get("has_animal", "error"))}">{html.escape(labels.get("has_animal", "error"))}</td>
              <td class="{html.escape(labels.get("has_flower", "error"))}">{html.escape(labels.get("has_flower", "error"))}</td>
              <td>{evidence}</td>
            </tr>
            """
        )

    yes_count = {
        key: sum(1 for item in results if item.get("classification", {}).get(key) == "yes")
        for key in ["has_human", "has_animal", "has_flower"]
    }
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WikiArt Schema-Guided Classification</title>
  <style>
    body {{
      margin: 24px;
      color: #222;
      background: #fafafa;
      font-family: Arial, Helvetica, sans-serif;
    }}
    h1 {{ margin-bottom: 6px; }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0;
    }}
    .pill {{
      border: 1px solid #d7d7d7;
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      vertical-align: top;
    }}
    th {{
      background: #efefef;
      position: sticky;
      top: 0;
    }}
    img {{
      display: block;
      max-width: 180px;
      max-height: 180px;
    }}
    td.yes {{ background: #e4f4e8; text-align: center; font-weight: 700; }}
    td.no {{ background: #f7e4e1; text-align: center; font-weight: 700; }}
    td.error {{ background: #fff0c2; text-align: center; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>WikiArt Schema-Guided Classification</h1>
  <p>Model: <code>{html.escape(model)}</code></p>
  <div class="summary">
    <div class="pill">Samples: {len(results)}</div>
    <div class="pill">Human yes: {yes_count["has_human"]}</div>
    <div class="pill">Animal yes: {yes_count["has_animal"]}</div>
    <div class="pill">Flower yes: {yes_count["has_flower"]}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Image</th>
        <th>Metadata</th>
        <th>has_human</th>
        <th>has_animal</th>
        <th>has_flower</th>
        <th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def save_pdf(results: list[dict], output_path: Path, model: str) -> None:
    from textwrap import wrap

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    page_width, page_height = A4
    margin = 18
    header_height = 72
    gap = 14
    card_width = (page_width - margin * 3) / 2
    card_height = (page_height - header_height - margin - gap * 2) / 3
    image_height = 132
    background = colors.HexColor("#0f1720")
    card_bg = colors.HexColor("#22303a")
    card_top = colors.HexColor("#16222b")
    text_main = colors.HexColor("#f3f0e8")
    text_muted = colors.HexColor("#a8b2b8")

    pdf = canvas.Canvas(str(output_path), pagesize=A4)

    def draw_header() -> None:
        pdf.setFillColor(background)
        pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        pdf.setFillColor(text_main)
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(margin, page_height - 34, "WikiArt Classification Results")
        pdf.setFillColor(text_muted)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(margin, page_height - 52, f"Model: {model}")

    def draw_pill(x: float, y: float, label: str, value: str) -> None:
        is_yes = value == "yes"
        pdf.setFillColor(colors.HexColor("#1f7a5a") if is_yes else colors.HexColor("#8a3b32"))
        pdf.roundRect(x, y, 72, 18, 9, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor("#c7f0dc") if is_yes else colors.HexColor("#ffd0c7"))
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawCentredString(x + 36, y + 5, f"{label}: {value}")

    def draw_wrapped_text(x: float, y: float, text: str, width_chars: int, max_lines: int) -> None:
        pdf.setFillColor(colors.HexColor("#e2ded3"))
        pdf.setFont("Helvetica", 8.3)
        line_y = y
        lines = wrap(text, width=width_chars)[:max_lines]
        if len(wrap(text, width=width_chars)) > max_lines and lines:
            lines[-1] = lines[-1].rstrip(".,;:") + "..."
        for line in lines:
            pdf.drawString(x, line_y, line)
            line_y -= 12

    def draw_card(result: dict, x: float, y: float) -> None:
        labels = result.get("classification", {})
        pdf.setFillColor(card_bg)
        pdf.roundRect(x, y, card_width, card_height, 10, fill=1, stroke=0)
        pdf.setFillColor(card_top)
        pdf.roundRect(x, y + card_height - image_height - 12, card_width, image_height + 12, 10, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        pdf.roundRect(x + 10, y + card_height - 25, 24, 18, 9, fill=1, stroke=0)
        pdf.setFillColor(text_main)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawCentredString(x + 22, y + card_height - 19, f"#{result['index']}")

        image_path = output_path.parent / result["image_path"]
        if image_path.exists():
            reader = ImageReader(str(image_path))
            image_width, image_original_height = reader.getSize()
            max_width = card_width - 44
            scale = min(max_width / image_width, image_height / image_original_height)
            draw_width = image_width * scale
            draw_height = image_original_height * scale
            image_x = x + (card_width - draw_width) / 2
            image_y = y + card_height - image_height - 8 + (image_height - draw_height) / 2
            pdf.drawImage(reader, image_x, image_y, draw_width, draw_height, preserveAspectRatio=True)

        pill_y = y + 68
        draw_pill(x + 10, pill_y, "Human", labels.get("has_human", "error"))
        draw_pill(x + 88, pill_y, "Animal", labels.get("has_animal", "error"))
        draw_pill(x + 166, pill_y, "Flower", labels.get("has_flower", "error"))
        evidence = labels.get("evidence") or result.get("error", "")
        draw_wrapped_text(x + 10, y + 50, evidence, width_chars=48, max_lines=4)

    draw_header()
    for position, result in enumerate(results):
        if position > 0 and position % 6 == 0:
            pdf.showPage()
            draw_header()
        slot = position % 6
        col = slot % 2
        row = slot // 2
        x = margin + col * (card_width + margin)
        y = page_height - header_height - (row + 1) * card_height - row * gap
        draw_card(result, x, y)

    pdf.save()


def load_completed_results(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        previous_results = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        item.get("index"): item
        for item in previous_results
        if isinstance(item, dict)
        and item.get("index") is not None
        and item.get("classification")
        and item.get("error", "") == ""
    }


def load_previous_results(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        previous_results = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        item.get("index"): item
        for item in previous_results
        if isinstance(item, dict) and item.get("index") is not None
    }


def result_with_current_sample(result: dict, sample: dict, image_file: Path) -> dict:
    updated = dict(result)
    updated.update(
        {
            "index": sample["index"],
            "artist": sample["artist"],
            "genre": sample["genre"],
            "style": sample["style"],
            "image_path": image_file.as_posix(),
        }
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify streamed WikiArt samples with OpenRouter.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Keep successful rows from results.json.")
    parser.add_argument("--max-new", type=int, default=None, help="Classify at most this many new/failed rows.")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--output-pdf", default="WikiArt Classification Results.pdf")
    args = parser.parse_args()

    token = token_from_env()

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_pdf = Path(args.output_pdf)
    previous = load_previous_results(output_json) if args.resume else {}
    completed = load_completed_results(output_json) if args.resume else {}
    results = []
    new_count = 0
    started_at = time.perf_counter()
    for sample in stream_wikiart_samples(args.limit):
        index = sample["index"]
        image_file = image_dir / f"wikiart_{index:03d}.jpg"
        jpeg_bytes = image_to_jpeg_bytes(sample["image"], args.max_size)
        image_file.write_bytes(jpeg_bytes)

        if index in completed:
            print(f"[{index + 1}/{args.limit}] keeping completed result for {image_file}", flush=True)
            results.append(result_with_current_sample(completed[index], sample, image_file))
            save_json(results, output_json)
            save_html(results, output_html, args.model)
            continue

        if args.max_new is not None and new_count >= args.max_new:
            print(f"[{index + 1}/{args.limit}] skipping new request for {image_file}", flush=True)
            skipped_result = previous.get(
                index,
                {
                    "index": index,
                    "artist": sample["artist"],
                    "genre": sample["genre"],
                    "style": sample["style"],
                    "image_path": image_file.as_posix(),
                    "classification": {},
                    "error": "Not attempted because --max-new was reached.",
                    "elapsed_seconds": 0,
                },
            )
            results.append(result_with_current_sample(skipped_result, sample, image_file))
            save_json(results, output_json)
            save_html(results, output_html, args.model)
            continue

        print(f"[{index + 1}/{args.limit}] classifying {image_file} ...", flush=True)
        new_count += 1
        item_started_at = time.perf_counter()
        classification = {}
        error = ""
        data_url = image_to_data_url(jpeg_bytes)
        for attempt in range(args.retries + 1):
            try:
                classification, error = classify_image_in_process(
                    token=token,
                    model=args.model,
                    data_url=data_url,
                    timeout=args.request_timeout,
                )
                if error:
                    raise RuntimeError(error)
                break
            except Exception as exc:
                error = clean_error(exc)
                if attempt < args.retries:
                    print(f"  retrying after error: {error}", flush=True)
                    time.sleep(args.sleep)

        results.append(
            {
                "index": index,
                "artist": sample["artist"],
                "genre": sample["genre"],
                "style": sample["style"],
                "image_path": image_file.as_posix(),
                "classification": classification,
                "error": error,
                "elapsed_seconds": round(time.perf_counter() - item_started_at, 3),
            }
        )
        save_json(results, output_json)
        save_html(results, output_html, args.model)
        time.sleep(args.sleep)

    save_pdf(results, output_pdf, args.model)
    print(
        f"Saved {args.output_json}, {args.output_html}, and {args.output_pdf} "
        f"in {time.perf_counter() - started_at:.1f}s"
    )


if __name__ == "__main__":
    main()
