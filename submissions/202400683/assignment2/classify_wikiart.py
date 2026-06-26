import argparse
import base64
import html
import io
import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SCHEMA = {
    "type": "object",
    "properties": {
        "has_human": {"type": "string", "enum": ["yes", "no"]},
        "has_animal": {"type": "string", "enum": ["yes", "no"]},
        "has_flower": {"type": "string", "enum": ["yes", "no"]},
        "evidence": {
            "type": "string",
            "description": (
                "A brief natural-language explanation. Mention approximate locations "
                "such as center, top-left, lower-right, foreground, or background when relevant."
            ),
        },
    },
    "required": ["has_human", "has_animal", "has_flower", "evidence"],
    "additionalProperties": False,
}


def image_to_jpeg_bytes(image, max_size):
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def classify_image(client, image_data_url, model, request_timeout):
    prompt = (
        "Look at this artwork image and decide whether it visibly contains each of these object types:\n"
        "- has_human (yes/no): Any visible human figure, person, portrait subject, crowd, or silhouette.\n"
        "- has_animal (yes/no): Any visible animal, bird, fish, insect, or other non-human creature (painted or real).\n"
        "- has_flower (yes/no): Any visible flowers, whether in a vase, garden, landscape background, or held by a person.\n\n"
        "Set has_human, has_animal, and has_flower to yes or no.\n"
        "Do not classify the artwork style, genre, or medium.\n"
        "Return only the JSON object required by the schema.\n"
        "For evidence, write one concise natural-language sentence explaining the yes/no labels. "
        "Mention approximate positions such as center, top-left, lower-right, foreground, or background when relevant."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "wikiart_presence_labels",
                "strict": True,
                "schema": SCHEMA,
            },
        },
        extra_body={
            "reasoning": {"effort": "none", "exclude": True},
            "plugins": [{"id": "response-healing"}],
        },
        extra_headers={"X-Title": "HUFS-LAI-MM WikiArt Assignment"},
        timeout=request_timeout,
    )
    message = response.choices[0].message
    content = message.content
    if isinstance(content, dict):
        return content
    if content is None and hasattr(message, "parsed") and message.parsed is not None:
        return message.parsed
    if content is None:
        raise RuntimeError(f"Model returned empty content. Raw message: {message}")
    return json.loads(content)


def get_samples(limit):
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    for index, item in enumerate(dataset.take(limit)):
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        yield {
            "index": index,
            "artist": item.get("artist", ""),
            "genre": item.get("genre", ""),
            "style": item.get("style", ""),
            "image": image,
        }


def write_html(results, output_path, model):
    rows = []
    for row in sorted(results, key=lambda item: item["index"]):
        labels = row["labels"]
        error = row.get("error", "")
        time_sec = f"{row.get('elapsed_seconds', 0.0):.2f}s"
        
        if error:
            label_cells = "<td class='error'>error</td><td class='error'>error</td><td class='error'>error</td>"
            evidence_cell = f"<td class='error'>{html.escape(error)}</td>"
            time_cell = f"<td class='error'>{time_sec}</td>"
        else:
            label_cells = (
                f"<td class=\"{labels['has_human']}\">{labels['has_human']}</td>"
                f"<td class=\"{labels['has_animal']}\">{labels['has_animal']}</td>"
                f"<td class=\"{labels['has_flower']}\">{labels['has_flower']}</td>"
            )
            evidence_cell = f"<td>{html.escape(labels.get('evidence', ''))}</td>"
            time_cell = f"<td>{time_sec}</td>"
            
        rows.append(
            f"""
            <tr>
              <td>{row["index"]}</td>
              <td><img src="{html.escape(row["image_path"])}" alt="sample {row["index"]}"></td>
              <td>
                <div><strong>Artist</strong>: {html.escape(str(row["artist"]))}</div>
                <div><strong>Genre</strong>: {html.escape(str(row["genre"]))}</div>
                <div><strong>Style</strong>: {html.escape(str(row["style"]))}</div>
              </td>
              {label_cells}
              {evidence_cell}
              {time_cell}
            </tr>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WikiArt Classification Results</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 24px; color: #333; background-color: #fafafa; }}
    h1 {{ margin-bottom: 4px; color: #111; }}
    .meta-info {{ margin-bottom: 20px; font-size: 0.95em; color: #666; }}
    table {{ border-collapse: collapse; width: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.1); background-color: #fff; }}
    th, td {{ border: 1px solid #e0e0e0; padding: 12px; vertical-align: top; text-align: left; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; font-weight: 600; color: #444; }}
    img {{ max-width: 160px; max-height: 160px; border-radius: 4px; border: 1px solid #eee; }}
    .yes {{ background: #e8f5e9; color: #2e7d32; font-weight: bold; text-align: center; }}
    .no {{ background: #ffebee; color: #c62828; font-weight: bold; text-align: center; }}
    .error {{ background: #fff8e1; color: #b78103; font-style: italic; }}
    tr:hover {{ background-color: #fcfcfc; }}
  </style>
</head>
<body>
  <h1>WikiArt Classification Results</h1>
  <div class="meta-info">
    <p><strong>Model:</strong> {html.escape(model)}</p>
    <p><strong>Total Samples:</strong> {len(results)}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th style="width: 50px;">#</th>
        <th style="width: 180px;">Image</th>
        <th style="width: 220px;">Metadata</th>
        <th style="width: 100px; text-align: center;">Human</th>
        <th style="width: 100px; text-align: center;">Animal</th>
        <th style="width: 100px; text-align: center;">Flower</th>
        <th>Evidence</th>
        <th style="width: 100px;">Inference Time</th>
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


def write_pdf(html_path, pdf_path):
    try:
        from xhtml2pdf import pisa
    except ImportError:
        print("Warning: xhtml2pdf is not installed. Skipping PDF generation.", flush=True)
        return False
    
    html_content = Path(html_path).read_text(encoding="utf-8")
    
    # Inject landscape page size and compact styles for xhtml2pdf
    pdf_style = """
    <style>
      @page {
        size: a4 landscape;
        margin: 0.5in;
      }
      body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 9pt;
      }
      table {
        width: 100%;
      }
      th, td {
        padding: 5px;
        font-size: 8pt;
      }
      img {
        max-width: 100px;
        max-height: 100px;
      }
    </style>
    """
    
    html_content = html_content.replace("</head>", f"{pdf_style}</head>")
    html_content = re.sub(r'style="width:\s*\d+px;[^"]*"', '', html_content)
    html_content = html_content.replace('box-shadow: 0 1px 3px rgba(0,0,0,0.1);', '')
    
    original_cwd = os.getcwd()
    os.chdir(Path(html_path).parent)
    
    try:
        with open(Path(pdf_path).name, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            if pisa_status.err:
                print(f"Error converting HTML to PDF: {pisa_status.err}", flush=True)
                return False
            print(f"Wrote {pdf_path}", flush=True)
            return True
    except Exception as e:
        print(f"Failed to generate PDF: {e}", flush=True)
        return False
    finally:
        os.chdir(original_cwd)


def process_sample(sample, token, model, image_dir, max_size, request_timeout):
    sample_started_at = time.perf_counter()
    index = sample["index"]
    print(f"Classifying sample {index}...", flush=True)

    jpeg_bytes = image_to_jpeg_bytes(sample["image"], max_size=max_size)
    image_path = image_dir / f"sample_{index:03d}.jpg"
    image_path.write_bytes(jpeg_bytes)
    
    encoded = base64.b64encode(jpeg_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{encoded}"

    error = ""
    try:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=token)
        labels = classify_image(client, image_data_url, model, request_timeout)
    except Exception as exc:
        labels = {}
        error = str(exc)

    elapsed = time.perf_counter() - sample_started_at
    if error:
        print(f"Failed sample {index} in {elapsed:.1f}s: {error}", flush=True)
    else:
        print(f"Done sample {index} in {elapsed:.1f}s", flush=True)

    return {
        "index": index,
        "artist": sample["artist"],
        "genre": sample["genre"],
        "style": sample["style"],
        "image_path": image_path.relative_to(image_path.parent.parent).as_posix(),
        "labels": labels,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
    }


def process_sample_worker(queue, sample, token, model, image_dir, max_size, request_timeout):
    queue.put(process_sample(sample, token, model, image_dir, max_size, request_timeout))


def timeout_result(sample, image_dir, max_size, started_at, request_timeout):
    index = sample["index"]
    image_path = image_dir / f"sample_{index:03d}.jpg"
    if not image_path.exists():
        image_path.write_bytes(image_to_jpeg_bytes(sample["image"], max_size=max_size))
    elapsed = time.perf_counter() - started_at
    error = f"Timed out after {request_timeout:.1f}s"
    print(f"Failed sample {index} in {elapsed:.1f}s: {error}", flush=True)
    return {
        "index": index,
        "artist": sample["artist"],
        "genre": sample["genre"],
        "style": sample["style"],
        "image_path": image_path.relative_to(image_path.parent.parent).as_posix(),
        "labels": {},
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
    }


def process_samples_parallel(samples, token, model, image_dir, max_size, request_timeout, workers, sleep):
    results = []
    result_queue = mp.Queue()
    pending = []
    sample_iter = iter(samples)

    def start_next():
        try:
            sample = next(sample_iter)
        except StopIteration:
            return False
        process = mp.Process(
            target=process_sample_worker,
            args=(result_queue, sample, token, model, image_dir, max_size, request_timeout),
        )
        process.start()
        pending.append({"process": process, "sample": sample, "started_at": time.perf_counter()})
        time.sleep(sleep)
        return True

    # Start initial workers
    for _ in range(workers):
        if not start_next():
            break

    while pending:
        # Collect results from queue
        while not result_queue.empty():
            results.append(result_queue.get())

        still_pending = []
        for item in pending:
            process = item["process"]
            process.join(timeout=0)
            if not process.is_alive():
                process.close()
                continue

            elapsed = time.perf_counter() - item["started_at"]
            if elapsed > request_timeout:
                print(f"Sample {item['sample']['index']} reached timeout {request_timeout}s. Terminating process...", flush=True)
                process.terminate()
                process.join(timeout=1)
                process.close()
                results.append(
                    timeout_result(item["sample"], image_dir, max_size, item["started_at"], request_timeout)
                )
            else:
                still_pending.append(item)
        pending = still_pending
        
        while len(pending) < workers:
            if not start_next():
                break
        time.sleep(0.1)

    while not result_queue.empty():
        results.append(result_queue.get())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not token:
        raise RuntimeError(
            "OPENROUTER_TOKEN is missing. Put it in .env or set it as an environment variable."
        )

    # Output paths should be relative to the script location
    script_dir = Path(__file__).resolve().parent
    image_dir = script_dir / args.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    
    output_html_path = script_dir / args.output_html
    output_json_path = script_dir / args.output_json

    load_started_at = time.perf_counter()
    samples = list(get_samples(args.limit))
    load_elapsed = time.perf_counter() - load_started_at
    print(f"Loaded {len(samples)} samples in {load_elapsed:.1f}s", flush=True)

    classify_started_at = time.perf_counter()
    results = []

    if args.workers <= 1:
        for sample in samples:
            results.append(
                process_sample(sample, token, args.model, image_dir, args.max_size, args.request_timeout)
            )
            time.sleep(args.sleep)
    else:
        results = process_samples_parallel(
            samples,
            token,
            args.model,
            image_dir,
            args.max_size,
            args.request_timeout,
            args.workers,
            args.sleep,
        )

    classify_elapsed = time.perf_counter() - classify_started_at
    write_html(results, output_html_path, args.model)
    
    # Write PDF copy
    output_pdf_path = output_html_path.with_suffix(".pdf")
    write_pdf(output_html_path, output_pdf_path)
    
    if args.output_json:
        output_json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote {output_html_path} and {output_json_path}")
    else:
        print(f"Wrote {output_html_path}")
        
    if results:
        print(
            f"Classified {len(results)} samples in {classify_elapsed:.1f}s "
            f"({classify_elapsed / len(results):.1f}s/sample)"
        )


if __name__ == "__main__":
    # Windows support for multiprocessing
    mp.freeze_support()
    main()
