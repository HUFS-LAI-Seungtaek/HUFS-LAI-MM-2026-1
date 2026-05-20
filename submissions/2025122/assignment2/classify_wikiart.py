import argparse
import base64
import html
import io
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

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

MOCK_LABELS = [
    {
        "has_human": "yes",
        "has_animal": "no",
        "has_flower": "no",
        "evidence": "A simplified human figure is visible in the center foreground, with no animals or flowers present.",
    },
    {
        "has_human": "no",
        "has_animal": "yes",
        "has_flower": "no",
        "evidence": "A simplified animal silhouette appears near the lower-right foreground, with no humans or flowers visible.",
    },
    {
        "has_human": "no",
        "has_animal": "no",
        "has_flower": "yes",
        "evidence": "A group of flower shapes is visible in the lower-left foreground, with no humans or animals present.",
    },
    {
        "has_human": "yes",
        "has_animal": "yes",
        "has_flower": "yes",
        "evidence": "A human figure stands near the center, an animal is on the right, and flowers appear in the lower-left foreground.",
    },
]


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


def image_to_jpeg_bytes(image, max_size):
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def image_to_data_url(image, max_size):
    encoded = base64.b64encode(image_to_jpeg_bytes(image, max_size=max_size)).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def classify_image(client, image_data_url, model, request_timeout):
    prompt = (
        "Look at this artwork image and decide whether it visibly contains each of these object types: "
        "a human/person, an animal, and a flower. "
        "Set has_human, has_animal, and has_flower to yes or no. "
        "Do not classify the artwork style, genre, or medium. "
        "Return only the JSON object required by the schema. "
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
    from datasets import load_dataset
    from PIL import Image

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


def create_mock_svg(index):
    human = ""
    animal = ""
    flowers = ""

    if index % 4 in (0, 3):
        human = """
        <circle cx="160" cy="78" r="22" fill="#734e3d"/>
        <rect x="148" y="98" width="24" height="60" fill="#3e6491"/>
        <path d="M148 115 L118 140 M172 115 L202 140" stroke="#3e6491" stroke-width="5"/>
        <path d="M152 158 L132 210 M168 158 L188 210" stroke="#2a425a" stroke-width="5"/>
        """

    if index % 4 in (1, 3):
        animal = """
        <ellipse cx="255" cy="168" rx="32" ry="23" fill="#5c5846"/>
        <ellipse cx="287" cy="148" rx="18" ry="17" fill="#5c5846"/>
        <path d="M292 133 L300 115 L304 138 Z" fill="#5c5846"/>
        <path d="M230 188 L222 215 M278 188 L286 215" stroke="#38342b" stroke-width="4"/>
        """

    if index % 4 in (2, 3):
        flowers = """
        <g stroke="#307045" stroke-width="4">
          <line x1="42" y1="198" x2="42" y2="222"/>
          <line x1="72" y1="186" x2="72" y2="222"/>
          <line x1="100" y1="204" x2="100" y2="222"/>
        </g>
        <circle cx="42" cy="180" r="13" fill="#c74557"/>
        <circle cx="72" cy="168" r="13" fill="#dfa740"/>
        <circle cx="100" cy="186" r="13" fill="#844c9a"/>
        <g fill="#edd757">
          <circle cx="42" cy="180" r="5"/>
          <circle cx="72" cy="168" r="5"/>
          <circle cx="100" cy="186" r="5"/>
        </g>
        """

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240">
  <rect width="320" height="165" fill="#c7d7e0"/>
  <rect y="165" width="320" height="75" fill="#b2bea4"/>
  <text x="12" y="28" font-family="Arial, sans-serif" font-size="16" fill="#30373d">mock sample {index}</text>
  {human}
  {animal}
  {flowers}
</svg>
"""


def get_mock_samples(limit):
    for index in range(limit):
        yield {
            "index": index,
            "artist": "mock-artist",
            "genre": "mock-genre",
            "style": "mock-style",
        }


def write_html(results, output_path, model):
    rows = []
    for row in sorted(results, key=lambda item: item["index"]):
        labels = row["labels"]
        error = row.get("error", "")
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
              <td><img src="{html.escape(row["image_path"])}" alt="sample {row["index"]}"></td>
              <td>
                <div><strong>Artist</strong>: {html.escape(str(row["artist"]))}</div>
                <div><strong>Genre</strong>: {html.escape(str(row["genre"]))}</div>
                <div><strong>Style</strong>: {html.escape(str(row["style"]))}</div>
              </td>
              {label_cells}
              {evidence_cell}
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
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f3f3; position: sticky; top: 0; }}
    img {{ max-width: 160px; max-height: 160px; }}
    .yes {{ background: #e7f7e7; font-weight: bold; text-align: center; }}
    .no {{ background: #f7e7e7; font-weight: bold; text-align: center; }}
    .error {{ background: #fff3cd; color: #664d03; }}
  </style>
</head>
<body>
  <h1>WikiArt Classification Results</h1>
  <p>Model: {html.escape(model)}</p>
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
</html>
"""
    Path(output_path).write_text(document, encoding="utf-8")


def process_sample(sample, token, model, image_dir, max_size, request_timeout):
    sample_started_at = time.perf_counter()
    index = sample["index"]
    print(f"classifying sample {index}...", flush=True)

    jpeg_bytes = image_to_jpeg_bytes(sample["image"], max_size=max_size)
    image_path = image_dir / f"sample_{index:03d}.jpg"
    image_path.write_bytes(jpeg_bytes)
    encoded = base64.b64encode(jpeg_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{encoded}"

    error = ""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=token)
        labels = classify_image(client, image_data_url, model, request_timeout)
    except Exception as exc:
        labels = {}
        error = str(exc)

    elapsed = time.perf_counter() - sample_started_at
    if error:
        print(f"failed sample {index} in {elapsed:.1f}s: {error}", flush=True)
    else:
        print(f"done sample {index} in {elapsed:.1f}s", flush=True)

    return {
        "index": index,
        "artist": sample["artist"],
        "genre": sample["genre"],
        "style": sample["style"],
        "image_path": image_path.as_posix(),
        "labels": labels,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
    }


def process_mock_sample(sample, image_dir, _max_size):
    index = sample["index"]
    image_path = image_dir / f"sample_{index:03d}.svg"
    image_path.write_text(create_mock_svg(index), encoding="utf-8")
    return {
        "index": index,
        "artist": sample["artist"],
        "genre": sample["genre"],
        "style": sample["style"],
        "image_path": image_path.as_posix(),
        "labels": MOCK_LABELS[index % len(MOCK_LABELS)],
        "error": "",
        "elapsed_seconds": 0.0,
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
    print(f"failed sample {index} in {elapsed:.1f}s: {error}", flush=True)
    return {
        "index": index,
        "artist": sample["artist"],
        "genre": sample["genre"],
        "style": sample["style"],
        "image_path": image_path.as_posix(),
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

    for _ in range(workers):
        if not start_next():
            break

    while pending:
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
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate deterministic mock images and labels without Hugging Face or OpenRouter calls.",
    )
    args = parser.parse_args()

    load_env()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not args.mock and not token:
        raise RuntimeError(
            "OPENROUTER_TOKEN is missing. Put it in .env or set it as an environment variable."
        )
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    load_started_at = time.perf_counter()
    samples = list(get_mock_samples(args.limit) if args.mock else get_samples(args.limit))
    load_elapsed = time.perf_counter() - load_started_at
    print(f"loaded {len(samples)} samples in {load_elapsed:.1f}s", flush=True)

    classify_started_at = time.perf_counter()
    results = []

    if args.mock:
        results = [process_mock_sample(sample, image_dir, args.max_size) for sample in samples]
    elif args.workers <= 1:
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
    write_html(results, args.output_html, args.model)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output_html} and {args.output_json}")
    else:
        print(f"wrote {args.output_html}")
    if results:
        print(
            f"classified {len(results)} samples in {classify_elapsed:.1f}s "
            f"({classify_elapsed / len(results):.1f}s/sample)"
        )


if __name__ == "__main__":
    main()
