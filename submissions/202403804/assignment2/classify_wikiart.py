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
        "For evidence, write one concise natural-language sentence explaining the yes/no labels. "
        "Mention approximate positions such as center, top-left, lower-right, foreground, or background when relevant."
    )
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }
    response = client.chat.completions.create(
        model=model,
        messages=[user_message],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "wikiart_presence_labels",
                "strict": True,
                "schema": SCHEMA,
            },
        },
        extra_body={
            "reasoning": {"enabled": True},
            "plugins": [{"id": "response-healing"}],
        },
        extra_headers={"X-Title": "HUFS-LAI-MM WikiArt Assignment"},
        timeout=request_timeout,
    )
    message = response.choices[0].message
    content = message.content
    if content is None and getattr(message, "reasoning_details", None):
        follow_up = client.chat.completions.create(
            model=model,
            messages=[
                user_message,
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_details": message.reasoning_details,
                },
                {
                    "role": "user",
                    "content": (
                        "Now provide the final answer. Return only the JSON object required by the schema."
                    ),
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
            extra_body={"reasoning": {"enabled": True}, "plugins": [{"id": "response-healing"}]},
            extra_headers={"X-Title": "HUFS-LAI-MM WikiArt Assignment"},
            timeout=request_timeout,
        )
        message = follow_up.choices[0].message
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
    sorted_results = sorted(results, key=lambda item: item["index"])
    total = len(sorted_results)
    human_count = sum(1 for row in sorted_results if row.get("labels", {}).get("has_human") == "yes")
    animal_count = sum(1 for row in sorted_results if row.get("labels", {}).get("has_animal") == "yes")
    flower_count = sum(1 for row in sorted_results if row.get("labels", {}).get("has_flower") == "yes")
    error_count = sum(1 for row in sorted_results if row.get("error"))

    summary_tiles = "".join(
        f"""
        <section class="metric">
          <span>{html.escape(label)}</span>
          <strong>{value}</strong>
        </section>
        """
        for label, value in [
            ("Samples", total),
            ("Human yes", human_count),
            ("Animal yes", animal_count),
            ("Flower yes", flower_count),
            ("Errors", error_count),
        ]
    )

    cards = []
    for row in sorted(results, key=lambda item: item["index"]):
        labels = row["labels"]
        error = row.get("error", "")
        if error:
            badges = (
                "<span class='badge error'>human: error</span>"
                "<span class='badge error'>animal: error</span>"
                "<span class='badge error'>flower: error</span>"
            )
            evidence = html.escape(error)
        else:
            badges = "".join(
                f"<span class='badge {html.escape(value)}'>{html.escape(name)}: {html.escape(value)}</span>"
                for name, value in [
                    ("human", labels["has_human"]),
                    ("animal", labels["has_animal"]),
                    ("flower", labels["has_flower"]),
                ]
            )
            evidence = html.escape(labels.get("evidence", ""))
        cards.append(
            f"""
            <article class="result">
              <img src="{html.escape(row["image_path"])}" alt="sample {row["index"]}">
              <div class="result-body">
                <div class="result-topline">
                  <h2>Sample {row["index"]:02d}</h2>
                  <span class="meta">artist {html.escape(str(row["artist"]))} / genre {html.escape(str(row["genre"]))} / style {html.escape(str(row["style"]))}</span>
                </div>
                <div class="badges">{badges}</div>
                <p>{evidence}</p>
              </div>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WikiArt Classification Results</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f6f3ee;
      color: #1d2528;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 36px 28px 48px;
    }}
    header {{
      border-top: 8px solid #0e6f6a;
      padding: 24px 0 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 36px;
      font-weight: 750;
      letter-spacing: 0;
    }}
    .subtitle {{
      max-width: 820px;
      margin: 10px 0 0;
      color: #536064;
      font-size: 15px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin: 18px 0 26px;
    }}
    .metric {{
      background: #ffffff;
      border: 1px solid #ded8cf;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric span {{
      display: block;
      color: #697478;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .metric strong {{
      display: block;
      margin-top: 5px;
      font-size: 26px;
    }}
    .results {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .result {{
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 16px;
      background: #ffffff;
      border: 1px solid #ded8cf;
      border-radius: 8px;
      padding: 14px;
      break-inside: avoid;
    }}
    .result img {{
      width: 150px;
      height: 150px;
      object-fit: contain;
      background: #ebe6dd;
      border-radius: 6px;
    }}
    .result h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .result-topline {{
      display: grid;
      gap: 3px;
    }}
    .meta {{
      color: #6f5d50;
      font-size: 12px;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 12px 0 10px;
    }}
    .badge {{
      border-radius: 999px;
      display: inline-block;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 9px;
      text-transform: uppercase;
    }}
    .yes {{
      background: #d9f0e8;
      color: #0b5c4a;
    }}
    .no {{
      background: #f3ddd8;
      color: #8a3528;
    }}
    .error {{
      background: #fff0be;
      color: #775200;
    }}
    .result p {{
      margin: 0;
      color: #2e383b;
      font-size: 14px;
    }}
    @media (max-width: 860px) {{
      main {{ padding: 26px 18px 36px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .results {{ grid-template-columns: 1fr; }}
      .result {{ grid-template-columns: 128px minmax(0, 1fr); }}
      .result img {{ width: 128px; height: 128px; }}
    }}
    @media print {{
      body {{ background: #ffffff; }}
      main {{ max-width: none; padding: 20px; }}
      .results {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .result {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>WikiArt Classification Results</h1>
      <p class="subtitle">Schema-guided OpenRouter classification for 20 WikiArt samples. Model: {html.escape(model)}</p>
    </header>
    <section class="metrics">
      {summary_tiles}
    </section>
    <section class="results">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    Path(output_path).write_text(document, encoding="utf-8")


def write_outputs(results, output_html, output_json, model):
    write_html(results, output_html, model)
    if output_json:
        Path(output_json).write_text(
            json.dumps(sorted(results, key=lambda item: item["index"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def read_existing_results(output_json):
    if not output_json:
        return []
    path = Path(output_json)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output-html", default="results.html")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_env()
    token = os.environ.get("OPENROUTER_TOKEN")
    if not token:
        raise RuntimeError(
            "OPENROUTER_TOKEN is missing. Put it in .env or set it as an environment variable."
        )
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    load_started_at = time.perf_counter()
    samples = list(get_samples(args.limit))
    results = read_existing_results(args.output_json) if args.resume else []
    done_indexes = {row["index"] for row in results}
    if done_indexes:
        samples = [sample for sample in samples if sample["index"] not in done_indexes]
        print(f"resuming with {len(done_indexes)} existing results; {len(samples)} samples remaining", flush=True)
    load_elapsed = time.perf_counter() - load_started_at
    print(f"loaded {len(samples)} samples in {load_elapsed:.1f}s", flush=True)

    classify_started_at = time.perf_counter()

    if args.workers <= 1:
        for sample in samples:
            results.append(
                process_sample(sample, token, args.model, image_dir, args.max_size, args.request_timeout)
            )
            write_outputs(results, args.output_html, args.output_json, args.model)
            time.sleep(args.sleep)
    else:
        results.extend(
            process_samples_parallel(
                samples,
                token,
                args.model,
                image_dir,
                args.max_size,
                args.request_timeout,
                args.workers,
                args.sleep,
            )
        )

    classify_elapsed = time.perf_counter() - classify_started_at
    write_outputs(results, args.output_html, args.output_json, args.model)
    if args.output_json:
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
