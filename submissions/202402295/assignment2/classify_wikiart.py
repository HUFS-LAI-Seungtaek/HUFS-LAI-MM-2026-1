#!/usr/bin/env python3
"""
Schema-guided WikiArt image classification with OpenRouter.

The script classifies each artwork image into three yes/no labels:
has_human, has_animal, and has_flower. It constrains model output with
JSON Schema, stores short location-aware rationales, and writes one
self-contained HTML report.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
DEFAULT_IMAGE_DIR = "images"
DEFAULT_OUTPUT = "results.html"
DEFAULT_JSON_OUTPUT = "results.json"
DEFAULT_IMAGE_SIZE = 96
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "wikiart_schema_guided_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "has_human": {"type": "string", "enum": ["yes", "no"]},
            "has_animal": {"type": "string", "enum": ["yes", "no"]},
            "has_flower": {"type": "string", "enum": ["yes", "no"]},
            "evidence": {
                "type": "string",
                "description": "One short natural-language evidence sentence with image location wording.",
            },
        },
        "required": [
            "has_human",
            "has_animal",
            "has_flower",
            "evidence",
        ],
    },
}


PROMPT = """You are classifying one WikiArt artwork image.

Your task is to decide whether the image visibly contains:
1. has_human: a person, face, body, portrait, human-like statue, or clearly human silhouette.
2. has_animal: an animal, bird, fish, insect, horse, dog, cat, mythological animal-like creature, or clear animal figure.
3. has_flower: a flower blossom, bouquet, flowering plant, floral still-life element, or clear flower motif.

Return exactly one JSON object following the provided JSON Schema.
Each has_* value must be either "yes" or "no".

For evidence:
- Write one short natural-language sentence explaining the three yes/no labels.
- Include a location expression such as "center", "top-left", "lower-right", "foreground", "background", or "near the edge".
- If a subject is not visible, say that briefly.
"""


@dataclass
class Result:
    image_path: Path
    classification: dict[str, str] | None
    error: str | None = None


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify WikiArt images using OpenRouter JSON Schema output."
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--skip-readme-update",
        action="store_true",
        help="Do not update README.md with interesting and confusing examples.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download sample images from huggan/wikiart before classification.",
    )
    parser.add_argument(
        "--download-count",
        type=int,
        default=20,
        help="Number of WikiArt images to save when --download is used.",
    )
    return parser.parse_args()


def maybe_download_wikiart(image_dir: Path, count: int) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: pip install -r requirements.txt"
        ) from exc

    image_dir.mkdir(parents=True, exist_ok=True)
    existing = list_images(image_dir, limit=None)
    if len(existing) >= count:
        return

    dataset = load_dataset("huggan/wikiart", split=f"train[:{count}]")
    for index, row in enumerate(dataset, start=1):
        output_path = image_dir / f"wikiart_{index:03d}.jpg"
        if output_path.exists():
            continue
        row["image"].convert("RGB").save(output_path, format="JPEG", quality=92)


def list_images(image_dir: Path, limit: int | None) -> list[Path]:
    if not image_dir.exists():
        return []
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:limit] if limit else images


def image_data_url(image_path: Path, image_size: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((image_size, image_size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def call_openrouter(
    image_path: Path,
    *,
    api_key: str,
    model: str,
    timeout: int,
    image_size: int,
) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path, image_size)}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": RESPONSE_SCHEMA,
        },
        "plugins": [{"id": "response-healing"}],
        "reasoning": {"effort": "none", "exclude": True},
        "max_tokens": 260,
        "temperature": 0,
    }
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sungyh03/HUFS-LAI-MM-2026-1",
            "X-Title": "HUFS LAI MM 2026 Assignment 2",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    data = json.loads(content)
    validate_classification(data)
    return data


def validate_classification(data: dict[str, Any]) -> None:
    required = {
        "has_human",
        "has_animal",
        "has_flower",
        "evidence",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    for key in ("has_human", "has_animal", "has_flower"):
        if data[key] not in {"yes", "no"}:
            raise ValueError(f"{key} must be yes/no, got {data[key]!r}")
    for key in ("evidence",):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} must be a non-empty string")


def badge(value: str | None) -> str:
    if value == "yes":
        return '<span class="badge yes">yes</span>'
    if value == "no":
        return '<span class="badge no">no</span>'
    return '<span class="badge error">error</span>'


def write_html(results: list[Result], output_path: Path, model: str) -> None:
    rows: list[str] = []
    for index, result in enumerate(results, start=1):
        image_name = html.escape(result.image_path.name)
        image_src = image_data_url(result.image_path, 320)
        if result.classification:
            item = result.classification
            status_cells = (
                f"<td>{badge(item['has_human'])}</td>"
                f"<td>{badge(item['has_animal'])}</td>"
                f"<td>{badge(item['has_flower'])}</td>"
            )
            evidence = f"""
                <p>{html.escape(item["evidence"])}</p>
            """
        else:
            status_cells = f"<td>{badge(None)}</td><td>{badge(None)}</td><td>{badge(None)}</td>"
            evidence = f'<p class="error-text">{html.escape(result.error or "Unknown error")}</p>'

        rows.append(
            f"""
            <tr>
                <td class="index">{index}</td>
                <td><img src="{image_src}" alt="{image_name}"></td>
                <td>
                    <div class="filename">{image_name}</div>
                    <div class="evidence">{evidence}</div>
                </td>
                {status_cells}
            </tr>
            """
        )

    ok_count = sum(1 for result in results if result.classification)
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Assignment 2 WikiArt Classification Results</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #65727f;
      --line: #d7dde5;
      --head: #eef2f6;
      --yes-text: #12613d;
      --yes-bg: #dff1e7;
      --no-text: #7c2d20;
      --no-bg: #f8ded8;
      --err-text: #8a1f3d;
      --err-bg: #f8dce6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    main {{ padding: 24px 32px 42px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--head);
      color: var(--muted);
      font-size: 13px;
    }}
    .index {{
      width: 44px;
      color: var(--muted);
      font-weight: 700;
    }}
    img {{
      display: block;
      width: 150px;
      max-height: 150px;
      object-fit: contain;
      border: 1px solid var(--line);
      background: #eef2f6;
    }}
    .filename {{
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 13px;
      word-break: break-word;
    }}
    .evidence p {{ margin: 4px 0; }}
    .badge {{
      display: inline-block;
      min-width: 46px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-align: center;
      text-transform: uppercase;
    }}
    .yes {{ color: var(--yes-text); background: var(--yes-bg); }}
    .no {{ color: var(--no-text); background: var(--no-bg); }}
    .error {{ color: var(--err-text); background: var(--err-bg); }}
    .error-text {{
      color: var(--err-text);
      white-space: pre-wrap;
    }}
    @media (max-width: 760px) {{
      header, main {{
        padding-left: 16px;
        padding-right: 16px;
      }}
      table, thead, tbody, tr, th, td {{
        display: block;
      }}
      thead {{ display: none; }}
      tr {{ padding: 12px 0; }}
      td {{ border-bottom: 0; }}
      img {{
        width: 100%;
        max-height: 280px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Assignment 2 WikiArt Classification Results</h1>
    <div class="meta">Model: {html.escape(model)} · Classified: {ok_count}/{len(results)} · Generated: {html.escape(generated_at)}</div>
  </header>
  <main>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Image</th>
          <th>Evidence</th>
          <th>Human</th>
          <th>Animal</th>
          <th>Flower</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


def write_json(results: list[Result], output_path: Path, model: str) -> None:
    payload = {
        "model": model,
        "results": [
            {
                "image": result.image_path.name,
                "classification": result.classification,
                "error": result.error,
            }
            for result in results
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_result(index: int, result: Result) -> str:
    if not result.classification:
        return f"이미지 #{index}: API 호출 또는 응답 처리 오류가 있어 분류 결과가 생성되지 않았다."
    item = result.classification
    labels = (
        f"human={item['has_human']}, animal={item['has_animal']}, "
        f"flower={item['has_flower']}"
    )
    reasons = item["evidence"].strip()
    return f"이미지 #{index} (`{result.image_path.name}`): {labels}. {reasons}"


def choose_interesting_cases(results: list[Result]) -> list[tuple[int, Result]]:
    scored: list[tuple[int, int, Result]] = []
    for index, result in enumerate(results, start=1):
        if not result.classification:
            continue
        item = result.classification
        yes_count = sum(
            item[key] == "yes" for key in ("has_human", "has_animal", "has_flower")
        )
        score = yes_count * 10
        if item["has_flower"] == "yes":
            score += 3
        if item["has_animal"] == "yes":
            score += 3
        if item["has_human"] == "yes":
            score += 2
        scored.append((score, index, result))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [(index, result) for _, index, result in scored[:3]]


def choose_confusing_cases(results: list[Result]) -> list[tuple[int, Result]]:
    cues = (
        "motif",
        "silhouette",
        "statue",
        "mythological",
        "abstract",
        "unclear",
        "not visible",
        "no clear",
        "background",
    )
    scored: list[tuple[int, int, Result]] = []
    for index, result in enumerate(results, start=1):
        if not result.classification:
            scored.append((100, index, result))
            continue
        text = result.classification["evidence"].lower()
        score = sum(cue in text for cue in cues)
        yes_count = sum(
            result.classification[key] == "yes"
            for key in ("has_human", "has_animal", "has_flower")
        )
        if yes_count == 0:
            score += 1
        scored.append((score, index, result))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [(index, result) for score, index, result in scored[:2] if score > 0]


def replace_section(readme: str, header: str, body: str) -> str:
    start = readme.find(header)
    if start == -1:
        return readme.rstrip() + "\n\n" + header + "\n\n" + body.rstrip() + "\n"
    body_start = start + len(header)
    next_header = readme.find("\n## ", body_start)
    if next_header == -1:
        return readme[:body_start].rstrip() + "\n\n" + body.rstrip() + "\n"
    return readme[:body_start].rstrip() + "\n\n" + body.rstrip() + "\n\n" + readme[next_header + 1 :]


def update_readme(results: list[Result], readme_path: Path) -> None:
    if not readme_path.exists():
        return
    readme = readme_path.read_text(encoding="utf-8")
    interesting = choose_interesting_cases(results)
    confusing = choose_confusing_cases(results)

    interesting_body = "\n".join(
        f"{number}. {summarize_result(index, result)}"
        for number, (index, result) in enumerate(interesting, start=1)
    )
    if not interesting_body:
        interesting_body = "1. 실행 결과에서 분류 성공 사례를 확인하지 못했다."

    confusing_body = "\n".join(
        f"{number}. {summarize_result(index, result)}"
        for number, (index, result) in enumerate(confusing, start=1)
    )
    if not confusing_body:
        confusing_body = "1. 실행 결과에서 특별히 애매한 사례를 찾지 못했다."

    readme = replace_section(
        readme,
        "## 결과에서 흥미로웠던 사례 3개",
        interesting_body,
    )
    readme = replace_section(
        readme,
        "## 모델이 헷갈렸거나 틀렸다고 생각되는 사례 2개",
        confusing_body,
    )
    readme_path.write_text(readme, encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    load_env(script_dir / ".env")

    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is missing. Create a local .env file first.", file=sys.stderr)
        return 2

    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = script_dir / image_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    json_output_path = Path(args.json_output)
    if not json_output_path.is_absolute():
        json_output_path = script_dir / json_output_path
    readme_path = Path(args.readme)
    if not readme_path.is_absolute():
        readme_path = script_dir / readme_path

    if args.download:
        print(f"Downloading {args.download_count} WikiArt images into {image_dir}")
        maybe_download_wikiart(image_dir, args.download_count)

    images = list_images(image_dir, args.limit)
    if not images:
        print(f"No images found in {image_dir}. Use --download or add files to images/.", file=sys.stderr)
        return 2

    results: list[Result] = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}")
        try:
            classification = call_openrouter(
                image_path,
                api_key=api_key,
                model=args.model,
                timeout=args.timeout,
                image_size=args.image_size,
            )
            results.append(Result(image_path=image_path, classification=classification))
        except Exception as exc:
            results.append(Result(image_path=image_path, classification=None, error=str(exc)))
            print(f"  error: {exc}", file=sys.stderr)
        if args.sleep and index < len(images):
            time.sleep(args.sleep)

    write_html(results, output_path, args.model)
    write_json(results, json_output_path, args.model)
    if not args.skip_readme_update:
        update_readme(results, readme_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {json_output_path}")
    if not args.skip_readme_update:
        print(f"Updated {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
