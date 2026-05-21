# Assignment 2: WikiArt Visual Classification

## Used model

- OpenRouter model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
- Dataset: `huggan/wikiart`
- Decoding method: OpenRouter `response_format` with strict JSON schema-guided decoding
- Output labels: `has_human`, `has_animal`, `has_flower`

## How to run

```bash
cd submissions/202400977/assignment2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
python classify_wikiart.py --sample-count 20 --api-timeout 10
```

The script saves 20 sampled images as `images/wikiart_01.jpg` through
`images/wikiart_20.jpg`, writes the report to `results.html`, and writes an
image-included `results.pdf`. If an OpenRouter request takes longer than 10
seconds, that row is kept as an `ERROR` case so the run finishes quickly.

## Main instructions given to the agentic coding tool

1. Read the assignment specification in `assignments/assignment2` and implement the task in the same general submission style as `submissions/2025122/assignment2`.
2. Load image samples from `huggan/wikiart`, classify each image through OpenRouter using `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`, and force the output into a JSON schema containing `has_human`, `has_animal`, and `has_flower`.
3. Save 20 sampled images as JPG files, organize the results in one HTML file, and generate a PDF where the images are visible.

## Interesting cases from the results

1. A landscape-like sample can still become `has_animal=true` if small birds or livestock are visible.
2. Portrait and figure studies are useful checks for `has_human=true` even when the person is stylized.
3. Floral still-life samples are the clearest cases for `has_flower=true`, especially when flowers are central rather than decorative.

## Cases that may be confusing or possibly wrong

1. Small background figures or animals can be difficult to verify manually and may make labels debatable.
2. Some OpenRouter calls may time out or return empty content; these are kept as explicit `ERROR` cases in the report.
