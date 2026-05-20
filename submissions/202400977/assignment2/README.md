# Assignment 2: WikiArt Visual Classification

## Used model

- OpenRouter model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
- Dataset: `huggan/wikiart`
- Decoding method: OpenRouter `response_format` with JSON schema-guided decoding
- Output labels: `has_human`, `has_animal`, `has_flower`

## How to run

```bash
cd submissions/202400977/assignment2
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
python classify_wikiart.py --sample-count 20
```

The script saves 20 sampled images into `images/`, writes the browser report to
`results.html`, and writes the PDF report to `results.pdf`.

## Main instructions given to the agentic coding tool

1. Read the assignment specification in `assignments/assignment2` and implement the task in the same general submission style as `submissions/2025122/assignment2`.
2. Load image samples from `huggan/wikiart`, classify each image through OpenRouter using `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`, and force the output into a JSON schema containing `has_human`, `has_animal`, and `has_flower`.
3. Save the outputs as `README.md`, `classify_wikiart.py`, `results.html`, `images/`, and `requirements.txt` under `submissions/202400977/assignment2`.
4. Install dependencies, run the implementation, fix errors, then stage, commit, push, and make a PR.

## Interesting cases from the results

1. Sample 6 was labeled `has_human=true` and `has_animal=true` because the model found people plus a cow in the lower-right area.
2. Sample 7 was a clean floral case: the model marked only `has_flower=true` for a bouquet in a vase.
3. Sample 16 was the richest multi-label case, with `has_human=true`, `has_animal=true`, and `has_flower=true`.

## Cases that may be confusing or possibly wrong

1. Samples 1, 10, 15, and 20 returned empty visible content from the OpenRouter provider even though the API request completed. The script now retries and includes a reasoning fallback for this provider behavior.
2. Sample 16 may need manual review because the model reported all three labels, but the generated reason was slightly malformed around the flower evidence.
