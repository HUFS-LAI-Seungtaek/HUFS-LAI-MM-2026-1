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
python classify_wikiart.py --sample-count 8
```

The script saves sampled images into `images/`, writes the browser report to
`results.html`, and writes the PDF report to `results.pdf`.

## Main instructions given to the agentic coding tool

1. Read the assignment specification in `assignments/assignment2` and implement the task in the same general submission style as `submissions/2025122/assignment2`.
2. Load image samples from `huggan/wikiart`, classify each image through OpenRouter using `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`, and force the output into a JSON schema containing `has_human`, `has_animal`, and `has_flower`.
3. Save the outputs as `README.md`, `classify_wikiart.py`, `results.html`, `images/`, and `requirements.txt` under `submissions/202400977/assignment2`.
4. Install dependencies, run the implementation, fix errors, then stage, commit, push, and make a PR.

## Interesting cases from the results

1. Sample 1 was labeled `has_animal=true` because the model noticed small birds in an otherwise landscape-like scene. This was interesting because the animal evidence was not the main subject.
2. Sample 2 was labeled `has_human=true` even though the visible human evidence was small and low-contrast. The model relied on a figure in the lower part of the image.
3. Sample 3 was a clearer portrait-style case, and the model confidently marked `has_human=true` while keeping animal and flower labels false.

## Cases that may be confusing or possibly wrong

1. Sample 1 may be debatable because the birds are very small; depending on grading criteria, tiny background birds might or might not be counted as `has_animal`.
2. Sample 2 may be a possible weak case because the model focused on a human-like figure in a dense black-and-white composition, where the figure boundary is not very clear.
