"""Step 2: generate Alpaca-format Q&A pairs from chunks with Gemini on Vertex AI.

Ported from nx-sol-ai "Unstructured_data_to_qa_pairs_using_gemini.py".
Resumable: files already listed in <output>.done are skipped.

Authenticate first with `gcloud auth application-default login`.

Usage:
  python generate_qa.py chunks/ qa_alpaca.json --project YOUR_GCP_PROJECT \
      [--model gemini-2.5-flash] [--max-groups-per-file 0]
"""

import argparse
import json
import re
import time
from pathlib import Path

from google import genai

PROMPT_TEMPLATE = '''
You are an expert educator and AI assistant. Your job is to generate 2 high-quality, instructional question-answer pairs in valid JSON format for fine-tuning a LLaMA 3.2 1B instruct model using LoRA.

ONLY return a valid JSON array. No explanations, no markdown, no commentary. Just the JSON list.

Each question should:
- Be clear and educational.
- Help students reflect on key ideas in the text.
- Include a variety of types, such as:
    - What should I do if...?
    - Describe how/why something works.
    - Yes/No questions with explanations.
    - Common mistakes and how to avoid them.
    - Step-by-step problem-solving or processes.
    - Comparisons, summaries, or practical use-cases.
    - In case the excerpt context doesn't have an information, simply say "Sorry. I do not have information to answer this question"

Each answer should:
- Be at least one paragraph long.
- Be structured with headings or lists if needed.
- Include examples, analogies, or simple explanations where possible.
- Mention common mistakes or misconceptions.
- Offer helpful tips, comparisons, or rules of thumb.
- Sound like an expert teacher explaining it to a curious student.

Use this Alpaca format:
[
  {{
    "instruction": "your question here",
    "input": "",
    "output": "your long, structured answer here"
  }},
  ...
]

Below is one excerpt your retriever will supply at inference time.
Craft Q&A only from what is stated or strongly implied in the excerpt—no outside facts.
"""
{chunk}
"""
'''


def parse_pairs(content: str) -> list[dict]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
    text = fenced.group(1) if fenced else content
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        pairs = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [
        {"instruction": p["instruction"], "input": p.get("input", ""), "output": p["output"]}
        for p in pairs
        if isinstance(p, dict) and p.get("instruction") and p.get("output")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chunk_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", required=True, help="Google Cloud project with Vertex AI enabled")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--chunks-per-prompt", type=int, default=3)
    parser.add_argument("--max-groups-per-file", type=int, default=0, help="0 = no limit")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    done_file = args.output.with_suffix(args.output.suffix + ".done")
    done = set(done_file.read_text().split("\n")) if done_file.exists() else set()
    pairs = json.loads(args.output.read_text()) if args.output.exists() else []

    for path in sorted(args.chunk_dir.glob("*.jsonl")):
        if path.name in done:
            continue
        chunks = [json.loads(line)["text"] for line in path.open(encoding="utf-8")]
        groups = [chunks[i:i + args.chunks_per_prompt] for i in range(0, len(chunks), args.chunks_per_prompt)]
        if args.max_groups_per_file:
            groups = groups[:args.max_groups_per_file]

        new = 0
        for group in groups:
            prompt = PROMPT_TEMPLATE.format(chunk=" ".join(group))
            try:
                response = client.models.generate_content(model=args.model, contents=prompt)
                got = parse_pairs(response.text or "")
            except Exception as e:
                print(f"  API error: {e}")
                got = []
            pairs.extend(got)
            new += len(got)
            time.sleep(args.delay)

        args.output.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
        with done_file.open("a") as f:
            f.write(path.name + "\n")
        print(f"{path.name}: +{new} pairs (total {len(pairs)})")


if __name__ == "__main__":
    main()
