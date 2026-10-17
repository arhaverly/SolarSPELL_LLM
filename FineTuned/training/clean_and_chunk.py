"""Step 1: extract, clean and chunk PDF text for Q&A generation.

Ported from nx-sol-ai "Cleaning.ipynb". Writes one JSONL file per PDF, one
chunk per line: {"source": "<pdf name>", "chunk": n, "text": "..."}.

Usage: python clean_and_chunk.py /path/to/pdfs chunks/ [--chunk-size 1000]
"""

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E\n]+", "", text)  # drop non-printable / non-ASCII
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split into chunks of at most chunk_size characters on word boundaries."""
    chunks, current, length = [], [], 0
    for word in text.split():
        if current and length + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(args.pdf_dir.glob("*.pdf")):
        try:
            chunks = chunk_text(clean_text(extract_text(pdf)), args.chunk_size)
        except Exception as e:  # encrypted or corrupt PDFs
            print(f"Skipping {pdf.name}: {e}")
            continue
        with open(args.out_dir / f"{pdf.stem}.jsonl", "w", encoding="utf-8") as out:
            for n, chunk in enumerate(chunks, start=1):
                out.write(json.dumps({"source": pdf.stem, "chunk": n, "text": chunk}) + "\n")
        print(f"{pdf.name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
