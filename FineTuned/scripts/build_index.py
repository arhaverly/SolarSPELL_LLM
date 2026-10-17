"""Build index.faiss + chunks.sqlite from a folder of PDFs.

Use this to index your own copy of the SolarSPELL library instead of (or in
addition to) the prebuilt nx-sol-ai index. Run it on a desktop; embedding
hundreds of thousands of chunks on a Raspberry Pi is slow.

Usage:
  python scripts/build_index.py /path/to/pdfs data/ [--model models/all-MiniLM-L6-v2]

Needs `pypdf` in addition to requirements.txt.
"""

import argparse
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_CHARS = 1000
OVERLAP_CHARS = 200


def chunk_words(text: str) -> list[str]:
    """Split into ~CHUNK_CHARS chunks on word boundaries, with overlap."""
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        if current and length + len(word) + 1 > CHUNK_CHARS:
            chunks.append(" ".join(current))
            # Carry the tail of this chunk into the next one.
            while current and length > OVERLAP_CHARS:
                length -= len(current.pop(0)) + 1
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def iter_chunks(pdf_dir: Path):
    for pdf in sorted(pdf_dir.rglob("*.pdf")):
        try:
            reader = PdfReader(pdf)
        except Exception as e:  # corrupt or encrypted PDFs are common in large libraries
            print(f"Skipping {pdf.name}: {e}")
            continue
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            for chunk in chunk_words(text):
                yield chunk, pdf.stem, str(number)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--model", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model, device=None)  # uses a GPU if available
    index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())

    db_path = args.out_dir / "chunks.sqlite"
    db_path.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT, page TEXT)")

    batch = []

    def flush():
        vectors = model.encode([text for text, _, _ in batch], batch_size=args.batch_size, convert_to_numpy=True)
        start = index.ntotal
        index.add(np.asarray(vectors, dtype="float32"))
        db.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            [(start + i, text, source, page) for i, (text, source, page) in enumerate(batch)],
        )
        db.commit()
        print(f"Indexed {index.ntotal} chunks")
        batch.clear()

    for row in iter_chunks(args.pdf_dir):
        batch.append(row)
        if len(batch) >= 4096:
            flush()
    if batch:
        flush()

    faiss.write_index(index, str(args.out_dir / "index.faiss"))
    db.close()
    print(f"Done: {index.ntotal} chunks in {args.out_dir}")


if __name__ == "__main__":
    main()
