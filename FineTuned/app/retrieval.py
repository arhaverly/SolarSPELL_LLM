"""Semantic search over the SolarSPELL library.

The FAISS index holds one 384-dim all-MiniLM-L6-v2 vector per chunk; row N of
the index is row id=N in chunks.sqlite (see scripts/convert_index.py and
scripts/build_index.py).
"""

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    text: str
    source: str
    page: str


class Retriever:
    def __init__(self, index_path: Path, db_path: Path, model_path: Path):
        for path in (index_path, db_path, model_path):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run install.sh (see README).")

        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(str(model_path), device="cpu")
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()

        if self.index.d != self.model.encode(["probe"]).shape[1]:
            raise ValueError("Embedding model does not match the FAISS index dimension.")

    def search(self, query: str, k: int) -> list[Chunk]:
        vector = self.model.encode([query], convert_to_numpy=True).astype("float32")
        _, ids = self.index.search(vector, k)

        chunks = []
        with self._lock:
            for i in ids[0]:
                if i < 0:
                    continue
                row = self.db.execute(
                    "SELECT text, source, page FROM chunks WHERE id = ?", (int(i),)
                ).fetchone()
                if row:
                    chunks.append(Chunk(*row))
        return chunks
