"""Convert the original LangChain docstore (index.pkl) into chunks.sqlite.

The original nx-sol-ai index was saved with LangChain's FAISS.save_local, which
pickles the chunk text. Loading that normally needs LangChain and
allow_dangerous_deserialization=True. This script uses a restricted unpickler
that only accepts the two LangChain classes involved (mapped to inert stubs),
so neither LangChain nor arbitrary pickled code is involved.

index.faiss is used as-is; row N of it becomes chunks.id = N.

Usage: python scripts/convert_index.py data/index.pkl data/chunks.sqlite
"""

import pickle
import sqlite3
import sys
from pathlib import Path

ALLOWED = {
    ("langchain_community.docstore.in_memory", "InMemoryDocstore"),
    ("langchain_core.documents.base", "Document"),
}


class _Stub:
    def __setstate__(self, state):
        self.state = state


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return _Stub
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")


def _fields(obj: _Stub) -> dict:
    # Pydantic models pickle as {"__dict__": fields, ...}; plain objects as fields.
    state = obj.state
    return state.get("__dict__", state)


def _page(metadata: dict) -> str:
    if metadata.get("page_label"):
        return str(metadata["page_label"])
    if "page" in metadata:
        return str(int(metadata["page"]) + 1)
    return ""


def main(pkl_path: Path, db_path: Path) -> None:
    with open(pkl_path, "rb") as f:
        docstore, index_to_id = _RestrictedUnpickler(f).load()
    documents = _fields(docstore)["_dict"]

    db_path.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT, page TEXT)")

    rows = []
    for position, doc_id in index_to_id.items():
        doc = _fields(documents[doc_id])
        metadata = doc.get("metadata") or {}
        source = Path(metadata.get("source", "")).stem
        rows.append((int(position), doc["page_content"], source, _page(metadata)))

    db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", rows)
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"Wrote {len(rows)} chunks to {db_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
