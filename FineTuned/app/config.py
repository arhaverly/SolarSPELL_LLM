"""Runtime settings. Every value can be overridden with an environment variable
(install.sh writes them to FineTuned/solarspell.env)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # FineTuned/
STATIC_DIR = Path(__file__).resolve().parent / "static"

INDEX_PATH = Path(os.getenv("INDEX_PATH", BASE_DIR / "data" / "index.faiss"))
CHUNKS_DB_PATH = Path(os.getenv("CHUNKS_DB_PATH", BASE_DIR / "data" / "chunks.sqlite"))
EMBEDDING_MODEL_PATH = Path(os.getenv("EMBEDDING_MODEL_PATH", BASE_DIR / "models" / "all-MiniLM-L6-v2"))

# llama-server started by install.sh / server.sh. Bound to localhost only;
# the browser talks to this app, never to llama-server directly.
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8081")

# "alpaca": the SolarSPELL fine-tuned model, trained on
#           "### Instruction / ### Input / ### Response" prompts.
# "chat":   any stock instruct model, using its own chat template.
PROMPT_FORMAT = os.getenv("PROMPT_FORMAT", "chat")

TOP_K = int(os.getenv("TOP_K", "3"))
CONTEXT_CHARS = int(os.getenv("CONTEXT_CHARS", "600"))  # per retrieved chunk
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
