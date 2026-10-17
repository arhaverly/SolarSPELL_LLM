# Ask SolarSPELL — Fine-Tuned + Retrieval

An offline chat assistant that answers questions from the SolarSPELL library.
For each question it:

1. embeds the question with `all-MiniLM-L6-v2`,
2. finds the most similar passages among 231,572 chunks from 1,494 SolarSPELL
   PDFs (health, nursing, microbiology, agriculture, development…) in a FAISS
   index,
3. hands those passages to a local Llama 3.2 1B model running in llama.cpp,
4. streams the answer to the browser, with the source documents and pages.

The web app and API are served on one port, so any phone or laptop on the
SolarSPELL hotspot can open `http://<pi-address>:8000`.

```
browser ──▶ app (FastAPI, :8000) ──▶ llama-server (127.0.0.1:8081)
                 │
                 └─ FAISS index + chunks.sqlite + MiniLM embeddings
```

This is a cleaned-up rewrite of ASU Next Lab's `nx-sol-ai`. See
[Changes from nx-sol-ai](#changes-from-nx-sol-ai).

## Hardware

* Raspberry Pi 5 (4 GB or more) or any 64-bit Linux machine
* 64-bit Raspberry Pi OS (Bookworm or later)
* ~3 GB free disk: index ≈ 580 MB, LLM ≈ 0.8 GB, Python packages ≈ 1 GB
* Active cooling: generation keeps all four cores busy

Memory at runtime is roughly 1 GB for the app (index + embedding model) plus
~1 GB for the 1B model.

## Install on a Raspberry Pi

You need two things: this repository and the library index file
`solarspell-index.tar.gz` (368 MB, not in git; see [Library index](#library-index)).

1. Copy `solarspell-index.tar.gz` to the Pi's home folder (USB stick or
   `scp solarspell-index.tar.gz pi@<pi-address>:~/`).
2. On the Pi:

   ```bash
   sudo apt-get update && sudo apt-get install -y git
   git clone https://github.com/arhaverly/SolarSPELL_LLM ~/SolarSPELL_LLM
   cd ~/SolarSPELL_LLM/FineTuned
   ./install.sh
   ```

3. Open the address it prints, e.g. `http://raspberrypi.local:8000`, from any
   device on the same network.

The installer can take 20+ minutes on a Pi, mostly building llama.cpp.
It needs Internet access while it runs, and none afterwards. It downloads stock
Llama 3.2 1B Instruct and installs two systemd services, so the app starts on
every boot:

| Service | Listens on |
|---|---|
| `solarspell-llm` | `127.0.0.1:8081` (local only) |
| `solarspell-app` | `0.0.0.0:8000` |

If `NonFineTuned/` was installed first, its `llama.cpp` build is reused.

Installer options (environment variables): `INDEX_BUNDLE` (path or URL of the
index file), `FINETUNED_GGUF` (use a fine-tuned model; see below), `THREADS`
(default 4), `CONTEXT` (2048), `APP_PORT` (8000), `LLM_PORT` (8081),
`NO_SYSTEMD=1` (don't install services; run `./server.sh` instead),
`SKIP_APT=1`, `REBUILD_LLAMA=1`.

### Library index

The index (231,572 passages with their embeddings) isn't stored in git: it's
large, and it contains the library text from ASU's private `nx-sol-ai`
repository. Make `solarspell-index.tar.gz` once on any machine, then reuse it
for every Pi:

* **From nx-sol-ai** (needs access to that repo and Git LFS):

  ```bash
  git lfs install && git clone https://github.com/asu-nextlab/nx-sol-ai
  cp nx-sol-ai/Backend/faiss_index/index.{faiss,pkl} FineTuned/data/
  ./install.sh                      # converts index.pkl → chunks.sqlite
  scripts/pack_index.sh             # writes solarspell-index.tar.gz
  ```

* **From your own PDFs** (on a desktop; embedding is slow on a Pi):

  ```bash
  pip install -r requirements.txt pypdf
  python scripts/build_index.py /path/to/pdfs data/
  scripts/pack_index.sh
  ```

To skip copying files around, upload the bundle somewhere the Pi can reach
(e.g. a private GitHub release) and run
`INDEX_BUNDLE=https://.../solarspell-index.tar.gz ./install.sh`.

## Use

Open `http://<pi-address>:8000` in a browser. The dot in the header turns
green when the language model is ready.

```bash
sudo systemctl status solarspell-app solarspell-llm
sudo journalctl -u solarspell-app -u solarspell-llm -f
./server.sh                      # foreground alternative to systemd
```

API:

```bash
curl -N http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of malaria?", "top_k": 3}'
```

The response is newline-delimited JSON: one `{"sources": [...]}` line, then
`{"token": "..."}` lines (or an `{"error": "..."}` line).

Runtime settings live in `solarspell.env` (restart the services after editing):

| Variable | Default | Meaning |
|---|---|---|
| `PROMPT_FORMAT` | set by installer | `alpaca` for the fine-tuned model, `chat` for stock instruct models |
| `TOP_K` | 3 | passages retrieved per question |
| `CONTEXT_CHARS` | 600 | characters kept from each passage |
| `MAX_TOKENS` | 512 | answer length limit |
| `TEMPERATURE` | 0.3 | sampling temperature |

## Optional: the fine-tuned model

ASU Next Lab LoRA-fine-tuned `meta-llama/Llama-3.2-1B-Instruct` on ~9,400
Gemini-generated Q&A pairs from SolarSPELL documents. **Neither the adapter
weights nor the Q&A dataset were published** (they were on ASU's compute
cluster), so you have two options:

1. **Get the original adapter** (`lora_llama/checkpoint-4000`, the checkpoint
   their app used) from the authors, then convert it:

   ```bash
   cd training
   pip install -r requirements.txt
   HF_TOKEN=hf_... ./export_gguf.sh /path/to/checkpoint-4000
   ```

2. **Reproduce it** with the pipeline in [`training/`](training/):

   | Step | Script | Runs on |
   |---|---|---|
   | 1. PDF → cleaned text chunks | `clean_and_chunk.py` | any machine |
   | 2. Chunks → Alpaca Q&A pairs (Gemini on Vertex AI, paid) | `generate_qa.py` | any machine |
   | 3. LoRA fine-tune | `train_lora.py` | CUDA GPU |
   | 4. Merge + convert + quantize to GGUF | `export_gguf.sh` | any machine |
   | 5. Evaluate (optional) | `evaluate.py` | GPU recommended |

   Llama 3.2 is gated on Hugging Face: accept the license and set `HF_TOKEN`.

Either way, install the result on the Pi with
`FINETUNED_GGUF=solarspell-llama3.2-1b-Q4_K_M.gguf ./install.sh`. The
installer switches `PROMPT_FORMAT` to `alpaca`, the prompt format the model
was trained on.

## Layout

```text
FineTuned/
├── install.sh              Pi installer (llama.cpp, venv, models, systemd)
├── server.sh               run in the foreground without systemd
├── requirements.txt        runtime Python packages
├── app/
│   ├── main.py             FastAPI app: /api/query, /api/health, web UI
│   ├── retrieval.py        FAISS + SQLite + MiniLM search
│   ├── llm.py              llama-server streaming client and prompts
│   ├── config.py           settings (environment variables)
│   └── static/             web UI (plain HTML/JS, no build step)
├── scripts/
│   ├── pack_index.sh       data/ → solarspell-index.tar.gz for other devices
│   ├── convert_index.py    LangChain index.pkl → chunks.sqlite
│   └── build_index.py      build a new index from PDFs
└── training/               dataset generation, fine-tuning, GGUF export
```

## Changes from nx-sol-ai

The original repository did not run as committed. Fixes and simplifications:

* **Index never loaded.** It was only loaded under `if __name__ == "__main__"`,
  which `uvicorn` never runs, so every query returned HTTP 500. It is now
  loaded at startup.
* **Broken imports.** It imported the `langchain` package, which wasn't in its
  requirements. LangChain is gone: the index is read with `faiss` and
  `sqlite3`, and the pickle is converted once by a restricted unpickler, with
  no `allow_dangerous_deserialization`.
* **Only worked on the Pi itself.** The UI was hard-coded to
  `http://localhost:8000` and CORS to `localhost:3000`, so phones on the
  hotspot couldn't use it. UI and API now share one origin.
* **Not actually offline.** The Ollama container downloaded `llama3.2` again
  on every start. The model is now downloaded once by the installer. Ollama
  and Docker were replaced with llama.cpp and systemd, matching
  `NonFineTuned/` and saving memory on the Pi.
* **Fine-tuned model was never used.** The app ran stock `llama3.2` (3B). It
  now runs a 1B model, and the fine-tuned model once you supply it, with its
  training prompt format.
* **Missing embedding model.** The model was a broken git submodule. The
  installer now downloads a pinned revision.
* **Simpler frontend.** The Vite/React/Tailwind app (about 200 KB of
  lockfile, and Node needed to build it) became a static page with vendored
  `marked` and `DOMPurify`. Model output is sanitized before rendering, and
  answers now show their sources.
* **Errors mid-stream.** Failures during streaming used to raise an exception
  after the response had started; now they arrive as an `error` event.
* **Training code.** The notebooks and script became runnable CLI scripts.
  A hard-coded Hugging Face token and Windows credential path were removed.
  The deprecated `vertexai` SDK was replaced with `google-genai`. Padding
  tokens are now masked from the loss, with an explicit EOS. Evaluation now
  scores only generated text; the notebook scored the prompt too.
