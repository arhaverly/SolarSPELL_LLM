#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Ask SolarSPELL (RAG + fine-tuned Llama 3.2 1B) installer
# Raspberry Pi OS 64-bit / Debian / Ubuntu
#
# Installs:
#   - llama.cpp's llama-server (shared with ../NonFineTuned)
#   - a Python venv with the retrieval app
#   - the all-MiniLM-L6-v2 embedding model
#   - the SolarSPELL library index (from an index bundle, see README)
#   - the LLM: $FINETUNED_GGUF if given, else stock Llama 3.2 1B Instruct
#   - systemd services: solarspell-llm and solarspell-app
#
# Needs Internet only while installing. See README.md for details.
# ============================================================

# ---------- Configuration -----------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AI_DIR="${AI_DIR:-$HOME/SolarSPELL_LLM}"
LLAMA_DIR="${LLAMA_DIR:-$AI_DIR/llama.cpp}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
DATA_DIR="$SCRIPT_DIR/data"
MODEL_DIR="$SCRIPT_DIR/models"
ENV_FILE="$SCRIPT_DIR/solarspell.env"

# Optional: path to a fine-tuned GGUF made with training/export_gguf.sh.
FINETUNED_GGUF="${FINETUNED_GGUF:-}"
FINETUNED_NAME="solarspell-finetuned.gguf"

# Fallback LLM: the base model the SolarSPELL fine-tune was trained from.
STOCK_NAME="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
STOCK_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/$STOCK_NAME"
STOCK_SHA256="6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"

EMBEDDING_REPO="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Library index bundle made by scripts/pack_index.sh: a local path or an
# http(s) URL. If unset, solarspell-index.tar.gz is looked for next to this
# script and in your home directory.
INDEX_BUNDLE="${INDEX_BUNDLE:-}"

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
LLM_PORT="${LLM_PORT:-8081}"
CONTEXT="${CONTEXT:-2048}"
THREADS="${THREADS:-4}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

SKIP_APT="${SKIP_APT:-0}"
NO_SYSTEMD="${NO_SYSTEMD:-0}"


# ---------- Helpers ------------------------------------------

download() {
    local url="$1" dest="$2" sha="$3"
    echo "Downloading $url"
    curl --fail --location --progress-bar --retry 5 --retry-delay 2 --retry-all-errors \
        "$url" --output "$dest.part"
    echo "$sha  $dest.part" | sha256sum --check -
    mv "$dest.part" "$dest"
}


# ---------- Basic checks -------------------------------------

if [[ "$EUID" -eq 0 ]]; then
    echo "ERROR: Run this script as your normal user, not with sudo."
    echo "The script will call sudo when required."
    exit 1
fi

echo
echo "============================================"
echo " System"
echo "============================================"
echo "Architecture : $(uname -m)"
echo "CPU cores    : $(nproc)"
free -h
echo


# ---------- Install dependencies -----------------------------

if [[ "$SKIP_APT" != "1" ]]; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: This script requires Debian/Ubuntu with apt (or SKIP_APT=1)."
        exit 1
    fi

    echo "Installing dependencies..."
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential cmake git curl ca-certificates libssl-dev pkg-config \
        python3 python3-venv python3-dev
fi

mkdir -p "$DATA_DIR" "$MODEL_DIR"


# ---------- llama.cpp ----------------------------------------

LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"

if [[ -x "$LLAMA_SERVER" && "${REBUILD_LLAMA:-0}" != "1" ]]; then
    echo
    echo "Using existing llama-server: $LLAMA_SERVER"
else
    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
        echo
        echo "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi

    echo
    echo "Building llama-server using $BUILD_JOBS jobs..."
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "$LLAMA_DIR/build" --config Release --target llama-server -j "$BUILD_JOBS"
fi

"$LLAMA_SERVER" --version


# ---------- Python environment -------------------------------

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"

echo
echo "Installing Python packages..."
"$PY" -m pip install --quiet --upgrade pip
# CPU-only torch: avoids multi-GB CUDA wheels on x86, and is the normal build on ARM.
"$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"


# ---------- Embedding model ----------------------------------

EMBEDDING_PATH="$MODEL_DIR/all-MiniLM-L6-v2"

if [[ ! -f "$EMBEDDING_PATH/modules.json" ]]; then
    echo
    echo "Downloading embedding model..."
    "$PY" -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('$EMBEDDING_REPO', revision='$EMBEDDING_REVISION', device='cpu').save('$EMBEDDING_PATH')
"
fi


# ---------- Library index ------------------------------------

if [[ -z "$INDEX_BUNDLE" ]]; then
    for candidate in "$SCRIPT_DIR/solarspell-index.tar.gz" "$HOME/solarspell-index.tar.gz"; do
        if [[ -f "$candidate" ]]; then
            INDEX_BUNDLE="$candidate"
            break
        fi
    done
fi

if [[ -f "$DATA_DIR/index.faiss" && -f "$DATA_DIR/chunks.sqlite" ]]; then
    echo
    echo "Using existing index in $DATA_DIR"
elif [[ -n "$INDEX_BUNDLE" ]]; then
    if [[ "$INDEX_BUNDLE" =~ ^https?:// ]]; then
        echo
        echo "Downloading index bundle..."
        curl --fail --location --progress-bar --retry 5 --retry-delay 2 --retry-all-errors \
            "$INDEX_BUNDLE" --output "$DATA_DIR/bundle.tar.gz.part"
        mv "$DATA_DIR/bundle.tar.gz.part" "$DATA_DIR/bundle.tar.gz"
        INDEX_BUNDLE="$DATA_DIR/bundle.tar.gz"
    fi
    echo
    echo "Extracting index from $INDEX_BUNDLE..."
    tar -xzf "$INDEX_BUNDLE" -C "$DATA_DIR" index.faiss chunks.sqlite
    rm -f "$DATA_DIR/bundle.tar.gz"
elif [[ -f "$DATA_DIR/index.faiss" && -f "$DATA_DIR/index.pkl" ]]; then
    echo
    echo "Converting index.pkl to chunks.sqlite..."
    "$PY" "$SCRIPT_DIR/scripts/convert_index.py" "$DATA_DIR/index.pkl" "$DATA_DIR/chunks.sqlite"
    rm -f "$DATA_DIR/index.pkl"
else
    echo
    echo "ERROR: No library index found."
    echo "Put solarspell-index.tar.gz in $SCRIPT_DIR or $HOME,"
    echo "or set INDEX_BUNDLE to its path or URL. See README.md, \"Library index\"."
    exit 1
fi


# ---------- LLM ----------------------------------------------

if [[ -n "$FINETUNED_GGUF" ]]; then
    echo
    echo "Installing fine-tuned model from $FINETUNED_GGUF"
    cp "$FINETUNED_GGUF" "$MODEL_DIR/$FINETUNED_NAME"
fi

if [[ -f "$MODEL_DIR/$FINETUNED_NAME" ]]; then
    LLM_MODEL="$MODEL_DIR/$FINETUNED_NAME"
    PROMPT_FORMAT="alpaca"
else
    if [[ ! -f "$MODEL_DIR/$STOCK_NAME" ]]; then
        echo
        echo "No fine-tuned model given; downloading stock $STOCK_NAME..."
        download "$STOCK_URL" "$MODEL_DIR/$STOCK_NAME" "$STOCK_SHA256"
    fi
    LLM_MODEL="$MODEL_DIR/$STOCK_NAME"
    PROMPT_FORMAT="chat"
fi


# ---------- Settings -----------------------------------------

cat > "$ENV_FILE" <<EOF
# Written by install.sh. Read by server.sh and the systemd services.
LLAMA_SERVER=$LLAMA_SERVER
LLM_MODEL=$LLM_MODEL
LLM_PORT=$LLM_PORT
CONTEXT=$CONTEXT
THREADS=$THREADS
APP_HOST=$APP_HOST
APP_PORT=$APP_PORT
LLM_URL=http://127.0.0.1:$LLM_PORT
PROMPT_FORMAT=$PROMPT_FORMAT
EOF


# ---------- systemd services ---------------------------------

if [[ "$NO_SYSTEMD" != "1" ]]; then
    echo
    echo "Installing systemd services..."

    sudo tee /etc/systemd/system/solarspell-llm.service >/dev/null <<EOF
[Unit]
Description=Ask SolarSPELL language model (llama.cpp)
After=network.target

[Service]
Type=simple
User=$USER
EnvironmentFile=$ENV_FILE
ExecStart=$LLAMA_SERVER -m \${LLM_MODEL} -c \${CONTEXT} -t \${THREADS} --host 127.0.0.1 --port \${LLM_PORT}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    sudo tee /etc/systemd/system/solarspell-app.service >/dev/null <<EOF
[Unit]
Description=Ask SolarSPELL web app
After=network.target solarspell-llm.service
Wants=solarspell-llm.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/uvicorn app.main:app --host \${APP_HOST} --port \${APP_PORT}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable solarspell-llm solarspell-app
    sudo systemctl restart solarspell-llm solarspell-app
fi


# ---------- Results ------------------------------------------

echo
echo "============================================"
echo " Installation complete"
echo "============================================"
echo
echo "LLM:            $LLM_MODEL"
echo "Prompt format:  $PROMPT_FORMAT"
echo
echo "Open from any device on the network:"
echo "  http://$(hostname).local:$APP_PORT"
for ip in $(hostname -I 2>/dev/null); do
    echo "  http://$ip:$APP_PORT"
done
echo
if [[ "$NO_SYSTEMD" != "1" ]]; then
    echo "Status:  sudo systemctl status solarspell-app solarspell-llm"
    echo "Logs:    sudo journalctl -u solarspell-app -u solarspell-llm -f"
else
    echo "Start with: ./server.sh"
fi
echo
echo "Done."
