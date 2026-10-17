#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# llama.cpp + Qwen3-0.6B installer / systemd host
# Ubuntu / Debian
# ============================================================

# ---------- Configuration -----------------------------------

AI_DIR="${AI_DIR:-$HOME/SolarSPELL_LLM}"
LLAMA_DIR="${LLAMA_DIR:-$AI_DIR/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$AI_DIR/models}"

MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B-Q4_0.gguf}"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

MODEL_URL="https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/$MODEL_NAME"

# Official SHA256 for Qwen3-0.6B-Q4_0.gguf
MODEL_SHA256="da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CONTEXT="${CONTEXT:-1024}"
THREADS="${THREADS:-4}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

SERVICE_NAME="llama-server"

API_CONFIG_DIR="$HOME/.config/llama-server"
API_KEY_FILE="$API_CONFIG_DIR/api.key"


# ---------- Basic checks -------------------------------------

if [[ "$EUID" -eq 0 ]]; then
    echo "ERROR: Run this script as your normal user, not with sudo."
    echo "The script will call sudo when required."
    exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: This script requires Debian/Ubuntu with apt."
    exit 1
fi

echo
echo "============================================"
echo " System"
echo "============================================"
echo "Architecture : $(uname -m)"
echo "CPU cores    : $(nproc)"
echo "Memory:"
free -h
echo


# ---------- Install dependencies -----------------------------

echo "Installing dependencies..."

sudo apt-get update

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    openssl \
    libssl-dev \
    pkg-config

echo
echo "CMake: $(cmake --version | head -n1)"
echo "GCC:   $(gcc --version | head -n1)"


# ---------- Directory setup ----------------------------------

mkdir -p "$AI_DIR"
mkdir -p "$MODEL_DIR"


# ---------- Clone/update llama.cpp ----------------------------

if [[ -d "$LLAMA_DIR/.git" ]]; then
    echo
    echo "Updating existing llama.cpp repository..."

    git -C "$LLAMA_DIR" fetch origin
    git -C "$LLAMA_DIR" checkout master
    git -C "$LLAMA_DIR" pull --ff-only
else
    echo
    echo "Cloning llama.cpp..."

    git clone \
        https://github.com/ggml-org/llama.cpp.git \
        "$LLAMA_DIR"
fi


# ---------- Configure ----------------------------------------

echo
echo "Configuring llama.cpp..."

cd "$LLAMA_DIR"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=ON \
    -DLLAMA_USE_PREBUILT_UI=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_OPENSSL=ON


# ---------- Build --------------------------------------------

echo
echo "Building llama-server using $BUILD_JOBS jobs..."

cmake --build build \
    --config Release \
    --target llama-server \
    -j "$BUILD_JOBS"


# ---------- Verify executable --------------------------------

LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "ERROR: llama-server was not built."
    exit 1
fi

echo
echo "llama-server version:"
"$LLAMA_SERVER" --version


# ---------- Download model -----------------------------------

if [[ ! -f "$MODEL_PATH" ]]; then
    echo
    echo "Downloading $MODEL_NAME..."

    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 2 \
        --retry-all-errors \
        "$MODEL_URL" \
        --output "$MODEL_PATH.part"

    mv "$MODEL_PATH.part" "$MODEL_PATH"
else
    echo
    echo "Model already exists:"
    echo "$MODEL_PATH"
fi


# ---------- Verify model -------------------------------------

echo
echo "Verifying model SHA256..."

echo "$MODEL_SHA256  $MODEL_PATH" | sha256sum --check -


# ---------- API key ------------------------------------------

mkdir -p "$API_CONFIG_DIR"
chmod 700 "$API_CONFIG_DIR"

if [[ ! -s "$API_KEY_FILE" ]]; then
    echo
    echo "Generating API key..."

    umask 077
    openssl rand -hex 32 > "$API_KEY_FILE"
fi

chmod 600 "$API_KEY_FILE"


# ---------- Install systemd service ---------------------------

echo
echo "Installing systemd service..."

sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<EOF
[Unit]
Description=llama.cpp Qwen3 LLM Server
Wants=network-online.target
After=network-online.target

[Service]
Type=simple

User=$USER
WorkingDirectory=$LLAMA_DIR

ExecStart=$LLAMA_SERVER \
    -m $MODEL_PATH \
    -c $CONTEXT \
    -t $THREADS \
    --reasoning-budget 0 \
    --host $HOST \
    --port $PORT \
    --api-key-file $API_KEY_FILE

Restart=on-failure
RestartSec=3

TimeoutStopSec=20
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF


# ---------- Start service ------------------------------------

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"


# ---------- Results ------------------------------------------

echo
echo "============================================"
echo " Installation complete"
echo "============================================"

echo
echo "Model:"
echo "  $MODEL_PATH"

echo
echo "Server:"
echo "  http://127.0.0.1:$PORT"
echo

echo "LAN addresses:"
hostname -I 2>/dev/null || true

echo
echo "API key:"
cat "$API_KEY_FILE"

echo
echo
echo "Service status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo
echo "Live logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo
echo "Restart:"
echo "  sudo systemctl restart $SERVICE_NAME"
echo
echo "Stop:"
echo "  sudo systemctl stop $SERVICE_NAME"

echo
echo "Done."
