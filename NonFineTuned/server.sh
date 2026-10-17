#!/usr/bin/env bash
set -Eeuo pipefail

AI_DIR="${AI_DIR:-$HOME/SolarSPELL_LLM}"
LLAMA_DIR="${LLAMA_DIR:-$AI_DIR/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$AI_DIR/models}"

MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B-Q4_0.gguf}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_NAME}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CONTEXT="${CONTEXT:-1024}"
THREADS="${THREADS:-4}"

LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
SERVICE_NAME="${SERVICE_NAME:-llama-server}"
API_KEY_FILE="${API_KEY_FILE:-$HOME/.config/llama-server/api.key}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "ERROR: llama-server not found at:"
    echo "  $LLAMA_SERVER"
    echo
    echo "Run ./install-llm.sh first."
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at:"
    echo "  $MODEL_PATH"
    echo
    echo "Run ./install-llm.sh first."
    exit 1
fi

# install-llm.sh also starts a systemd service on the same port.
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "The SolarSPELL LLM is already running as the '$SERVICE_NAME' service:"
    echo "  http://localhost:$PORT"
    echo
    echo "To run it in this terminal instead, stop the service first:"
    echo "  sudo systemctl stop $SERVICE_NAME"
    exit 0
fi

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "ERROR: Port $PORT is already in use by another program."
    echo "Stop it, or choose another port:  PORT=8090 ./server.sh"
    exit 1
fi

API_KEY_ARGS=()
if [[ -s "$API_KEY_FILE" ]]; then
    API_KEY_ARGS=(--api-key-file "$API_KEY_FILE")
fi

echo "Starting SolarSPELL LLM"
echo
echo "Model:   $MODEL_PATH"
echo "Host:    $HOST"
echo "Port:    $PORT"
echo "Context: $CONTEXT"
echo "Threads: $THREADS"
echo
echo "Server URL:"
echo "  http://localhost:$PORT"
if [[ ${#API_KEY_ARGS[@]} -gt 0 ]]; then
    echo
    echo "API key: $API_KEY_FILE"
fi
echo
echo "Press Ctrl+C to stop."
echo

exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT" \
    -t "$THREADS" \
    --reasoning-budget 0 \
    --host "$HOST" \
    --port "$PORT" \
    "${API_KEY_ARGS[@]}"
