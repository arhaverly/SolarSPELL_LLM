```bash
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
echo
echo "Press Ctrl+C to stop."
echo

exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT" \
    -t "$THREADS" \
    --reasoning-budget 0 \
    --host "$HOST" \
    --port "$PORT"
```
