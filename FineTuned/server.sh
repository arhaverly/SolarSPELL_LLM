#!/usr/bin/env bash
set -Eeuo pipefail

# Run Ask SolarSPELL in the foreground (llama-server + web app), using the
# settings install.sh wrote. If the systemd services are installed, use
# `sudo systemctl start solarspell-llm solarspell-app` instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/solarspell.env"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found. Run ./install.sh first."
    exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

echo "Starting Ask SolarSPELL"
echo
echo "Model:         $LLM_MODEL"
echo "Prompt format: $PROMPT_FORMAT"
echo "Web app:       http://localhost:$APP_PORT"
echo
echo "Press Ctrl+C to stop."
echo

"$LLAMA_SERVER" -m "$LLM_MODEL" -c "$CONTEXT" -t "$THREADS" \
    --host 127.0.0.1 --port "$LLM_PORT" &
LLM_PID=$!
trap 'kill "$LLM_PID" 2>/dev/null || true' EXIT

cd "$SCRIPT_DIR"
"$VENV_DIR/bin/uvicorn" app.main:app --host "$APP_HOST" --port "$APP_PORT"
