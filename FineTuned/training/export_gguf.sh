#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Step 4: LoRA adapter -> merged model -> quantized GGUF
#
# Usage:
#   HF_TOKEN=hf_... ./export_gguf.sh <adapter-dir> [output.gguf]
#
# Example with the original nx-sol-ai checkpoint:
#   ./export_gguf.sh lora_llama/checkpoint-4000
#
# Run on a desktop/server (not the Pi). Copy the resulting .gguf to the Pi
# and install it with:  FINETUNED_GGUF=/path/to/file.gguf ../install.sh
# ============================================================

ADAPTER="${1:?Usage: $0 <adapter-dir> [output.gguf]}"
OUTPUT="${2:-solarspell-llama3.2-1b-Q4_K_M.gguf}"
QUANT="${QUANT:-Q4_K_M}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/SolarSPELL_LLM/llama.cpp}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    echo "Cloning llama.cpp into $LLAMA_DIR..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

if [[ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]]; then
    echo "Building llama-quantize..."
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
    cmake --build "$LLAMA_DIR/build" --config Release --target llama-quantize -j "$(nproc)"
fi

echo
echo "Installing GGUF conversion requirements..."
pip install -q -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"

echo
echo "Merging adapter $ADAPTER..."
python "$SCRIPT_DIR/merge_lora.py" "$ADAPTER" "$WORK_DIR/merged"

echo
echo "Converting to GGUF (f16)..."
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$WORK_DIR/merged" \
    --outtype f16 \
    --outfile "$WORK_DIR/model-f16.gguf"

echo
echo "Quantizing to $QUANT..."
"$LLAMA_DIR/build/bin/llama-quantize" "$WORK_DIR/model-f16.gguf" "$OUTPUT" "$QUANT"

echo
echo "Done: $OUTPUT"
sha256sum "$OUTPUT"
