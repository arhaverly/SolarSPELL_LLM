#!/usr/bin/env bash
set -Eeuo pipefail

# Pack data/index.faiss + data/chunks.sqlite into one file for installing on
# other devices (USB stick, scp, or a download link).
#
# Usage: scripts/pack_index.sh [output.tar.gz]
# Then on the Pi: put the file in ~/ or FineTuned/, and run ./install.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
OUTPUT="${1:-solarspell-index.tar.gz}"

for f in index.faiss chunks.sqlite; do
    if [[ ! -e "$DATA_DIR/$f" ]]; then
        echo "ERROR: $DATA_DIR/$f not found. Run ./install.sh on this machine first."
        exit 1
    fi
done

echo "Packing $DATA_DIR into $OUTPUT..."
# -h: follow symlinks, so a data/ that links to nx-sol-ai still packs real files.
tar -czhf "$OUTPUT" -C "$DATA_DIR" index.faiss chunks.sqlite
ls -lh "$OUTPUT"
sha256sum "$OUTPUT"
