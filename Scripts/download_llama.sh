#!/usr/bin/env bash
# Downloads Llama 3.2 3B Instruct Q4_K_M GGUF from Hugging Face.
# The model is placed at Models/Llama/ where the Xcode project expects it.
#
# Prerequisites:
#   • Accept the Llama 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
#     (Note: the GGUF below is from a community conversion — it also requires the base license.)
#   • Either install huggingface-cli (`pip install huggingface_hub`) OR just use curl (default).
#
# Usage:
#   bash Scripts/download_llama.sh

set -euo pipefail

REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DEST_DIR="$(dirname "$0")/../Models/Llama"
DEST_NAME="llama-3.2-3b-instruct-q4_k_m.gguf"
DEST="$DEST_DIR/$DEST_NAME"
URL="https://huggingface.co/$REPO/resolve/main/$FILE"

mkdir -p "$DEST_DIR"

if [ -f "$DEST" ]; then
    echo "Model already present at $DEST — nothing to do."
    exit 0
fi

echo "Downloading $FILE (~2.0 GB)…"
echo "  from: $URL"
echo "  to:   $DEST"
echo ""

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$REPO" "$FILE" --local-dir "$DEST_DIR" --local-dir-use-symlinks False
    # huggingface-cli uses the original filename; rename to match the Xcode bundle resource name
    mv -f "$DEST_DIR/$FILE" "$DEST"
else
    curl -L --progress-bar -o "$DEST" "$URL"
fi

echo ""
echo "Done. Model saved to:"
echo "  $DEST"
echo ""
echo "Next steps:"
echo "  1. Open the project in Xcode."
echo "  2. The file is already referenced at Models/Llama/ in the project."
echo "     If Xcode shows it with a red filename, right-click → 'Show in Finder'"
echo "     to confirm the path, then rebuild."
echo "  3. Build and run — the Assistant tab uses Llama 3.2."
