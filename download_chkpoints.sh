#!/bin/bash

# download checkpoints from huggingface with curl
# 1) get json of repo 2) filter to .pth 3) download 
USERNAME="rmdocherty"
MODEL_ID="dino-saw"
TARGET_DIR="models/checkpoints/trained"

mkdir -p "$TARGET_DIR"

# Get list of files from Hugging Face API
curl -s \
 "https://huggingface.co/api/models/$USERNAME/$MODEL_ID" \
    | grep -oP '"rfilename"\s*:\s*"\K[^"]+' \
    | grep '\.pth' \
    | while read FILE; do
        URL="https://huggingface.co/$USERNAME/$MODEL_ID/resolve/main/$FILE"
        OUT_FILE="$TARGET_DIR/$FILE"
        echo "Downloading $FILE ..."
        curl -s -L "$URL" -o "$OUT_FILE"
    done