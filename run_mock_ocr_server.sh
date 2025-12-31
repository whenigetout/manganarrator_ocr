#!/bin/bash
set -e

# Always run from project root
cd "$(dirname "$0")"

ENV_NAME="qwen"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "🚀 Starting OCR FastAPI server..."
MOCK_OCR=1 uvicorn ocr_server:app --host 0.0.0.0 --port 7860 --reload
