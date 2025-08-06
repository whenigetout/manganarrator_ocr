#!/bin/bash

# Set environment name
ENV_NAME="qwen"

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Run the OCR server
echo "🚀 Starting OCR FastAPI server..."
uvicorn ocr_server:app --host 0.0.0.0 --port 7860 --reload
