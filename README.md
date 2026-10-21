> Returning to this repo? Start with [AGENTS.md](AGENTS.md) and [project context](docs/PROJECT_CONTEXT.md).

# MangaNarrator OCR

OCR processor module for manga/manhwa panels using Qwen2.5-VL or other models.

## Setup

```bash
conda activate qwenfix
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config config.yaml
```

Or run as an API server:

```bash
python ocr_server.py
```
