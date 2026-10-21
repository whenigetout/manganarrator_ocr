# Hosted vision backend

This optional backend uses OpenAI or Gemini image understanding instead of loading Qwen locally.

1. Install the existing OCR requirements, including `httpx`.
2. In `config.yaml`, set `ocr_model: openai` or `ocr_model: gemini`.
3. Set `OPENAI_API_KEY` or `GEMINI_API_KEY` in the process environment. Never commit API keys.
4. Optionally set `VISION_MODEL` (defaults: `gpt-4.1-mini` or `gemini-2.5-flash`).
5. Launch the OCR server as usual.

The backend returns the existing `InferImageResponse`, using the existing dialogue-line parser and downstream contracts. Existing Qwen and PaddleOCR paths are retained. Hosted providers incur separate API costs; the entire JPEG image is sent to the selected provider. Speaker identification remains a best-effort guess. Validate the resulting dialogue and bbox matches in the workbench.

This is an initial adapter, not a complete structured-output or cross-page character-consistency solution. Run a small sample before processing a whole chapter.
