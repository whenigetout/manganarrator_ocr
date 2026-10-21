# manganarrator_ocr: project context

Source review: 2026-10-21. Baseline commit: `09639816b021100275e4ac6421ca9b57b0ec7cdf`.
For later sessions, compare relevant paths against that commit; avoid re-auditing unrelated services.

## Purpose and boundaries

Manga image OCR and dialogue extraction, with local Qwen and optional hosted vision backends and Paddle bbox alignment.

Emits mn_contracts.ocr.OCRRun; Paddle helper supplies spatial boxes. Stories bypass OCR entirely. Preserve image identifiers, original geometry, and domain-to-contract conversion.

## Code map

ocr_server.py; app/ocr_runner.py; app/backends/qwen_backend.py; app/backends/hosted_vision_backend.py; app/special_utils/paddle_bbox_mapper.py; app/models/domain_to_contract.py; docs/HOSTED_VISION.md

## Setup and checks

Use the configured GPU environment and requirements appropriate to that backend. CLI: `python main.py --config config.yaml`. API: `python -m uvicorn ocr_server:app --host 127.0.0.1 --port 7860`. Importing may require model dependencies; syntax compilation alone does not test OCR.

## Known details and gaps

config.yaml selects qwen2.5vl/qwen3vl or hosted provider. Hosted backends read OPENAI_API_KEY/GEMINI_API_KEY and optional VISION_MODEL; using them transmits image data and may cost money. No keys in docs or fixtures. No checked-in automated test suite was found.

## Current state / next work

This note was grounded in source inspection at the baseline above. Runtime verification for this service is recorded below only when actually executed. Documentation-only edits do not establish that GPU inference, external APIs, or the complete distributed manga pipeline work in this environment.

For story production, first inspect [Story Studio in the video service](https://github.com/whenigetout/manganarrator_video/blob/main/docs/STORY_STUDIO.md). The first slice uses imported segment audio; OCR and Paddle are unchanged. Native story TTS is a later adapter, not an image-shaped workaround.

## Related repositories

[Central ecosystem map](https://github.com/whenigetout/manganarrator/blob/main/docs/ECOSYSTEM.md) owns service responsibilities and the cross-repo task routing table. Consult it only for work crossing a service boundary.
