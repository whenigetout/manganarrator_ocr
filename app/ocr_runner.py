from pathlib import Path
from typing import List, Dict, Optional, Union
from .config import load_config
from .utils import ensure_output_dir, save_jsonl, Timer
from .backends.qwen_backend import QwenOCRBackend
import uuid
import json
from datetime import datetime
from app.utils import parse_dialogue
import re
import os
from collections import defaultdict
from PIL import Image
from app.models.domain import (
    MediaRef, InferImageResponse, InferImageError, DialogueLineResponse, 
    OCRImage, ProcessImageError, OCRRunError, OCRRunResponse,
    SaveJSONError
    )

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class OCRProcessor:
    def __init__(self, config_path: Union[str, Path]):
        self.config = load_config(str(config_path))

        # self.input_folder = Path(self.config["input_root_folder"]).resolve()
        # self.output_base = Path(self.config["output_root_folder"]).resolve()
        try:
            self.media_root: str = ""
            path = Path(self.config["media_root"]).resolve()
            self.media_root = str(path)
        except:
            raise ValueError("❌ Failed to load media_root path from config")
        self.model_name = self.config["ocr_model"]
        self.paddle_ocr_api = self.config["paddle_ocr_api"]
        self.mock_mode = os.getenv("MOCK_OCR", "0") == "1"

        ensure_output_dir(Path(self.media_root))
        if self.mock_mode:
            print("🧪 MOCK MODE ENABLED — skipping model load.")
            self.backend = None
        else:
            self.backend = self._init_backend(self.model_name)

    def _init_backend(self, model_name: str):
        if model_name == "qwen2.5vl":
            return QwenOCRBackend(self.config)
        else:
            raise ValueError(f"OCR model '{model_name}' not supported.")

    def process_image(self, imgRef: MediaRef, 
                      prompt: Optional[str]=None
                      ) -> OCRImage:
        try:
            image_id = str(uuid.uuid4())
            img_path = Path(self.media_root) / imgRef.namespace / imgRef.path
            img_name = img_path.name

            with Timer(f"🖼️ Process Image {img_name}"):
                if self.mock_mode:
                    result = InferImageResponse(
                        image_ref=MediaRef(
                            namespace="inputs",
                            path=""
                        ),
                        image_text="[Speaker 1 | female | happy]: \"This is a mock line.\"\n[Speaker 2 | male | angry]: \"Mock angry line here.\"",
                        image_width=1080,
                        image_height=1920,
                        input_tokens=42,
                        output_tokens=17,
                        throughput=59.1
                    )
                else:
                    result = self.backend.infer_image(imgRef, prompt=prompt) if self.backend else None

            text = result.image_text if result else ""
            parsed_dialogue_lines = parse_dialogue(text, image_id) if isinstance(text, str) else []

            return OCRImage(
                image_id=image_id,
                inferImageRes=result,
                parsedDialogueLines=parsed_dialogue_lines,
            )

        except Exception as e:
            raise ProcessImageError(f"❌ Error processing image {img_path.name}") from e

    def process_batch(self, 
                      inputfolderRef: MediaRef, 
                      run_id: Optional[str]=None,
                      prompt: Optional[str]=None
                      ) -> OCRRunResponse:
        try:
            folder_path = Path(self.media_root) / inputfolderRef.namespace / inputfolderRef.path
            folder_path.resolve()
            if not folder_path.exists() or not folder_path.is_dir():
                raise OCRRunError(f"Invalid folder path: {folder_path}")

            image_paths = sorted(
                list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.png")),
                key=lambda p: natural_key(p.name)
            )


            print(f"🔎 Found {len(image_paths)} image(s) under: {folder_path}")
            processImageResults = []
            for img_path in image_paths:
                path_base = Path(self.media_root) / inputfolderRef.namespace
                imgRef = MediaRef(
                    namespace=inputfolderRef.namespace,
                    path=str(img_path.relative_to(path_base))
                )
                processImageResult = self.process_image(
                    imgRef=imgRef,
                    prompt=prompt
                )
                processImageResults.append(processImageResult)

            return OCRRunResponse(
                run_id=run_id if run_id else "",
                imageResults=processImageResults
            )
        except Exception as e:
            return OCRRunResponse(
                run_id=run_id if run_id else "",
                error=str(e)
            )

    def save_output(self, ocr_run: OCRRunResponse, manga_folder_ref: MediaRef, run_name: Optional[str] = None) -> MediaRef:
        try:
            run_id = ocr_run.run_id 
            if not run_id:
                run_id = run_name if run_name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            base_out = Path(self.media_root)/ 'outputs' / run_id
            base_out.mkdir(parents=True, exist_ok=True)

            if not ocr_run.imageResults:
                raise SaveJSONError(f"Invalid OCRRun, no results found.")
            for ocrimg in ocr_run.imageResults:
                infer_img_res = ocrimg.inferImageRes
                if not infer_img_res:
                    raise SaveJSONError(f"Invalid OCRRun, invalid image result found.")
                
            # Save ONE json per run/batch
            out_dir = base_out / manga_folder_ref.path
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / "ocr_output.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(ocr_run.model_dump(), f, indent=2, ensure_ascii=False)
            media_outputs_Path = Path(self.media_root) / 'outputs'
            out_file_ref = MediaRef(
                namespace="outputs",
                path=str(out_file.relative_to(media_outputs_Path))
            )

            print(f"✅ Saved OCR outputs to: {out_dir}")
            return out_file_ref

        except Exception as e:
            raise SaveJSONError(f"Failed to save JSON for run_id/Output Folder: {out_dir}")