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


def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class OCRProcessor:
    def __init__(self, config_path: Union[str, Path]):
        self.config = load_config(config_path)

        self.input_folder = Path(self.config["input_root_folder"]).resolve()
        self.output_base = Path(self.config["output_root_folder"]).resolve()
        self.model_name = self.config["ocr_model"]
        self.mock_mode = os.getenv("MOCK_OCR", "0") == "1"

        ensure_output_dir(self.output_base)
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

    def process_image(self, img_path: Path, base_folder: Optional[Path] = None, 
                      input_folder_rel_path_from_input_root: Optional[Path] = None,
                      run_id: Optional[str]=None
                      ) -> Dict:
        try:
            image_id = str(uuid.uuid4())
            img_name = img_path.name
            with Timer(f"🖼️ Process Image {img_name}"):
                if self.mock_mode:
                    result = {
                        "text": "[Speaker 1 | female | happy]: \"This is a mock line.\"\n[Speaker 2 | male | angry]: \"Mock angry line here.\"",
                        "input_tokens": 42,
                        "output_tokens": 17,
                        "throughput": 59.1
                    }
                else:
                    result = self.backend.infer_image(img_path)


            base = base_folder or self.input_folder
            try:
                image_file_name = str(img_path.relative_to(base.resolve()))
            except ValueError:
                image_file_name = img_path.name  # fallback if outside base path

            text = result.get("text", "")
            parsed = parse_dialogue(text, image_id, image_file_name, str(input_folder_rel_path_from_input_root)) if isinstance(text, str) else []

            return {
                "image_file_name": image_file_name,
                "image_rel_path_from_root": str(input_folder_rel_path_from_input_root),
                "image_id": image_id,
                "run_id": run_id,
                "result": result,
                "parsed_dialogue": parsed
            }

        except Exception as e:
            print(f"❌ Error processing image {img_path.name}: {e}")
            return {
                "image_file_name": str(img_path.name),
                "image_rel_path_from_root": input_folder_rel_path_from_input_root,
                "image_id": image_id,
                "error": str(e),
                "parsed_dialogue": []
            }

    def process_batch(self, folder_path: Path, input_folder_rel_path_from_input_root: Path, run_id: Optional[str]=None) -> Dict[str, List[Dict]]:
        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid folder path: {folder}")

        image_paths = sorted(
            list(folder.rglob("*.jpg")) + list(folder.rglob("*.png")),
            key=lambda p: natural_key(p.name)
        )

        grouped_results = defaultdict(list)

        print(f"🔎 Found {len(image_paths)} image(s) under: {input_folder_rel_path_from_input_root}")
        for img_path in image_paths:
            result = self.process_image(
                img_path,
                base_folder=folder,
                input_folder_rel_path_from_input_root=img_path.parent.relative_to(self.input_folder),
                run_id=run_id
            )
            rel_folder = str(img_path.parent.relative_to(self.input_folder))
            grouped_results[rel_folder].append(result)

        return grouped_results

    def save_output(self, grouped_results: Dict[str, List[Dict]], run_name: Optional[str] = None):
        if not run_name:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        base_out = self.output_base / run_name
        base_out.mkdir(parents=True, exist_ok=True)

        for rel_path, results in grouped_results.items():
            out_dir = base_out / rel_path
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / "ocr_output.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved grouped OCR outputs to: {base_out}")

