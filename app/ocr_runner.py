from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from .config import load_config
from .utils import ensure_output_dir, save_jsonl, Timer
from .backends.qwen_backend import QwenOCRBackend
import uuid
import json
from datetime import datetime
from app.utils import parse_dialogue, save_model_json
import re
import os
from collections import defaultdict
from PIL import Image
from app.models.domain import (
    MediaRef, InferImageResponse,  DialogueLineResponse, 
    OCRImage,  OCRRunResponse,
    MediaNamespace, PaddleAugmentedOCRRunResponse
    )
from app.models.exceptions import (
    InferImageError,ProcessImageError, OCRRunError, SaveJSONError, PaddleAugmentationError
)
import app.utils as utils
import app.models.domain_states as ds
import httpx
from app.special_utils.paddle_bbox_mapper import PaddleBBoxMapper

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

    def save_checkpoint(self, ocrrun: OCRRunResponse | PaddleAugmentedOCRRunResponse, error: str):
        '''
        Placeholder for a "save checkpoint" fn which allows RESUMING a batch after failure
        '''
        pass

    def run_ocr_batch_pre_paddle(self, 
                            folderRef: MediaRef,
                            custom_prompt: Optional[str] = None
                            ) -> Tuple[str, OCRRunResponse, MediaRef, Path]:
        try:
            folder_path = folderRef.resolve(Path(self.media_root))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"api_batch_{timestamp}_{uuid.uuid4().hex[:8]}"

            config_min_img_chunk_size = self.config.get("min_chunk_size", 4800)
            config_max_img_chunk_size = self.config.get("max_chunk_size", 7000)

            utils.preprocess_and_split_tall_images(folderRef=folderRef, 
                                            media_root=self.media_root,
                                            max_chunk=config_max_img_chunk_size,
                                            min_chunk=config_min_img_chunk_size)
            
            ocrrun = self.process_batch_pre_paddle(
                                            folderRef,
                                            run_id=run_id,
                                            prompt=custom_prompt
                )
            
            # This is an intentional try-catch to allow resuming failed batches in the future
            # any objects/values inside this block are for validation only, not to be used
            # or exported elsewhere
            try:
                # -------- Boundary 1: OCRRun is now consumable --------
                images = ds.require_images(ocrrun)

                for img in images:
                    inferred = ds.require_inferred(img)
                    parsed = ds.require_parsed(inferred)

            except Exception as e:
                self.save_checkpoint(ocrrun, error=str(e))
                raise

            saved_ocr_json_ref = self.save_output_pre_paddle(ocrrun, manga_folder_ref=folderRef)
            saved_json_path = saved_ocr_json_ref.resolve(Path(self.media_root))
            return run_id, ocrrun, saved_ocr_json_ref, saved_json_path
        except:
            raise

    def process_image_pre_paddle(self, imgRef: MediaRef, 
                                image_id: str,
                      prompt: Optional[str]=None
                      ) -> OCRImage:
        try:
            img_path = imgRef.resolve(Path(self.media_root))
            img_name = img_path.name

            with Timer(f"🖼️ Process Image {img_name}"):
                if self.mock_mode:
                    result = InferImageResponse(
                        image_ref=MediaRef(
                            namespace=MediaNamespace.INPUTS,
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

    def process_batch_pre_paddle(self, 
                      inputfolderRef: MediaRef, 
                      run_id: Optional[str]=None,
                      prompt: Optional[str]=None
                      ) -> OCRRunResponse:
        try:
            folder_path = inputfolderRef.resolve(Path(self.media_root))
            folder_path.resolve()
            if not folder_path.exists() or not folder_path.is_dir():
                raise OCRRunError(f"Invalid folder path: {folder_path}")

            image_paths = sorted(
                list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.png")),
                key=lambda p: natural_key(p.name)
            )


            print(f"🔎 Found {len(image_paths)} image(s) under: {folder_path}")
            processImageResults = []
            for idx, img_path in enumerate(image_paths):
                path_base = inputfolderRef.namespace_path(Path(self.media_root))
                imgRef = MediaRef(
                    namespace=inputfolderRef.namespace,
                    path=str(img_path.relative_to(path_base))
                )
                processImageResult = self.process_image_pre_paddle(
                    imgRef=imgRef,
                    image_id=str(idx + 1),
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

    def save_output_pre_paddle(self, ocr_run: OCRRunResponse, manga_folder_ref: MediaRef, run_name: Optional[str] = None) -> MediaRef:
        try:
            run_id = ocr_run.run_id 
            if not run_id:
                run_id = run_name if run_name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            base_out = Path(self.media_root)/ MediaNamespace.OUTPUTS.value / run_id
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

            save_model_json(
                model=ocr_run,
                json_path=out_file
            )

            media_outputs_Path = Path(self.media_root) / MediaNamespace.OUTPUTS.value
            out_file_ref = MediaRef(
                namespace=MediaNamespace.OUTPUTS,
                path=str(out_file.relative_to(media_outputs_Path))
            )

            print(f"✅ Saved OCR outputs to: {out_dir}")
            return out_file_ref

        except Exception as e:
            raise SaveJSONError(f"Failed to save JSON for run_id/Output Folder: {out_dir}")
        

    def annotatePaddleBBoxes(self,
            paddle_augmented_ocrrun: PaddleAugmentedOCRRunResponse, 
            final_json_path: Path,
            bbox_mapper: PaddleBBoxMapper):
        image_root = Path(
                    str(self.config.get("input_root_folder"))
                ).resolve()
        out_dir_for_images = final_json_path.parent

        annotation_items = []

        for img in paddle_augmented_ocrrun.imageResults or []:
            if not img.parsedDialogueLines or not img.inferImageRes or not img.inferImageRes.image_ref:
                continue

            imgRef = img.inferImageRes.image_ref
            img_file_name = Path(imgRef.path).name
            image_rel_path_from_root = str(Path(img.inferImageRes.image_ref.path).parent)

            annotation_items.append({
                "image_file_name": img_file_name,  # IMPORTANT: must match actual filename
                "image_rel_path_from_root": image_rel_path_from_root,    # OK if image_root already points correctly
                "parsed_dialogue": [
                    {
                        "id": dlg.id,
                        "paddle_bbox": dlg.paddlebbox.model_dump()
                        if dlg.paddlebbox else None,
                    }
                    for dlg in img.parsedDialogueLines
                    if dlg.paddlebbox
                ],
            })

        saved_imgs = bbox_mapper.annotate_batch(
            annotation_items,
            image_root=image_root,
            out_dir=out_dir_for_images,
        )

        for p in saved_imgs:
            print(f"🖼️ Annotated image saved: {p}")

    def mapPaddleBBoxes(self,
            new_json_path: str, 
            bbox_mapper: PaddleBBoxMapper,
            ):
        with open(new_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

            raw["ocr_json_file"] = MediaRef(
                namespace=MediaNamespace.OUTPUTS,
                path=""
            )
            ocr_run = PaddleAugmentedOCRRunResponse.model_validate(raw)
            paddle_augmented_ocrrun = bbox_mapper.map_and_save_paddle_bboxes(ocr_run, Path(new_json_path))

            return paddle_augmented_ocrrun


    async def run_batch_paddle(self,
                               json_path: Path,
                                bbox_mapper: PaddleBBoxMapper,
                                annotate_bboxes: Optional[bool] = False
                         ) -> Tuple[MediaRef, PaddleAugmentedOCRRunResponse]:
        try:
            async with httpx.AsyncClient() as client:
                json_rel_path = str(json_path.relative_to(Path(self.media_root)/MediaNamespace.OUTPUTS.value))
                resp = await client.post(
                        self.paddle_ocr_api,  # URL from config
                        data={"ocr_json_path": json_path}
                    )
                if resp.status_code != 200:
                    raise PaddleAugmentationError(f"⚠️ PaddleOCR augmentation failed for {json_path}: {resp.text}")
                
                data = resp.json()
                new_json_path = str(data["output_file"]).strip()
                # ---------------------------------------------------
                #  STEP 3: Attach Paddle B-Boxes to Qwen OCR Dialogues (new robust mapper)
                # ---------------------------------------------------
                paddle_augmented_ocrrun = self.mapPaddleBBoxes(
                    new_json_path, 
                    bbox_mapper=bbox_mapper,
                )

                final_json_path = Path(new_json_path).parent / "ocr_output_with_bboxes.json"
                rel_path = final_json_path.relative_to(Path(self.media_root)/MediaNamespace.OUTPUTS.value)

                paddle_augmented_ocrrun.ocr_json_file = MediaRef(
                    namespace=MediaNamespace.OUTPUTS,
                    path=rel_path.as_posix()
                )

                # VALIDATE before saving
                try:
                    paddle_ready_ocr = ds.require_paddle_ready_ocrrun(paddle_augmented_ocrrun)
                except Exception as e:
                    self.save_checkpoint(paddle_augmented_ocrrun, error=str(e))
                    raise

                # Save final JSON with bboxes (same path pattern as before)
                json_with_bboxes_attached = save_model_json(
                    model=paddle_augmented_ocrrun,
                    json_path=final_json_path
                )
                print(f"✅ Final JSON with bboxes saved: {final_json_path}")

                # Optionally render annotated image(s) right next to the JSON ---
                if annotate_bboxes:
                    self.annotatePaddleBBoxes(
                        paddle_augmented_ocrrun=paddle_augmented_ocrrun,
                        final_json_path=final_json_path,
                        bbox_mapper=bbox_mapper
                    )

            return paddle_augmented_ocrrun.ocr_json_file, paddle_augmented_ocrrun
        except:
            raise