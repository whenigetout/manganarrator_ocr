from fastapi import FastAPI, UploadFile, File, Form, Query, Request
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from pathlib import Path
from app.ocr_runner import OCRProcessor
from typing import Optional, Any
import uuid
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from app.utils import log_exception, preprocess_and_split_tall_images
from glob import glob
import httpx 
from app.special_utils.paddle_bbox_mapper import PaddleBBoxMapper
from app.models.domain import OCRRunError, MediaRef, PaddleAugmentedOCRRunResponse, PaddleOCRImage, OCRRunResponse, MediaNamespace
import app.models.domain_states as ds
from app.utils import save_model_json

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context Manager for application lifespan.
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    # print("Application startup: Initializing resources...")
    app.state.processor = OCRProcessor("config.yaml")
    app.state.bbox_mapper = PaddleBBoxMapper(debug=False) # debug off so that it's not noisy in the terminal
    yield  # The application starts serving requests here
    # print("Application shutdown: Cleaning up resources...")
    # # Clean up resources
    # print("Resources cleaned up.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_checkpoint(ocrrun: OCRRunResponse | PaddleAugmentedOCRRunResponse, error: str):
    '''
    Placeholder for a "save checkpoint" fn which allows RESUMING a batch after failure
    '''
    pass

def ocr_image_to_bbox_item(img: PaddleOCRImage) -> dict:
    """
    Adapter: OCRImage / PaddleOCRImage -> dict expected by PaddleBBoxMapper
    """
    return {
        "image_file_name": "",
        "image_rel_path_from_root": "",
        "parsed_dialogue": [
            {
                "id": dlg.id,
                "text": dlg.text,
                # IMPORTANT: mapper will ADD paddle_bbox here
            }
            for dlg in (img.parsedDialogueLines or [])
        ],
        "paddleocr_result": img.paddleocr_result,
    }

def annotatePaddleBBoxes(
        paddle_augmented_ocrrun: PaddleAugmentedOCRRunResponse, 
        final_json_path: Path,
        processor: OCRProcessor,
        bbox_mapper: PaddleBBoxMapper):
    image_root = Path(
                str(processor.config.get("input_root_folder"))
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

def mapPaddleBBoxes(
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

@app.get(
        "/paddle_augmented_ocr_run",
    response_model=PaddleAugmentedOCRRunResponse,
    include_in_schema=True,
)
def schema_paddle_augmented_ocr_run():
    """
    Schema-only endpoint.

    This endpoint exists solely to expose the
    PaddleAugmentedOCRRunResponse model to OpenAPI
    so frontend tooling can generate types.

    Do not call this endpoint at runtime.
    """
    raise RuntimeError("Schema-only endpoint. Do not call.")

@app.post("/ocr/run_paddle")
async def run_paddle(
    request: Request,
    input_path: str = Form(
        ..., 
        description="Relative path of json file under media_root/outputs/, e.g. test_mangas/test_manga1/ocr_json.json"
    ),
    annotate_bboxes: Optional[bool] = Query(False)
):

    try:

        json_pre_paddle_Ref: MediaRef = MediaRef(
            namespace=MediaNamespace.OUTPUTS,
            path=input_path
        )

        processor: OCRProcessor = request.app.state.processor
        bbox_mapper: PaddleBBoxMapper = request.app.state.bbox_mapper

        json_pre_paddle_path = json_pre_paddle_Ref.resolve(Path(processor.media_root))
        if not json_pre_paddle_path.exists() or not json_pre_paddle_path.is_file():
            return JSONResponse(status_code=400, content={"error": "Invalid file path"})

        paddle_augmented_json: MediaRef
        paddle_augmented_json, paddle_augm_ocr_run = await processor.run_batch_paddle(
            json_path=json_pre_paddle_path,
            bbox_mapper=bbox_mapper,
            annotate_bboxes=annotate_bboxes
        )

        response = {
            "status": "success",
            "run_id": paddle_augm_ocr_run.run_id,
            "count": len(paddle_augm_ocr_run.imageResults) if paddle_augm_ocr_run.imageResults else 0,
            "json_pre_paddle": json_pre_paddle_Ref,
            "json_post_paddle": paddle_augmented_json
        }

        return response

    except Exception as e:
        log_exception("Exception during batch processing:")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ocr/folder")
async def ocr_from_folder(
    request: Request,
    input_path: str = Form(
        ..., 
        description="Relative path under media_root/inputs/, e.g. test_mangas/test_manga1"
    ),
    output_all_results_to_json: Optional[bool] = Query(False),
    attach_bboxes: Optional[bool] = Query(False), #This will run PaddleOcr if True
    annotate_bboxes: Optional[bool] = Query(False),   
    custom_prompt: Optional[str] = None
):

    try:

        folderRef: MediaRef = MediaRef(
            namespace=MediaNamespace.INPUTS,
            path=input_path
        )

        processor: OCRProcessor = request.app.state.processor
        bbox_mapper: PaddleBBoxMapper = request.app.state.bbox_mapper

        folder_path = folderRef.resolve(Path(processor.media_root))
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse(status_code=400, content={"error": "Invalid folder path"})

        run_id, ocrrun, saved_ocr_json_ref, saved_json_path = processor.run_ocr_batch_pre_paddle(
            folderRef=folderRef,
            custom_prompt=custom_prompt
        )

        paddle_augmented_json: MediaRef
        if attach_bboxes:
            paddle_augmented_json, paddle_augm_ocr_run = await processor.run_batch_paddle(
                json_path=saved_json_path,
                bbox_mapper=bbox_mapper,
                annotate_bboxes=annotate_bboxes
            )

        response = {
            "status": "success",
            "run_id": run_id,
            "count": len(ocrrun.imageResults) if ocrrun.imageResults else 0,
            "json_pre_paddle": saved_ocr_json_ref,
            "json_post_paddle": paddle_augmented_json if attach_bboxes else None
        }
        if output_all_results_to_json:
            response["ocr_run"] = ocrrun
        return response

    except Exception as e:
        log_exception("Exception during batch processing:")
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/ocr/results")
async def get_ocr_results(
    request: Request,
    run_id: str = Query(..., description="The run ID of the OCR batch"),
):
    try:
        processor = request.app.state.processor
        base_path = Path(processor.media_root) / "outputs" / run_id
        if not base_path.exists():
            return JSONResponse(status_code=404, content={"error": "Run ID not found"})

        # Find all nested ocr_output.json files
        ocr_files = list(base_path.rglob("ocr_output.json"))

        if not ocr_files:
            return JSONResponse(status_code=404, content={"error": "No OCR outputs found in run folder"})

        results = []
        for path in ocr_files:
            rel_path = path.relative_to(base_path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "path": str(rel_path),
                    "count": len(data)
                })
            except Exception as e:
                results.append({
                    "path": str(rel_path),
                    "error": f"Failed to read file: {str(e)}"
                })

        return {
            "status": "success",
            "run_id": run_id,
            "files": results,
            "total": sum(item.get("count", 0) for item in results if "count" in item)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/debug/map_paddle_bboxes")
async def debug_map_paddle_bboxes(
    request: Request,
    json_path: str = Form(
        ...,
        description="Path to *_with_paddle.json or paddle-augmented OCR JSON"
    ),
    annotate_bboxes: bool = Form(
        False,
        description="Whether to also save annotated images"
    )
):
    """
    Debug endpoint to test mapPaddleBBoxes() in isolation.
    Calls the existing function directly. No duplicated logic.
    """
    try:
        bbox_mapper = request.app.state.bbox_mapper
        json_path_ = Path(json_path).resolve()
        if not json_path_.exists():
            return JSONResponse(
                status_code=400,
                content={"error": f"JSON file not found: {json_path_}"}
            )

        # 🔑 reuse existing logic — single source of truth
        mapPaddleBBoxes(
            new_json_path=str(json_path_),
            bbox_mapper=bbox_mapper,
        )

        return {
            "status": "success",
            "input_json": str(json_path_),
            "message": "mapPaddleBBoxes executed successfully"
        }

    except Exception as e:
        log_exception("debug_map_paddle_bboxes")
        return JSONResponse(status_code=500, content={"error": str(e)})
