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
from app.models.domain import MediaRef, PaddleAugmentedOCRRunResponse, PaddleOCRImage, OCRRunResponse, MediaNamespace, scale_paddle_bbox_to_original
import app.models.domain_states as ds
from app.utils import save_model_json
import mn_contracts.ocr as contract_ocr
from mn_contracts.ocr import OCRRun
from app.models.domain_to_contract import paddle_augmented_run_to_ocr_run

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

def force_attach_original_bboxes(
    run: PaddleAugmentedOCRRunResponse,
) -> None:
    """
    Mutates the run IN PLACE.
    Ensures every dialogue line with a paddlebbox
    gets an original_bbox.
    """

    if not run.imageResults:
        return

    for img in run.imageResults:
        if (
            img.paddleResizeInfo is None
            or not img.has_text
            or not img.parsedDialogueLines
        ):
            continue

        for line in img.parsedDialogueLines:
            if (
                line.paddlebbox is not None
                and line.original_bbox is None
            ):
                line.original_bbox = scale_paddle_bbox_to_original(
                    line.paddlebbox,
                    img.paddleResizeInfo,
                )


def domain_run_to_contract_ocr_run(
    run: OCRRunResponse | PaddleAugmentedOCRRunResponse,
    ocrrun_json_path: Path,
    outputs_namespace_path: Path,
) -> contract_ocr.OCRRun:
    images: list[contract_ocr.OCRImage] = []
    for img in run.imageResults or []:
        if img.inferImageRes is None:
            raise ValueError(f"Cannot convert image {img.image_id}: missing image info")

        dialogue_lines: list[contract_ocr.DialogueLine] = []
        for line in img.parsedDialogueLines or []:
            bbox = getattr(line, "original_bbox", None)
            dialogue_lines.append(contract_ocr.DialogueLine(
                id=int(line.id),
                image_id=int(line.image_id),
                speaker=line.speaker,
                gender=line.gender,
                emotion=line.emotion,
                text=line.text,
                status=getattr(line, "status", "ok"),
                error=getattr(line, "error", None),
                original_bbox=contract_ocr.OriginalImageBBox(
                    x1=bbox.x1,
                    y1=bbox.y1,
                    x2=bbox.x2,
                    y2=bbox.y2,
                ) if bbox is not None else None,
            ))

        images.append(contract_ocr.OCRImage(
            image_id=int(img.image_id),
            has_text=img.has_text,
            image_info=contract_ocr.ImageInfo(
                image_ref=contract_ocr.MediaRef(
                    namespace=contract_ocr.MediaNamespace.INPUTS,
                    path=img.inferImageRes.image_ref.path,
                ),
                image_width=img.inferImageRes.image_width,
                image_height=img.inferImageRes.image_height,
            ),
            dialogue_lines=dialogue_lines,
        ))

    return contract_ocr.OCRRun(
        run_id=run.run_id,
        error=run.error,
        ocr_json_file=contract_ocr.MediaRef(
            namespace=contract_ocr.MediaNamespace.OUTPUTS,
            path=ocrrun_json_path.relative_to(outputs_namespace_path).as_posix(),
        ),
        images=images,
    )
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
        force_attach_original_bboxes(paddle_augm_ocr_run)

        ocrrun_json_path = paddle_augmented_json.resolve(Path(processor.media_root)).parent / "ocrrun.json"
        ocrrun_json_namespace_path = Path(processor.media_root) / MediaNamespace.OUTPUTS.value
        ocrrun: OCRRun = paddle_augmented_run_to_ocr_run(paddle_augm_ocr_run, ocrrun_json_path, ocrrun_json_namespace_path)
        
        save_model_json(
            model=ocrrun,
            json_path=ocrrun_json_path
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

        source_for_editor: OCRRunResponse | PaddleAugmentedOCRRunResponse = ocrrun
        paddle_augmented_json: MediaRef | None = None
        if attach_bboxes:
            paddle_augmented_json, paddle_augm_ocr_run = await processor.run_batch_paddle(
                json_path=saved_json_path,
                bbox_mapper=bbox_mapper,
                annotate_bboxes=annotate_bboxes
            )
            force_attach_original_bboxes(paddle_augm_ocr_run)
            source_for_editor = paddle_augm_ocr_run

        outputs_namespace_path = Path(processor.media_root) / MediaNamespace.OUTPUTS.value
        ocrrun_json_path = saved_json_path.parent / "ocrrun.json"
        contract_ocrrun = domain_run_to_contract_ocr_run(
            source_for_editor,
            ocrrun_json_path,
            outputs_namespace_path,
        )
        save_model_json(
            model=contract_ocrrun,
            json_path=ocrrun_json_path
        )

        response = {
            "status": "success",
            "run_id": run_id,
            "count": len(contract_ocrrun.images),
            "ocr_json_file": contract_ocrrun.ocr_json_file,
            "json_pre_paddle": saved_ocr_json_ref,
            "json_post_paddle": paddle_augmented_json if attach_bboxes else None
        }
        if output_all_results_to_json:
            response["ocr_run"] = contract_ocrrun
        return response

    except Exception as e:
        log_exception("Exception during batch processing:")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/debug/annotate_bboxes")
async def annotate_bboxes(
    request: Request,
    input_path: str = Form(
        ..., 
        description="Relative path of json file under media_root/outputs/, e.g. test_mangas/test_manga1/ocr_json_with_bboxes.json"
    ),
):

    try:

        json_with_bboxes_Ref: MediaRef = MediaRef(
            namespace=MediaNamespace.OUTPUTS,
            path=input_path
        )

        processor: OCRProcessor = request.app.state.processor
        bbox_mapper: PaddleBBoxMapper = request.app.state.bbox_mapper

        json_with_bboxes_path = json_with_bboxes_Ref.resolve(Path(processor.media_root))
        if not json_with_bboxes_path.exists() or not json_with_bboxes_path.is_file():
            return JSONResponse(status_code=400, content={"error": "Invalid file path"})
        
        with open(json_with_bboxes_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

            ocr_run = PaddleAugmentedOCRRunResponse.model_validate(raw)

        res = processor.annotatePaddleBBoxes(
            ocr_run,
            json_with_bboxes_path,
            bbox_mapper
        )

        response = {
            "status": "success",
        }

        return response

    except Exception as e:
        log_exception("Exception during batch processing:")
        return JSONResponse(status_code=500, content={"error": str(e)})
