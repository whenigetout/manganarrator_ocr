from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from app.ocr_runner import OCRProcessor
from typing import Optional
import tempfile
import shutil
import uuid
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from app.utils import log_exception, preprocess_and_split_tall_images
from glob import glob
import httpx 
from app.special_utils.paddle_bbox_mapper import PaddleBBoxMapper

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
processor = OCRProcessor("config.yaml")
bbox_mapper = PaddleBBoxMapper(debug=False)  # turn off debug in prod if noisy

@app.post("/ocr/image")
async def ocr_single_image(
    file: UploadFile = File(...),
    save_uploaded_image: Optional[bool] = Query(None),
    custom_prompt: Optional[str] = None
):
    try:
        config_saving_uploaded_image = processor.config.get("save_uploaded_images", True)
        should_save = config_saving_uploaded_image if save_uploaded_image is None else save_uploaded_image

        if should_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_id = f"api_image_{timestamp}_{uuid.uuid4().hex[:8]}"
            uploads_dir = Path(processor.input_folder) / "uploads" / folder_id
            uploads_dir.mkdir(parents=True, exist_ok=True)
            image_path = uploads_dir / file.filename
        else:
            temp_dir = Path(tempfile.mkdtemp())
            image_path = temp_dir / file.filename

        with open(image_path, "wb") as f:
            f.write(await file.read())

        result = processor.process_image(image_path, base_folder=image_path.parent, prompt=custom_prompt)

        if not should_save:
            shutil.rmtree(temp_dir)
        else:
            processor.save_output([result], run_name=folder_id)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ocr/folder")
async def ocr_from_folder(
    input_path: str = Form(
        ..., 
        description="Example: /mnt/e/pcc_shared/manga_narrator_runs/inputs/test_mangas/test_manga1"
    ),
    output_all_results_to_json: Optional[bool] = Query(False),
    attach_bboxes: Optional[bool] = Query(True),
    annotate_bboxes: Optional[bool] = Query(False),   
    custom_prompt: Optional[str] = None
):

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"api_batch_{timestamp}_{uuid.uuid4().hex[:8]}"

        folder_path = Path(input_path).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse(status_code=400, content={"error": "Invalid folder path"})
        
        config_input_root = Path(processor.config.get("input_root_folder")).resolve()
        if not config_input_root.exists() or not config_input_root.is_dir():
            return JSONResponse(status_code=400, content={"error": "Invalid input-root-folder path in config"})
        input_folder_rel_path_from_input_root = folder_path.relative_to(config_input_root)

        config_min_img_chunk_size = processor.config.get("min_chunk_size", 4800)
        config_max_img_chunk_size = processor.config.get("max_chunk_size", 7000)

        preprocess_and_split_tall_images(folder_path=folder_path,
                                         max_chunk=config_max_img_chunk_size,
                                         min_chunk=config_min_img_chunk_size)

        results = processor.process_batch(folder_path=folder_path, 
                                          input_folder_rel_path_from_input_root=input_folder_rel_path_from_input_root,
                                          run_id=run_id,
                                          prompt=custom_prompt)

        out_dir = Path(processor.output_base) / run_id
        processor.save_output(results, run_name=run_id)

        # ---------------------------------------------------
        #  STEP 2: call PaddleOCR augmentation
        # ---------------------------------------------------
        if attach_bboxes:
            for json_path in out_dir.rglob("ocr_output.json"):
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        processor.paddle_ocr_api,  # URL from config
                        data={"ocr_json_path": str(json_path)}
                    )
                if resp.status_code != 200:
                    print(f"⚠️ PaddleOCR augmentation failed for {json_path}: {resp.text}")
                    continue
                data = resp.json()
                new_json_path = Path(data["output_file"])
                # ---------------------------------------------------
                #  STEP 3: Attach Paddle B-Boxes to Qwen OCR Dialogues (new robust mapper)
                # ---------------------------------------------------
                with open(new_json_path, "r", encoding="utf-8") as f:
                    paddle_data = json.load(f)

                # Map in-place; prints per-dialogue debug if bbox_mapper.debug = True
                paddle_data = bbox_mapper.map_batch(paddle_data if isinstance(paddle_data, list) else [paddle_data])

                # Save final JSON with bboxes (same path pattern as before)
                final_json_path = Path(new_json_path).parent / "ocr_output_with_bboxes.json"
                with open(final_json_path, "w", encoding="utf-8") as f:
                    json.dump(paddle_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Final JSON with bboxes saved: {final_json_path}")

                # Optionally render annotated image(s) right next to the JSON ---
                if annotate_bboxes:
                    image_root = Path(processor.config.get("input_root_folder")).resolve()
                    out_dir_for_images = final_json_path.parent
                    items = paddle_data if isinstance(paddle_data, list) else [paddle_data]
                    saved_imgs = bbox_mapper.annotate_batch(items, image_root=image_root, out_dir=out_dir_for_images)
                    for p in saved_imgs:
                        print(f"🖼️ Annotated image saved: {p}")
        # ---------------------------------------------------

        response = {
            "status": "success",
            "run_id": run_id,
            "count": len(results),
        }
        if output_all_results_to_json:
            response["results"] = results
        return response

    except Exception as e:
        log_exception("Exception during batch processing:")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/ocr/results")
async def get_ocr_results(
    run_id: str = Query(..., description="The run ID of the OCR batch"),
):
    try:
        base_path = Path(processor.output_base) / run_id
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
