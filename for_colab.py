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
import asyncio, inspect

processor = OCRProcessor("config.yaml")
bbox_mapper = PaddleBBoxMapper(debug=False)  # turn off debug in prod if noisy

async def ocr_from_folder(
    input_path: str,
    output_all_results_to_json: Optional[bool] = False,
    attach_bboxes: Optional[bool] = False,
    annotate_bboxes: Optional[bool] = False,   
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

# =================== CLI passthrough for ocr_from_folder ===================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run OCR on a folder (no servers).")

    # IMPORTANT: map all common flags to the SAME dest: input_path
    parser.add_argument("--input-path", "--input_path", "--input-dir", "--input_dir",
                        dest="input_path", required=True,
                        help="Folder with images (can be nested if --recursive).")

    # output_dir is optional
    parser.add_argument("--output-dir", "--output_dir",
                        dest="output_dir", required=False, default=None,
                        help="Optional output folder override.")

    # whatever flags you already use — examples:
    parser.add_argument("--attach-bboxes", "--attach_bboxes",
                        dest="attach_bboxes", action="store_true",
                        help="Attach Paddle bboxes (leave off to skip).")
    parser.add_argument("--recursive", action="store_true", default=True)

    args = parser.parse_args()

    # normalize paths
    args.input_path = str(Path(args.input_path))
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir))

    # call your function EXACTLY with input_path=

    if inspect.iscoroutinefunction(ocr_from_folder):
        result = asyncio.run(ocr_from_folder(
            input_path=args.input_path,
            attach_bboxes=args.attach_bboxes,
        ))
    else:
        result = ocr_from_folder(
            input_path=args.input_path,
            attach_bboxes=args.attach_bboxes,
        )

    print(result if result is not None else "Done.")

    import json
from starlette.responses import JSONResponse

# 'result' is what ocr_from_folder returns
if isinstance(result, JSONResponse):
    try:
        data = json.loads(result.body.decode("utf-8"))
    except Exception:
        # Starlette may keep .body as bytes already; fallback
        try:
            data = result.media  # if you used Starlette 0.38+
        except Exception:
            data = None

    print("API result:", data)

    # helpful logs
    if isinstance(data, dict):
        # common keys your FastAPI route likely returns — adjust if different
        out_dir  = data.get("output_dir") or data.get("output_path") or data.get("out_dir")
        written  = data.get("written") or data.get("pages") or data.get("files")
        print("Resolved output_dir:", out_dir)
        print("Written entries:", written)

    # regardless of the response, list what’s on disk so you can see files
    import glob, os
    root = os.environ.get("OUT_DIR_OVERRIDE", "")  # ignore if you didn't set this
    search_root = out_dir if os.path.isdir(out_dir) else (out_dir or ".")
    cands = glob.glob(search_root + "/**/*.json*", recursive=True)
    print(f"Found {len(cands)} JSON/JSONL files under {search_root}:")
    for p in sorted(cands)[:20]:
        print(" -", p)


