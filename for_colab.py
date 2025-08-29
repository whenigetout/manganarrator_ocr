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
    import argparse, inspect, sys, json
    from pathlib import Path

    # --- robust bool parser ---
    def str2bool(v):
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        if s in {"1","true","t","yes","y","on"}:  return True
        if s in {"0","false","f","no","n","off"}: return False
        raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")

    parser = argparse.ArgumentParser(
        description="Thin CLI wrapper: call ocr_from_folder(...) with provided args."
    )
    # common args (aliases supported)
    parser.add_argument("--input-dir",  "--input_dir",  dest="input_dir",  required=True,
                        help="Folder containing images (can be nested if --recursive).")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True,
                        help="Folder to write outputs (JSON, etc.).")

    # optional flags (defaults are conservative and can be changed)
    parser.add_argument("--attach-bboxes", "--attach_bboxes", dest="attach_bboxes",
                        type=str2bool, default=False,
                        help="If true, attach PaddleOCR bboxes (your function must support it).")
    parser.add_argument("--recursive", type=str2bool, default=True,
                        help="Recurse into subfolders.")
    parser.add_argument("--run-id", "--run_id", dest="run_id", default=None,
                        help="Optional run_id to tag outputs.")
    parser.add_argument("--write-after-each-page", "--write_after_each_page",
                        dest="write_after_each_page", type=str2bool, default=True,
                        help="If supported, write incremental JSON after each page.")

    # you can add more flags here if your function accepts them, e.g. min/max chunk
    parser.add_argument("--min-chunk", "--min_chunk", dest="min_chunk", type=int, default=None)
    parser.add_argument("--max-chunk", "--max_chunk", dest="max_chunk", type=int, default=None)

    args = parser.parse_args()

    # normalize basic path-like args to strings (most fns accept str fine)
    args.input_dir  = str(Path(args.input_dir))
    args.output_dir = str(Path(args.output_dir))

    # import the function from this file's namespace
    try:
        fn = ocr_from_folder  # defined earlier in for_colab.py
    except NameError as e:
        print("ERROR: ocr_from_folder(...) is not defined in this module.", file=sys.stderr)
        raise

    # only pass kwargs that the function actually declares
    sig = inspect.signature(fn)
    all_args = vars(args)
    call_kwargs = {k: v for k, v in all_args.items()
                   if (v is not None) and (k in sig.parameters)}

    try:
        result = fn(**call_kwargs)
        # pretty-print if the function returns something JSONable
        try:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception:
            print(result if result is not None else "Done.")
    except Exception as exc:
        print("ERROR during ocr_from_folder(...):", exc, file=sys.stderr)
        raise
# ============================================================================
