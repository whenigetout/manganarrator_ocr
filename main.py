from pathlib import Path
import os
from app.ocr_runner import OCRProcessor
from app.special_utils.paddle_bbox_mapper import PaddleBBoxMapper

# ===============================
# 🔧 DEBUG TOGGLES (EDIT HERE)
# ===============================
MOCK_MODE = True          # ← flip True / False
CONFIG_PATH = Path(__file__).parent / "config.yaml"
# ===============================

from ocr_server import mapPaddleBBoxes, save_checkpoint
import app.models.domain_states as ds

def test_paddle_mapping():
    # 
    processor = OCRProcessor(str(CONFIG_PATH))
    bbox_mapper = PaddleBBoxMapper(debug=False)

    new_json_path = str("/mnt/e/pcc_shared/manga_narrator_runs/outputs/api_batch_20260105_071144_be083864/test/ocr_output_with_paddle.json").strip()
    # ---------------------------------------------------
    #  STEP 3: Attach Paddle B-Boxes to Qwen OCR Dialogues (new robust mapper)
    # ---------------------------------------------------
    paddle_augmented_ocrrun = mapPaddleBBoxes(
        new_json_path, 
        bbox_mapper=bbox_mapper,
    )

    # VALIDATE before saving
    try:
        pass
        # paddle_ready_ocr = ds.require_paddle_ready_ocrrun(paddle_augmented_ocrrun)
    except Exception as e:
        save_checkpoint(paddle_augmented_ocrrun, error=str(e))
        raise

def test_item_mapping():
    bbox_mapper = PaddleBBoxMapper(debug=False)

    import json
    with open("local_tmp/data.json", "r", encoding="utf-8") as f:
        item_dict = json.load(f)
    
    bbox_mapper.map_image_item(item=item_dict)

def main():
    if MOCK_MODE:
        os.environ["MOCK_OCR"] = "1"
    else:
        os.environ.pop("MOCK_OCR", None)

    print(f"📖 Config: {CONFIG_PATH}")
    print(f"🧪 Mock mode: {'ON' if MOCK_MODE else 'OFF'}")

    test_paddle_mapping()

if __name__ == "__main__":
    main()
