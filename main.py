import argparse
from app.ocr_runner import OCRProcessor

def main():
    parser = argparse.ArgumentParser(description="MangaNarrator OCR Processor")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    print(f"\n📖 Loading config: {args.config}")
    processor = OCRProcessor(args.config)

    prompt = '''The following is a manhwa panel.
Extract the dialogue lines and output them in the following format:
[SPEAKER | GENDER | EMOTION | BBOX_2D]: "TEXT"
SPEAKER = "Speaker 1", "Speaker 2", "Narrator", etc.
GENDER = "male", "female", or "unknown"
EMOTION = Pick from: neutral, happy, sad, angry, excited, nervous, aroused, scared, curious, playful, serious, calm
BBOX_2D = Outline coordinates of the dialogue line

Preserve the original order. Output only the formatted lines.
'''

    print(f"\n📁 Starting batch OCR for folder: {processor.input_folder}")
    results = processor.process_batch(processor.input_folder, '', prompt=prompt)

    processor.save_output(results)

if __name__ == "__main__":
    main()
