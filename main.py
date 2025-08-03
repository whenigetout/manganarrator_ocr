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

    print(f"\n📁 Starting batch OCR for folder: {processor.input_folder}")
    results = processor.process_batch()

    processor.save_output(results)

if __name__ == "__main__":
    main()
