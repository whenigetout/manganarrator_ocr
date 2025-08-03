import os
import json
import time
import re
from typing import List, Dict
from pathlib import Path
import shutil
# For rich spinner/loading spinner in console
from rich.console import Console

import traceback

def log_exception(context: str = "Unhandled exception", label: str = "💀"):
    print(f"\n{label} {context}:")
    traceback.print_exc()

def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_jsonl(results, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class Timer:
    last_duration = 0.0

    def __init__(self, label: str = "", use_spinner: bool = True):
        self.label = label
        self.start_time = None
        self.use_spinner = use_spinner
        self.console = Console()
        self.status = None

    def __enter__(self):
        if self.use_spinner:
            self.status = self.console.status(
                f"[bold cyan]{self.label}...[/]",
                spinner="bouncingBar",
                spinner_style="bold green",
            )
            self.status.__enter__()

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        Timer.last_duration = duration

        if self.use_spinner and self.status:
            self.status.__exit__(exc_type, exc_val, exc_tb)

        if self.label:
            self.console.print(
                f"✅ [green]{self.label}[/] done in [yellow]{duration:.2f}s[/]"
            )


def parse_dialogue(text: str, image_id: str, image_file_name: str, input_folder_rel_path_from_input_root: str) -> List[Dict]:
    """
    Parses raw OCR text output into structured JSON.
    Tries strict regex first, then falls back to a more permissive format.
    """
    dialogue = []
    strict_pattern = re.compile(
        r'^\s*\[(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\]\s*[:：]\s*["“”]?(.*?)["“”]?$',
        re.IGNORECASE
    )
    loose_pattern = re.compile(
        r'^\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*[:：]\s*["“”]?(.*?)["“”]?$',
        re.IGNORECASE
    )

    for i, line in enumerate(text.strip().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        match = strict_pattern.match(line)
        if match:
            speaker, gender, emotion, content = match.groups()
        else:
            match = loose_pattern.match(line)
            if match:
                speaker, gender, emotion, content = match.groups()
                print(f"⚠️ Loose parse used for line {i}: {line}")
            else:
                print(f"❌ Failed to parse line {i}: {line}")
                continue

        dialogue.append({
            "id": i,
            "image_id": image_id,
            "image_file_name": image_file_name,
            "image_rel_path_from_root": input_folder_rel_path_from_input_root,
            "speaker": speaker.strip(),
            "gender": gender.strip(),
            "emotion": emotion.strip(),
            "text": content.strip()
        })

    if not dialogue:
        print("⚠️ No valid dialogue lines found — check input format.")
    return dialogue


def clear_folders(folder_names=["input", "output"]) -> None:

    root = Path(__file__).parent.parent
    for name in folder_names:
        folder = root / name
        if not folder.exists():
            print(f"⚠️  Missing folder: {folder}")
            continue
        for item in folder.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                print(f"✅ Deleted: {item}")
            except Exception as e:
                print(f"❌ Failed: {item} — {e}")
