import os
import json
import time
import re
from typing import List, Union
from pathlib import Path
import shutil
import json

# For rich spinner/loading spinner in console
from rich.console import Console
from PIL import Image
from app.models.domain import DialogueLineResponse, MediaRef
from app.models.exceptions import ParseDialogueError

from pydantic import BaseModel

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

import re

def fix_casing(text):
    # If the text is all uppercase or mostly uppercase, fix it
    # You can adjust the threshold for "mostly uppercase"
    uppercase_ratio = sum(1 for c in text if c.isupper()) / (len(text) or 1)
    if uppercase_ratio > 0.7:
        text = text.lower()
    # Capitalize first letter of each sentence
    def capitalize_match(match):
        return match.group(1) + match.group(2).upper()
    # This regex finds sentence boundaries
    text = re.sub(r'(^|(?<=[.!?]\s))([a-z])', capitalize_match, text)
    return text

import threading

class Timer:
    last_duration = 0.0

    def __init__(self, label: str = "", use_spinner: bool = True):
        self.label = label
        self.start_time = None
        self.use_spinner = use_spinner
        self.console = Console()
        self.status = None
        self._stop_event = threading.Event()
        self._timer_thread = None

    def _live_counter(self):
        while not self._stop_event.is_set():
            elapsed = int(time.perf_counter()) - int(self.start_time if self.start_time else 0)
            self.console.print(
                f"[cyan]{self.label}[/] [yellow]Elapsed: {elapsed}s[/]", end="\r"
            )
            time.sleep(1)

    def __enter__(self):
        if self.use_spinner:
            self.status = self.console.status(
                f"[bold cyan]{self.label}...[/]",
                spinner="bouncingBar",
                spinner_style="bold green",
            )
            self.status.__enter__()
        self.start_time = time.perf_counter()
        self._stop_event.clear()
        self._timer_thread = threading.Thread(target=self._live_counter, daemon=True)
        self._timer_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._timer_thread is not None:
            self._timer_thread.join()
        duration = int(time.perf_counter()) - int(self.start_time if self.start_time else 0)
        Timer.last_duration = duration

        if self.use_spinner and self.status:
            self.status.__exit__(exc_type, exc_val, exc_tb)
        if self.label:
            self.console.print(
                f"\n✅ [green]{self.label}[/] done in [yellow]{duration:.2f}s[/]"
            )


def _is_no_dialogue_marker(line: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", line.casefold()).strip()
    if not normalized:
        return True
    no_dialogue_markers = {
        "no dialogue",
        "no dialogue lines",
        "no dialogue lines found",
        "no readable dialogue",
        "no readable text",
        "no text",
        "none",
        "n a",
    }
    return normalized in no_dialogue_markers


def _coerce_raw_dialogue(line: str) -> str | None:
    line = re.sub(r"^\s*[-*]\s*", "", line).strip()
    line = re.sub(r"^(?:dialogue|text|line)\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip()
    line = line.strip("\"'“”")
    if not line or _is_no_dialogue_marker(line):
        return None
    if len(line) > 220:
        return None

    alpha_chars = [c for c in line if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0
    )
    looks_spoken = (
        line.endswith(("!", "?", "..."))
        or uppercase_ratio > 0.55
        or bool(re.search(r"^[\"'“”].+[\"'“”]$", line))
    )
    if not looks_spoken:
        return None
    return line


def parse_dialogue(text: str, image_id: str) -> List[DialogueLineResponse]:
    """
    Parses raw OCR text output into structured JSON.
    Tries strict regex first, then falls back to a more permissive format.
    """
    try:
        dialogueLines = []
        strict_pattern = re.compile(
            r'^\s*\[(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\]\s*[:：]\s*["“”]?(.*?)["“”]?$',
            re.IGNORECASE
        )
        loose_pattern = re.compile(
            r'^\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*[:：]\s*["“”]?(.*?)["“”]?$',
            re.IGNORECASE
        )

        unmatched_lines: list[tuple[int, str]] = []

        for i, line in enumerate(text.strip().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if _is_no_dialogue_marker(line):
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
                    unmatched_lines.append((i, line))
                    continue

            dialogueLines.append(
                DialogueLineResponse(
                    id=len(dialogueLines) + 1,
                    image_id=image_id,
                    speaker=speaker.strip(),
                    gender=gender.strip(),
                    emotion=emotion.strip(),
                    text=fix_casing(content).strip()
                )
               )

        if not dialogueLines and unmatched_lines:
            for source_line_num, raw_line in unmatched_lines:
                content = _coerce_raw_dialogue(raw_line)
                if content is None:
                    print(f"⚠️ Ignored non-dialogue OCR line {source_line_num}: {raw_line}")
                    continue
                dialogueLines.append(
                    DialogueLineResponse(
                        id=len(dialogueLines) + 1,
                        image_id=image_id,
                        speaker="Speaker 1",
                        gender="unknown",
                        emotion="neutral",
                        text=fix_casing(content).strip(),
                    )
                )

        if not dialogueLines:
            return dialogueLines  # may be empty
        return dialogueLines
    except Exception as e:
        raise ParseDialogueError from e


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

import os
from pathlib import Path

def optimal_split(image_path, output_prefix, max_chunk=7000, min_chunk=4000):
    """
    Splits a tall image (e.g., manhwa/webtoon panel) into optimally sized vertical chunks.

    Logic:
        - If the image height is less than or equal to max_chunk, saves the image as a single chunk.
        - Otherwise, splits the image vertically into chunks of size `max_chunk` as much as possible.
        - If the remainder (last chunk) is >= min_chunk, keeps it as the last chunk.
        - If the remainder is < min_chunk, merges it into the previous chunk(s) and splits those into two nearly equal sizes.
        - Ensures no chunk is smaller than `min_chunk` or larger than `max_chunk` (except possibly for balancing the last two chunks).
        - Output files are named as `output_prefix_partN.jpg` in order, preserving the original sort order.

    Args:
        image_path (str or Path): Path to the input image.
        output_prefix (str or Path): Prefix for output split images.
        max_chunk (int): Maximum allowed chunk height (pixels).
        min_chunk (int): Minimum allowed chunk height (pixels).

    Returns:
        None. (Writes output images to disk.)
    
    Notes:
        - This is meant for use before OCR processing, so that each resulting image chunk can be processed efficiently.
        - Designed to avoid tiny, mostly-empty panels and minimize awkward splits.
    """
    img = Image.open(image_path)
    width, height = img.size
    s = height
    max = max_chunk
    min_ = min_chunk

    print(f"img height is: {s}")

    if s <= max:
        img.save(f"{output_prefix}_part1.jpg")
        print(f"Image is short ({height}px). Saved as a single part.")
        return

    q = s // max
    r = s % max

    cuts = []
    if q == 0:
        # Should not happen, as s > max
        img.save(f"{output_prefix}_part1.jpg")
        print(f"Error: q=0 for s={s}, max={max}")
        return
    elif r >= min_:
        # q chunks of max, and one r-sized chunk
        for i in range(q):
            cuts.append(max)
        cuts.append(r)
    else:
        r_new = s - (q - 1) * max
        # Optionally split last big chunk into two nearly equal chunks for better balance
        half1 = r_new // 2
        half2 = r_new - half1
        for i in range(q - 1):
            cuts.append(max)
        cuts.append(half1)
        cuts.append(half2)

    # Now save slices
    y = 0
    for idx, h in enumerate(cuts):
        img_crop = img.crop((0, y, width, y + h))
        img_crop.save(f"{output_prefix}_part{idx + 1}.jpg")
        print(f"Saved {output_prefix}_part{idx + 1}.jpg ({y} to {y + h}, size={h})")
        y += h

    print(f"Done. Total splits: {len(cuts)}")

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def _natural_path_key(path: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


def _prepare_rgb_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img.close()
        return background
    return img.convert("RGB")


def _vertical_cuts(height: int, max_chunk: int, min_chunk: int) -> list[tuple[int, int]]:
    if height <= max_chunk:
        return [(0, height)]

    q = height // max_chunk
    r = height % max_chunk
    sizes: list[int] = []

    if r >= min_chunk:
        sizes = [max_chunk] * q + [r]
    else:
        merged_tail = height - (q - 1) * max_chunk
        half1 = merged_tail // 2
        half2 = merged_tail - half1
        sizes = [max_chunk] * max(0, q - 1) + [half1, half2]

    cuts = []
    y = 0
    for size in sizes:
        cuts.append((y, min(height, y + size)))
        y += size
    return cuts


def _save_vertical_stack(image_paths: list[Path], out_path: Path) -> None:
    opened = [_prepare_rgb_image(path) for path in image_paths]
    try:
        max_width = max(img.width for img in opened)
        total_height = sum(img.height for img in opened)
        canvas = Image.new("RGB", (max_width, total_height), (255, 255, 255))
        y = 0
        for img in opened:
            x = (max_width - img.width) // 2
            canvas.paste(img, (x, y))
            y += img.height
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=95)
    finally:
        for img in opened:
            img.close()


def _save_split_image(image_path: Path, out_dir: Path, start_index: int, max_chunk: int, min_chunk: int) -> int:
    img = _prepare_rgb_image(image_path)
    try:
        next_index = start_index
        for y1, y2 in _vertical_cuts(img.height, max_chunk=max_chunk, min_chunk=min_chunk):
            out_path = out_dir / f"page_{next_index:04d}.jpg"
            img.crop((0, y1, img.width, y2)).save(out_path, quality=95)
            next_index += 1
        return next_index
    finally:
        img.close()


def preprocess_and_split_tall_images(folderRef: MediaRef, media_root: str, max_chunk, min_chunk) -> MediaRef:
    """
    Creates a generated, normalized image sequence for OCR/video.

    - Very tall source images are split into sane vertical chunks.
    - Very short consecutive source images are merged until they form a useful chunk.
    - The original scraped folder is left untouched.
    """
    folder_path = folderRef.resolve(Path(media_root))
    if Path(folderRef.path).parts[:1] == ("_normalized",):
        return folderRef

    normalized_rel_path = (Path("_normalized") / Path(folderRef.path)).as_posix()
    normalized_ref = MediaRef(namespace=folderRef.namespace, path=normalized_rel_path)
    normalized_folder = normalized_ref.resolve(Path(media_root))

    image_files = sorted(
        [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=_natural_path_key,
    )
    if not image_files:
        return folderRef

    if normalized_folder.exists():
        shutil.rmtree(normalized_folder)
    normalized_folder.mkdir(parents=True, exist_ok=True)

    merge_max_chunk = max_chunk + max(512, int(max_chunk * 0.08))
    page_index = 1
    pending_stack: list[Path] = []
    pending_height = 0

    def flush_pending() -> None:
        nonlocal page_index, pending_stack, pending_height
        if not pending_stack:
            return
        out_path = normalized_folder / f"page_{page_index:04d}.jpg"
        _save_vertical_stack(pending_stack, out_path)
        print(
            f"[Normalize] merged {len(pending_stack)} source image(s) "
            f"into {out_path.name}"
        )
        page_index += 1
        pending_stack = []
        pending_height = 0

    for img_file in image_files:
        with Image.open(img_file) as img:
            height = img.height

        if height > merge_max_chunk:
            flush_pending()
            print(f"[Normalize] splitting tall image {img_file.name}: height={height}")
            page_index = _save_split_image(
                img_file,
                normalized_folder,
                page_index,
                max_chunk=max_chunk,
                min_chunk=min_chunk,
            )
            continue

        if pending_stack and pending_height >= min_chunk and pending_height + height > merge_max_chunk:
            flush_pending()

        if pending_stack and pending_height + height > merge_max_chunk:
            flush_pending()

        pending_stack.append(img_file)
        pending_height += height

    flush_pending()

    print(f"[Preprocessing] Normalized image sequence saved to: {normalized_folder}\n")
    return normalized_ref


def preprocess_and_split_tall_images_in_place(folderRef: MediaRef, media_root: str, max_chunk, min_chunk) -> None:
    """
    Legacy mutating splitter retained for one-off scripts. The service uses
    preprocess_and_split_tall_images(), which leaves source folders untouched.
    """
    exts = IMAGE_EXTS
    folder_path = folderRef.resolve(Path(media_root))
    for img_file in sorted(folder_path.glob('*')):
        if not img_file.suffix.lower() in exts:
            continue

        with Image.open(img_file) as img:
            height = img.height
            if height <= max_chunk:
                # No split needed
                continue

        # Split and report
        base = img_file.stem
        prefix = img_file.parent / (base + "_split")
        print(f"\n[Split] {img_file.name}: height={height} > max_chunk={max_chunk}")
        optimal_split(str(img_file), str(prefix), max_chunk=max_chunk, min_chunk=min_chunk)

        # Delete the original tall image
        img_file.unlink()
        print(f"[Delete] Removed oversized original: {img_file.name}")

        # Rename splits to match original sort order (important!)
        for idx, split_file in enumerate(sorted(prefix.parent.glob(f"{prefix.stem}_part*.jpg")), 1):
            new_name = f"{base}_part{idx:02d}{img_file.suffix}"
            new_path = img_file.parent / new_name
            split_file.rename(new_path)
            print(f"[Rename] {split_file.name} → {new_name}")

        # Optionally, remove split prefix files (they were just renamed)

    print("[Preprocessing] Done splitting tall images.\n")

def save_model_json(
    model: BaseModel,
    json_path: Union[str, Path],
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """
    Save any Pydantic model to a JSON file.

    - Accepts str or Path
    - Creates parent directories if needed
    - Uses Pydantic v2 model_dump()
    - Returns the resolved output Path
    """
    path = Path(json_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            model.model_dump(),
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    return path
