# app/utils/paddle_bbox_mapper.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
# --- add these imports near the top of the file ---
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

class PaddleBBoxMapper:
    """
    Maps Qwen/parsed_dialogue lines to PaddleOCR rects.
    - Sorts OCR lines top→bottom, then left→right.
    - Filters junk rects (empty/1-char after normalization).
    - Progressively concatenates consecutive rects until the dialogue prefix matches.
    - Reserves the entire matched span so later dialogues can't reuse them.
    - Attaches the FIRST rect's bbox/poly to each dialogue as 'paddle_bbox'.
    """

    def __init__(
        self,
        min_coverage: float = 0.55,
        min_chars: int = 20,
        min_rect_norm_len: int = 2,  # filters out single glyphs like '京', '气', 'o?'
        debug: bool = False,
        debug_preview_count: int = 3,
    ) -> None:
        self.min_coverage = min_coverage
        self.min_chars = min_chars
        self.min_rect_norm_len = min_rect_norm_len
        self.debug = debug
        self.debug_preview_count = debug_preview_count

    # ---------- helpers ----------
    @staticmethod
    def _normalize(s: str) -> str:
        """lowercase & drop all non-alphanumerics (spaces/punct go away)."""
        return re.sub(r"[^0-9a-z]+", "", s.lower())

    def _sort_and_filter(self, item: Dict[str, Any]):
        rect_texts: List[str] = item["paddleocr_result"]["rec_texts"]
        rec_boxes:  List[List[int]] = item["paddleocr_result"]["rec_boxes"]
        rec_polys:  List[List[List[int]]] = item["paddleocr_result"]["rec_polys"]

        # sort top→bottom then left→right
        order = sorted(range(len(rec_boxes)), key=lambda k: (rec_boxes[k][1], rec_boxes[k][0]))
        rect_texts_sorted = [rect_texts[i] for i in order]
        rec_boxes_sorted  = [rec_boxes[i]  for i in order]
        rec_polys_sorted  = [rec_polys[i]  for i in order]
        rect_norms_sorted = [self._normalize(t) for t in rect_texts_sorted]

        # filter junk/empties
        keep_mask = [len(n) >= self.min_rect_norm_len for n in rect_norms_sorted]
        rect_texts_f = [t for t,k in zip(rect_texts_sorted, keep_mask) if k]
        rec_boxes_f  = [b for b,k in zip(rec_boxes_sorted,  keep_mask) if k]
        rec_polys_f  = [p for p,k in zip(rec_polys_sorted, keep_mask) if k]
        rect_norms_f = [n for n,k in zip(rect_norms_sorted, keep_mask) if k]

        # map filtered idx -> original Paddle idx
        filtered_to_orig = [order[idx] for idx, k in enumerate(keep_mask) if k]

        return rect_texts_f, rec_boxes_f, rec_polys_f, rect_norms_f, filtered_to_orig

    # ---------- public API ----------
    def map_image_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        In-place: attaches 'paddle_bbox' to each dialogue in item['parsed_dialogue'].
        Returns the same item for convenience.
        """
        (
            rect_texts_f,
            rec_boxes_f,
            rec_polys_f,
            rect_norms_f,
            filtered_to_orig,
        ) = self._sort_and_filter(item)

        dialogs = item.get("parsed_dialogue", [])
        claimed: set[int] = set()
        start_cursor = 0

        for dlg in dialogs:
            dlg_text = dlg.get("text", "")
            dlg_norm = self._normalize(dlg_text)

            matched = False
            matched_start: Optional[int] = None
            matched_end: Optional[int] = None

            i = start_cursor
            while i < len(rect_norms_f):
                if i in claimed:
                    i += 1
                    continue

                concat = ""
                j = i
                while (
                    j < len(rect_norms_f)
                    and j not in claimed
                    and len(concat) < max(len(dlg_norm), self.min_chars * 2)
                ):
                    concat += rect_norms_f[j]

                    if dlg_norm.startswith(concat):
                        coverage = len(concat) / max(1, len(dlg_norm))
                        if coverage >= self.min_coverage or (
                            len(concat) >= self.min_chars and coverage >= 0.35
                        ):
                            matched = True
                            matched_start = i
                            matched_end = j
                            break
                    else:
                        break
                    j += 1

                if matched:
                    # reserve span
                    for k in range(matched_start, matched_end + 1):
                        claimed.add(k)
                    # advance cursor
                    start_cursor = matched_end + 1
                    while start_cursor < len(rect_norms_f) and start_cursor in claimed:
                        start_cursor += 1
                    break

                i += 1

            if matched and matched_start is not None:
                bbox = rec_boxes_f[matched_start]
                poly = rec_polys_f[matched_start]
                orig_start = filtered_to_orig[matched_start]
                orig_end   = filtered_to_orig[matched_end]

                dlg["paddle_bbox"] = {
                    "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                    "poly": poly,
                    "matched_rec_text_index": matched_start,          # filtered/sorted stream
                    "matched_rec_text_index_orig": orig_start         # original Paddle index
                }

                if self.debug:
                    span_preview = " | ".join(
                        rect_texts_f[matched_start:matched_end+1][: self.debug_preview_count]
                    )
                    extra = " ..." if (matched_end - matched_start + 1) > self.debug_preview_count else ""
                    print(
                        f"[OK] dlg#{dlg.get('id')} -> rects {matched_start}-{matched_end} "
                        f"(orig idx {orig_start}..{orig_end}): {span_preview}{extra}"
                    )
            else:
                dlg["paddle_bbox"] = None
                if self.debug:
                    print(f"[MISS] dlg#{dlg.get('id')} -> no match")

        return item

    def map_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.map_image_item(it) for it in items]

    # ---------- drawing helpers ----------
    def _resolve_image_path(self, item: Dict[str, Any], image_root: Path, fallback_dir: Path) -> Path:
        """
        Resolve the original page image location from JSON fields. If not found under
        image_root / image_rel_path_from_root, try a filename search under fallback_dir.
        """
        name = item.get("image_file_name")
        rel = item.get("image_rel_path_from_root", "")
        if name:
            candidate = (image_root / rel / name).resolve()
            if candidate.exists():
                return candidate
        # fallback: search by filename under the folder that holds the JSONs
        for p in fallback_dir.rglob(name or "*"):
            if p.name == name:
                return p
        # last resort: raise (keeps behavior explicit)
        raise FileNotFoundError(f"Could not resolve image path for {name!r}")

    def _draw_item(self, item: Dict[str, Any], image_path: Path, out_path: Path) -> None:
        """
        Draw polygons/rectangles for each dialogue that has 'paddle_bbox'.
        Saves an annotated copy to out_path.
        """
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        outline = (255, 64, 64, 255)
        fill = (255, 64, 64, 60)
        label_bg = (0, 0, 0, 180)
        label_fg = (255, 255, 255, 255)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        def text_size(s: str):
            try:
                return draw.textbbox((0, 0), s, font=font)[2:4]
            except Exception:
                try:
                    return (draw.textlength(s, font=font), 12)
                except Exception:
                    return (7 * len(s), 12)

        for dlg in item.get("parsed_dialogue", []):
            bbox = dlg.get("paddle_bbox")
            if not bbox:
                continue

            # prefer polygon; fallback to axis-aligned rect
            poly = bbox.get("poly")
            if poly and isinstance(poly, list) and len(poly) >= 3:
                pts = [(int(x), int(y)) for x, y in poly]
                draw.polygon(pts, fill=fill, outline=outline)
                tlx = min(x for x, y in pts)
                tly = min(y for x, y in pts)
            else:
                x1, y1, x2, y2 = map(int, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
                draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=3)
                tlx, tly = x1, y1

            # label: "#<dialogue_id>@<orig_idx>"
            did = dlg.get("id")
            orig_idx = bbox.get("matched_rec_text_index_orig", bbox.get("matched_rec_text_index"))
            label = f"#{did}@{orig_idx}"

            tw, th = text_size(label)
            pad = 2
            box = [tlx, tly - th - 2 * pad, tlx + tw + 2 * pad, tly]
            if box[1] < 0:
                box = [tlx, tly, tlx + tw + 2 * pad, tly + th + 2 * pad]
                text_anchor = (box[0] + pad, box[1] + pad)
            else:
                text_anchor = (box[0] + pad, box[1] + pad)
            draw.rectangle(box, fill=label_bg)
            draw.text(text_anchor, label, fill=label_fg, font=font)

        # optional tiny legend
        legend_lines = ["Dialogue bbox sanity check", "Label = #id@origIdx"]
        legend_w = max(text_size(s)[0] for s in legend_lines)
        legend_h = sum(text_size(s)[1] for s in legend_lines) + 6
        legend_box = [6, 6, 12 + int(legend_w), 12 + legend_h]
        draw.rectangle(legend_box, fill=(0, 0, 0, 180))
        y = 10
        for s in legend_lines:
            draw.text((10, y), s, fill=(255, 255, 255, 255), font=font)
            y += text_size(s)[1]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=95)

    def annotate_batch(self, items: List[Dict[str, Any]], image_root: Path, out_dir: Path) -> list[Path]:
        """
        For each item, resolve its image path, draw dialogue bboxes, and save as
        <image_stem>_annotated.<ext> into out_dir. Returns list of saved paths.
        """
        saved = []
        for it in items:
            image_path = self._resolve_image_path(it, image_root=image_root, fallback_dir=out_dir)
            stem = image_path.stem
            ext = image_path.suffix or ".jpg"
            out_path = (out_dir / f"{stem}_annotated{ext}").resolve()
            try:
                self._draw_item(it, image_path, out_path)
                saved.append(out_path)
                if self.debug:
                    print(f"[DRAW] {image_path.name} -> {out_path.name}")
            except Exception as e:
                if self.debug:
                    print(f"[DRAW-ERR] {image_path}: {e}")
        return saved
