from pydantic import BaseModel, field_validator, model_validator
from pathlib import Path
from typing import Literal, Any, Optional, List
from enum import Enum

class MediaNamespace(str, Enum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"

class MediaRef(BaseModel):
    namespace: MediaNamespace
    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        p = Path(v)

        # must be relative
        if p.is_absolute():
            raise ValueError("MediaRef.path must be a relative path")

        # no traversal
        if ".." in p.parts:
            raise ValueError("MediaRef.path must not contain '..'")

        # normalize to posix
        return p.as_posix()

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix
    
    @property
    def posix_path_from_media_root(self) -> str:
        return str(Path(self.namespace.value) / self.path)
    
    @property
    def posix_path_from_namespace(self) -> str:
        return str(Path(self.path).as_posix)
    
    def namespace_path(self, media_root: Path) -> Path:
        return media_root / self.namespace.value 
    
    def resolve(self, media_root: Path) -> Path:
        return media_root / self.namespace.value / self.path

class InferImageResponse(BaseModel):
    image_ref: MediaRef
    image_text: Any
    image_width: int
    image_height: int
    input_tokens: int
    output_tokens: int
    throughput: float

class DialogueLineResponse(BaseModel):
    id: int
    image_id: str
    speaker: str
    gender: str
    emotion: str
    text: str

class PaddleResizeInfo(BaseModel):
    original_h: int
    original_w: int
    resized_h: int
    resized_w: int
    ratio_h: float
    ratio_w: float

class OCRImage(BaseModel):
    image_id: str
    inferImageRes: Optional[InferImageResponse] = None
    parsedDialogueLines: Optional[list[DialogueLineResponse]] = None
    paddleResizeInfo: Optional[PaddleResizeInfo] = None

    @model_validator(mode="after")
    def attach_paddle_resize_info(self):
        # If already computed, do nothing
        if self.paddleResizeInfo is not None:
            return self

        # Only compute if we have dimensions
        if self.inferImageRes is None:
            return self

        h = self.inferImageRes.image_height
        w = self.inferImageRes.image_width

        self.paddleResizeInfo = paddle_resize_info(h, w)
        return self

class OCRRunResponse(BaseModel):
    run_id: str
    imageResults: Optional[List[OCRImage]] = None
    error: Optional[str] = None

# Augmented by paddleocr (this service)
class PaddleBBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    poly: Optional[list[list[float]]] = None
    matched_rec_text_index: Optional[int] = None
    matched_rec_text_index_orig: Optional[int] = None

class OriginalImageBBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class PaddleDialogueLineResponse(DialogueLineResponse):
    paddlebbox: Optional[PaddleBBox] = None
    original_bbox: Optional[OriginalImageBBox] = None

class PaddleOCRImage(OCRImage):
    parsedDialogueLines: Optional[list[PaddleDialogueLineResponse]] = None
    paddleocr_result: Optional[Any] = None

    @model_validator(mode="after")
    def attach_resize_info_to_lines(self):
        if (
            self.paddleResizeInfo is None
            or not self.parsedDialogueLines
        ):
            return self

        for line in self.parsedDialogueLines:
            if (
                line.paddlebbox is not None
                and line.original_bbox is None
            ):
                line.original_bbox = scale_paddle_bbox_to_original(
                    line.paddlebbox,
                    self.paddleResizeInfo,
                )

        return self

class PaddleAugmentedOCRRunResponse(OCRRunResponse):
    # path to THE json file (with bboxes)
    ocr_json_file: MediaRef
    imageResults: Optional[List[PaddleOCRImage]] = None

# Helpers
def paddle_resize_info(
    h: int,
    w: int,
    *,
    limit_type: str = "min",
    limit_side_len: int = 8000,
    max_side_limit: int = 4000,
    round_to: int = 32,
) -> PaddleResizeInfo:
    # ----- stage 1: limit_type scaling -----
    if limit_type == "min":
        if min(h, w) < limit_side_len:
            ratio = limit_side_len / min(h, w)
        else:
            ratio = 1.0
    elif limit_type == "max":
        if max(h, w) > limit_side_len:
            ratio = limit_side_len / max(h, w)
        else:
            ratio = 1.0
    elif limit_type == "resize_long":
        ratio = limit_side_len / max(h, w)
    else:
        raise ValueError("Unsupported limit_type")

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    # ----- stage 2: max_side_limit -----
    if max(resize_h, resize_w) > max_side_limit:
        ratio2 = max_side_limit / max(resize_h, resize_w)
        resize_h = int(resize_h * ratio2)
        resize_w = int(resize_w * ratio2)

    # ----- stage 3: round to multiple of 32 -----
    resize_h = max(int(round(resize_h / round_to) * round_to), round_to)
    resize_w = max(int(round(resize_w / round_to) * round_to), round_to)

    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return PaddleResizeInfo(
        original_h=h,
        original_w=w,
        resized_h=resize_h,
        resized_w=resize_w,
        ratio_h=ratio_h,
        ratio_w=ratio_w,
    )

def scale_paddle_bbox_to_original(
    bbox: PaddleBBox,
    resize_info: PaddleResizeInfo,
) -> OriginalImageBBox:
    return OriginalImageBBox(
        x1=bbox.x1 / resize_info.ratio_w,
        y1=bbox.y1 / resize_info.ratio_h,
        x2=bbox.x2 / resize_info.ratio_w,
        y2=bbox.y2 / resize_info.ratio_h,
    )

