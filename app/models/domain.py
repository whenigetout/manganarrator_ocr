from pydantic import BaseModel
from pathlib import Path
from typing import Literal, Any, Optional, List

class MediaRef(BaseModel):
    namespace: Literal["inputs", "outputs"]
    path: str

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix

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

class OCRImage(BaseModel):
    image_id: str
    inferImageRes: Optional[InferImageResponse] = None
    parsedDialogueLines: Optional[list[DialogueLineResponse]] = None

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

class PaddleDialogueLineResponse(DialogueLineResponse):
    paddlebbox: Optional[PaddleBBox] = None

class PaddleOCRImage(OCRImage):
    parsedDialogueLines: Optional[list[PaddleDialogueLineResponse]] = None
    paddleocr_result: Optional[Any] = None

class PaddleAugmentedOCRRunResponse(OCRRunResponse):
    imageResults: Optional[List[PaddleOCRImage]] = None
    

# Exceptions
class InferImageError(Exception):
    pass

class ProcessImageError(InferImageError):
    pass

class OCRRunError(ProcessImageError):
    pass

class ParseDialogueError(Exception):
    pass

class PaddleAugmentationError(Exception):
    pass

class SaveJSONError(Exception):
    pass