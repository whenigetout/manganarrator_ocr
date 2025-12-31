# domain_states.py
import domain as d
from typing import cast

class InferredOCRImage(d.OCRImage):
    """State marker: inferImageRes is guaranteed at runtime"""
    pass

def require_inferred(img: d.OCRImage) -> InferredOCRImage:
    if img.inferImageRes is None:
        raise d.InferImageError("OCRImage is not inferred yet")
    return cast(InferredOCRImage, img)

class ParsedOCRImage(InferredOCRImage):
    """State marker: parsedDialogueLines is guaranteed at runtime"""
    pass

def require_parsed(img: InferredOCRImage) -> ParsedOCRImage:
    if img.parsedDialogueLines is None:
        raise d.ParseDialogueError("OCRImage is not parsed yet")
    return cast(ParsedOCRImage, img)

class PaddleReadyOCRImage(ParsedOCRImage):
    """State marker: PaddleDialogueLineResponse is guaranteed at runtime"""
    pass

def require_paddle(img: ParsedOCRImage) -> PaddleReadyOCRImage:
    lines = img.parsedDialogueLines
    if not lines or not isinstance(lines[0], d.PaddleDialogueLineResponse):
        raise d.ProcessImageError("OCRImage is not paddle-augmented yet")
    return cast(PaddleReadyOCRImage, img)

def require_images(run: d.OCRRunResponse) -> list[d.OCRImage]:
    if run.imageResults is None:
        raise d.OCRRunError("OCR run produced no image results")
    return run.imageResults
