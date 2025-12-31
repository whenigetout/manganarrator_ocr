# domain_states.py
import domain as d
from typing import cast

def require_images(run: d.OCRRunResponse) -> list[d.OCRImage]:
    if not run.imageResults:
        raise d.OCRRunError("OCR run produced no image results")
    return run.imageResults

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
    if not img.parsedDialogueLines or not isinstance(img.parsedDialogueLines[0], d.DialogueLineResponse):
        raise d.ParseDialogueError("OCRImage dialgoue lines are not parsed yet")
    return cast(ParsedOCRImage, img)

class PaddleReadyOCRRun(d.PaddleAugmentedOCRRunResponse):
    pass

def require_paddle_ready_ocrrun(run: d.PaddleAugmentedOCRRunResponse):
    if not run.imageResults or not isinstance(run.imageResults[0], d.PaddleOCRImage):
        raise d.PaddleAugmentationError("Paddle Augmentation failed, no images found")
    for img in run.imageResults:
        if not img.parsedDialogueLines or not isinstance(img.parsedDialogueLines[0], d.PaddleDialogueLineResponse):
            raise d.PaddleAugmentationError("Paddle Augmentation failed, no dialogue lines found")
        for dlg in img.parsedDialogueLines:
            if not dlg.paddlebbox or not isinstance(dlg.paddlebbox, d.PaddleBBox) or not dlg.paddlebbox.poly or not dlg.paddlebbox.matched_rec_text_index or not dlg.paddlebbox.matched_rec_text_index_orig:
                raise d.PaddleAugmentationError("Paddle Augmentation failed, invalid bbox")
            
    return cast(PaddleReadyOCRRun, run)


