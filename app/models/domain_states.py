# domain_states.py
import app.models.domain as d
import app.models.exceptions as ex
from typing import cast

def require_images(run: d.OCRRunResponse) -> list[d.OCRImage]:
    if not run.imageResults:
        raise ex.OCRRunError("OCR run produced no image results")
    return run.imageResults

class InferredOCRImage(d.OCRImage):
    """State marker: inferImageRes is guaranteed at runtime"""
    pass

def require_inferred(img: d.OCRImage) -> InferredOCRImage:
    if img.inferImageRes is None or img.paddleResizeInfo is None:
        raise ex.InferImageError("OCRImage is not inferred yet")
    return cast(InferredOCRImage, img)

class ParsedOCRImage(InferredOCRImage):
    """State marker: parsedDialogueLines is guaranteed at runtime (may be empty)"""
    pass

def require_parsed(img: InferredOCRImage) -> ParsedOCRImage:
    if img.parsedDialogueLines is None:
        raise ex.ParseDialogueError("OCRImage dialgoue lines are not parsed yet")
    return cast(ParsedOCRImage, img)

class DialogueOCRImage(ParsedOCRImage):
    """At least one dialogue line exists"""

def require_dialogue(img: ParsedOCRImage) -> DialogueOCRImage:
    if not img.parsedDialogueLines:
        raise ex.NoDialogueError("OCRImage has no dialogue lines")
    return cast(DialogueOCRImage, img)

class PaddleReadyOCRRun(d.PaddleAugmentedOCRRunResponse):
    pass

def require_paddle_ready_ocrrun(run: d.PaddleAugmentedOCRRunResponse):
    if not run.ocr_json_file:
        raise ex.PaddleAugmentationError("Paddle Augmentation failed, invalid json file path")
    if not run.imageResults or not isinstance(run.imageResults[0], d.PaddleOCRImage):
        raise ex.PaddleAugmentationError("Paddle Augmentation failed, no images found")
    for img in run.imageResults:
        if img.has_text is None:
            raise ex.PaddleAugmentationError(f"Img does not have has_text field img_id: {img.image_id}")
        if not img.has_text:
            continue  # silent image, valid, skip paddle checks
        if not img.parsedDialogueLines:
            raise ex.PaddleAugmentationError("Image marked has_text but has no dialogue lines")
        if not isinstance(img.parsedDialogueLines[0], d.PaddleDialogueLineResponse):
            raise ex.PaddleAugmentationError("Dialogue lines not paddle-augmented")
        for dlg in img.parsedDialogueLines:
            if dlg.status is None:
                raise ex.PaddleAugmentationError(f"status field doesn't exist on img_id: {img.image_id} dlg_id: {dlg.id}")
            if dlg.status != "ok":
                continue
            if not dlg.paddlebbox or not isinstance(dlg.paddlebbox, d.PaddleBBox) or not dlg.paddlebbox.poly or not isinstance(dlg.paddlebbox.matched_rec_text_index, int) or not isinstance(dlg.paddlebbox.matched_rec_text_index_orig, int):
                raise ex.PaddleAugmentationError(f"Paddle Augmentation failed, invalid bbox for img: {img.inferImageRes.image_ref.filename if img.inferImageRes else 'invalid img name'} - dlgId: {dlg.id}")
            
    return cast(PaddleReadyOCRRun, run)


