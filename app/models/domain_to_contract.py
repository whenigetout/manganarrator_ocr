from typing import Optional, List
from pathlib import Path

# -------- domain imports --------
from app.models.domain import (
    PaddleAugmentedOCRRunResponse,
    PaddleOCRImage,
    PaddleDialogueLineResponse,
    MediaRef as DomainMediaRef,
    MediaNamespace as DomainMediaNamespace,
    OriginalImageBBox as DomainBBox,
)

# -------- contract imports --------
from mn_contracts.ocr import (
    OCRRun,
    OCRImage,
    DialogueLine,
    ImageInfo,
    MediaRef as ContractMediaRef,
    MediaNamespace as ContractMediaNamespace,
    OriginalImageBBox as ContractBBox,
)

# ---------------------------------------------------------------------
# leaf converters (THE important part)
# ---------------------------------------------------------------------

def to_contract_media_ref(
    src: DomainMediaRef,
) -> ContractMediaRef:
    return ContractMediaRef(
        namespace=ContractMediaNamespace(src.namespace.value),
        path=src.path,
    )

def to_contract_bbox(
    src: DomainBBox | None,
) -> ContractBBox | None:
    if src is None:
        return None
    return ContractBBox(
        x1=src.x1,
        y1=src.y1,
        x2=src.x2,
        y2=src.y2,
    )


# ---------------------------------------------------------------------
# main adapter
# ---------------------------------------------------------------------

def paddle_augmented_run_to_ocr_run(
    src: PaddleAugmentedOCRRunResponse,
    ocr_json_path: Path,
    namespace_path: Path
) -> OCRRun:
    images: Optional[List[OCRImage]] = None

    if src.imageResults:
        images = [_convert_image(img) for img in src.imageResults]

    return OCRRun(
        run_id=src.run_id,
        error=src.error,
        ocr_json_file=to_contract_media_ref(DomainMediaRef(
            namespace=DomainMediaNamespace.OUTPUTS,
            path=str(ocr_json_path.relative_to(namespace_path))
        )),
        images=images,
    )


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _convert_image(img: PaddleOCRImage) -> OCRImage:
    image_info: ImageInfo | None = None

    if img.inferImageRes is not None:
        image_info = ImageInfo(
            image_ref=to_contract_media_ref(
                img.inferImageRes.image_ref
            ),
            image_width=img.inferImageRes.image_width,
            image_height=img.inferImageRes.image_height,
        )

    dialogue_lines: list[DialogueLine] | None = None

    if img.parsedDialogueLines:
        dialogue_lines = [
            _convert_dialogue_line(line)
            for line in img.parsedDialogueLines
        ]

    return OCRImage(
        image_id=img.image_id,
        image_info=image_info,
        dialogue_lines=dialogue_lines,
    )


def _convert_dialogue_line(
    line: PaddleDialogueLineResponse,
) -> DialogueLine:
    return DialogueLine(
        id=line.id,
        image_id=line.image_id,
        speaker=line.speaker,
        gender=line.gender,
        emotion=line.emotion,
        text=line.text,
        original_bbox=to_contract_bbox(line.original_bbox),
    )
