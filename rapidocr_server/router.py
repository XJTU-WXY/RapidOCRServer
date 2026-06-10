import logging
from typing import Dict, Optional

from fastapi import APIRouter, Form, Request, UploadFile

from .exceptions import ImageInputError, InvalidParameterError
from .ocr_engine import OCREngine
from .schemas import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def get_engine(request: Request) -> OCREngine:
    return request.app.state.engine


@router.get("/")
def root():
    return {"message": "It works!"}


@router.post(
    "/ocr",
    responses={
        200: {"description": "Recognition successful"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        422: {"model": ErrorResponse, "description": "Image decoding failed"},
        500: {"model": ErrorResponse, "description": "OCR processing error"},
    },
)
async def ocr(
    request: Request,

    image_file: Optional[UploadFile] = None,
    image_data: Optional[str] = Form(None, description="Base64-encoded image data"),

    use_det: Optional[bool] = Form(None, description="Whether to run text detection"),
    use_cls: Optional[bool] = Form(None, description="Whether to run orientation classification"),
    use_rec: Optional[bool] = Form(None, description="Whether to run text recognition"),

    return_word_box: Optional[bool] = Form(None, description="Whether to return word-level bounding boxes"),
    return_single_char_box: Optional[bool] = Form(None, description="Whether to return character-level bounding boxes"),
    text_score: Optional[float] = Form(None, ge=0.0, le=1.0, description="Confidence threshold for text recognition"),
    box_thresh: Optional[float] = Form(None, ge=0.0, le=1.0, description="Confidence threshold for detection boxes"),
    unclip_ratio: Optional[float] = Form(None, gt=0.0, description="Expansion ratio for detection boxes"),
) -> Dict:
    engine: OCREngine = get_engine(request)

    if text_score is not None and not (0.0 <= text_score <= 1.0):
        raise InvalidParameterError("text_score must be in the range [0.0, 1.0]")
    if box_thresh is not None and not (0.0 <= box_thresh <= 1.0):
        raise InvalidParameterError("box_thresh must be in the range [0.0, 1.0]")
    if unclip_ratio is not None and unclip_ratio <= 0.0:
        raise InvalidParameterError("unclip_ratio must be greater than 0")

    if image_file is not None:
        img = engine.open_image(image_file.file)
    elif image_data is not None:
        img = engine.decode_image(image_data)
    else:
        raise ImageInputError("Provide an image via image_file or image_data")

    result = engine.run(
        image=img,
        use_det=use_det,
        use_cls=use_cls,
        use_rec=use_rec,
        return_word_box=return_word_box,
        return_single_char_box=return_single_char_box,
        text_score=text_score,
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
    )

    return result
