from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64-encoded image data")
    use_det: Optional[bool] = Field(None, description="Whether to run text detection")
    use_cls: Optional[bool] = Field(None, description="Whether to run orientation classification")
    use_rec: Optional[bool] = Field(None, description="Whether to run text recognition")
    return_word_box: Optional[bool] = Field(None, description="Whether to return word-level bounding boxes")
    return_single_char_box: Optional[bool] = Field(None, description="Whether to return character-level bounding boxes")
    text_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence threshold for text recognition")
    box_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence threshold for detection boxes")
    unclip_ratio: Optional[float] = Field(None, gt=0.0, description="Expansion ratio for detection boxes")


class OCRResultItem(BaseModel):
    rec_txt: str = Field(..., description="Recognized text")
    dt_boxes: List[List[float]] = Field(..., description="Bounding box coordinates")
    score: float = Field(..., description="Recognition confidence score")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[Any] = Field(None, description="Additional error detail")
