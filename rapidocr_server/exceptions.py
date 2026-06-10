from fastapi import HTTPException, status


class ImageInputError(HTTPException):
    """Raised when no image input is provided or the input is invalid."""

    def __init__(self, message: str = "Invalid image input"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ImageInputError", "message": message},
        )


class ImageDecodeError(HTTPException):
    """Raised when the image cannot be decoded."""

    def __init__(self, message: str = "Failed to decode image"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "ImageDecodeError", "message": message},
        )


class OCRProcessError(HTTPException):
    """Raised when an error occurs during OCR inference."""

    def __init__(self, message: str = "OCR processing failed", detail: str = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "OCRProcessError", "message": message, "detail": detail},
        )


class InvalidParameterError(HTTPException):
    """Raised when a request parameter fails semantic validation."""

    def __init__(self, message: str = "Invalid parameter", detail: str = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidParameterError", "message": message, "detail": detail},
        )
