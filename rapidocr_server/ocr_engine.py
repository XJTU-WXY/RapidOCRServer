import base64
import io
import logging
import threading
import time
from typing import Dict, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from rapidocr import RapidOCR

from .exceptions import ImageDecodeError, OCRProcessError

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(
        self,
        config_path: Optional[str] = None,
        idle_timeout_minutes: float = 0,
    ) -> None:
        self._config_path = config_path
        self._idle_timeout_seconds: float = idle_timeout_minutes * 60
        self._lock = threading.RLock()
        self._ocr: Optional[RapidOCR] = None
        self._last_request_time: float = 0.0
        self._watchdog_thread: Optional[threading.Thread] = None

        self._load_model()

        if self._idle_timeout_seconds > 0:
            logger.info(
                "Idle auto-unload enabled: model will be released after %.1f minute(s) of inactivity",
                idle_timeout_minutes,
            )
            self._start_watchdog()
        else:
            logger.info("Idle auto-unload disabled")

    # ------------------------------------------------------------------
    # Internal model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        logger.info("Loading RapidOCR model (config_path=%s) ...", self._config_path)
        try:
            if self._config_path:
                self._ocr = RapidOCR(config_path=self._config_path)
            else:
                self._ocr = RapidOCR()
            self._last_request_time = time.monotonic()
            logger.info("RapidOCR model loaded successfully")
        except Exception as e:
            logger.exception("Failed to load RapidOCR model")
            raise RuntimeError(f"Failed to load RapidOCR model: {e}") from e

    def _unload_model(self) -> None:
        if self._ocr is not None:
            self._ocr = None
            logger.info("RapidOCR model unloaded due to idle timeout")

    def _ensure_model_loaded(self) -> None:
        if self._ocr is None:
            logger.info("Model is not loaded, reloading ...")
            self._load_model()

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def _start_watchdog(self) -> None:
        """Start the background idle-watchdog thread (daemon so it won't block shutdown)."""
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="ocr-idle-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Poll every 30 s (or half the timeout, whichever is smaller) and unload when idle."""
        poll_interval = min(30.0, self._idle_timeout_seconds / 2)
        while True:
            time.sleep(poll_interval)
            with self._lock:
                if self._ocr is None:
                    # Already unloaded; nothing to do until next request reloads it.
                    continue
                idle_seconds = time.monotonic() - self._last_request_time
                if idle_seconds >= self._idle_timeout_seconds:
                    self._unload_model()

    # ------------------------------------------------------------------
    # Public image helpers
    # ------------------------------------------------------------------

    def decode_image(self, image_data: str) -> Image.Image:
        """Decode a Base64 string into a PIL.Image."""
        try:
            img_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(img_bytes))
        except (base64.binascii.Error, ValueError) as e:
            raise ImageDecodeError(f"Base64 decoding failed: {e}") from e
        except UnidentifiedImageError as e:
            raise ImageDecodeError(f"Unrecognized image format: {e}") from e
        except Exception as e:
            raise ImageDecodeError(f"Unexpected error while decoding image: {e}") from e

    def open_image(self, file_like) -> Image.Image:
        """Open a PIL.Image from a file-like object."""
        try:
            return Image.open(file_like)
        except UnidentifiedImageError as e:
            raise ImageDecodeError(f"Unrecognized image format: {e}") from e
        except Exception as e:
            raise ImageDecodeError(f"Failed to read uploaded file: {e}") from e

    # ------------------------------------------------------------------
    # Public OCR entry point
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        return_word_box: Optional[bool] = None,
        return_single_char_box: Optional[bool] = None,
        text_score: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
    ) -> Dict:
        """Run OCR and return a structured result dict."""
        img_array = np.array(image)

        optional_kwargs = {
            k: v for k, v in {
                "return_word_box": return_word_box,
                "return_single_char_box": return_single_char_box,
                "text_score": text_score,
                "box_thresh": box_thresh,
                "unclip_ratio": unclip_ratio,
            }.items() if v is not None
        }

        with self._lock:
            self._ensure_model_loaded()
            self._last_request_time = time.monotonic()

            try:
                ocr_res = self._ocr(
                    img_array,
                    use_det=use_det,
                    use_cls=use_cls,
                    use_rec=use_rec,
                    **optional_kwargs,
                )
            except Exception as e:
                logger.exception("Error during OCR inference")
                raise OCRProcessError(message="OCR inference failed", detail=str(e)) from e

        if ocr_res.boxes is None or ocr_res.txts is None or ocr_res.scores is None:
            return {}

        return {
            i: {
                "rec_txt": txt,
                "dt_boxes": boxes.tolist(),
                "score": float(score),
            }
            for i, (boxes, txt, score) in enumerate(
                zip(ocr_res.boxes, ocr_res.txts, ocr_res.scores)
            )
        }