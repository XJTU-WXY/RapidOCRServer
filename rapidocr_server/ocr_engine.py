import base64
import io
import logging
import multiprocessing as mp
import threading
import time
from typing import Dict, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from .exceptions import ImageDecodeError, OCRProcessError

logger = logging.getLogger(__name__)


def _worker_process(
    config_path: Optional[str],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    try:
        from rapidocr import RapidOCR
        ocr = RapidOCR(config_path=config_path) if config_path else RapidOCR()
    except Exception as e:
        result_queue.put({"error": f"Model load failed: {e}"})
        return

    result_queue.put({"ready": True})

    while True:
        task = task_queue.get()
        if task is None: 
            break
        try:
            img_array, kwargs = task
            res = ocr(img_array, **kwargs)

            if res.boxes is None or res.txts is None or res.scores is None:
                result_queue.put({"result": {}})
            else:
                result_queue.put({
                    "result": {
                        i: {
                            "rec_txt": txt,
                            "dt_boxes": boxes.tolist(),
                            "score": float(score),
                        }
                        for i, (boxes, txt, score) in enumerate(
                            zip(res.boxes, res.txts, res.scores)
                        )
                    }
                })
        except Exception as e:
            result_queue.put({"error": str(e)})



class OCREngine:
    _mp_ctx = mp.get_context("spawn")

    def __init__(
        self,
        config_path: Optional[str] = None,
        idle_timeout_minutes: float = 0,
    ) -> None:
        self._config_path = config_path
        self._idle_timeout_seconds: float = idle_timeout_minutes * 60
        self._lock = threading.RLock()
        self._process: Optional[mp.Process] = None
        self._task_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._last_request_time: float = 0.0

        self._load_model()

        if self._idle_timeout_seconds > 0:
            logger.info(
                "Idle auto-unload enabled: model will be released after %.1f minute(s) of inactivity",
                idle_timeout_minutes,
            )
            self._start_watchdog()
        else:
            logger.info("Idle auto-unload disabled")

    def _load_model(self) -> None:
        logger.info("Starting OCR engine worker process (config_path=%s) ...", self._config_path)

        task_queue = self._mp_ctx.Queue()
        result_queue = self._mp_ctx.Queue()

        process = self._mp_ctx.Process(
            target=_worker_process,
            args=(self._config_path, task_queue, result_queue),
            daemon=True,
        )
        process.start()

        try:
            signal = result_queue.get(timeout=180)
        except Exception:
            process.terminate()
            process.join()
            raise RuntimeError("OCR engine worker process did not respond within 180 s")

        if "error" in signal:
            process.terminate()
            process.join()
            raise RuntimeError(signal["error"])

        self._process = process
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._last_request_time = time.monotonic()
        logger.info("OCR engine worker process ready (pid=%d)", process.pid)

    def _unload_model(self) -> None:
        if self._process is None:
            return

        pid = self._process.pid
        try:
            self._task_queue.put(None)
            self._process.join(timeout=5)

            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)

            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        except Exception as e:
            logger.warning("Error while terminating OCR engine worker (pid=%d): %s", pid, e)
        finally:
            self._process = None
            self._task_queue = None
            self._result_queue = None
            logger.info(
                "OCR engine worker process (pid=%d) terminated due to inactivity.", pid
            )

    def _ensure_model_loaded(self) -> None:
        if self._process is None or not self._process.is_alive():
            logger.info("OCR engine worker is not running, reloading ...")
            self._load_model()

    def _start_watchdog(self) -> None:
        t = threading.Thread(
            target=self._watchdog_loop,
            name="ocr-idle-watchdog",
            daemon=True,
        )
        t.start()

    def _watchdog_loop(self) -> None:
        poll_interval = min(30.0, self._idle_timeout_seconds / 2)
        while True:
            time.sleep(poll_interval)
            with self._lock:
                if self._process is None:
                    continue
                idle_seconds = time.monotonic() - self._last_request_time
                if idle_seconds >= self._idle_timeout_seconds:
                    self._unload_model()

    def decode_image(self, image_data: str) -> Image.Image:
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
        try:
            return Image.open(file_like)
        except UnidentifiedImageError as e:
            raise ImageDecodeError(f"Unrecognized image format: {e}") from e
        except Exception as e:
            raise ImageDecodeError(f"Failed to read uploaded file: {e}") from e

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
        img_array = np.array(image)

        ocr_kwargs = {
            k: v for k, v in {
                "use_det": use_det,
                "use_cls": use_cls,
                "use_rec": use_rec,
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

            self._task_queue.put((img_array, ocr_kwargs))
            try:
                response = self._result_queue.get(timeout=60)
            except Exception as e:
                raise OCRProcessError(
                    message="OCR engine worker did not respond within 60 s",
                    detail=str(e),
                )

        if "error" in response:
            raise OCRProcessError(
                message="OCR engine inference failed",
                detail=response["error"],
            )

        return response["result"]