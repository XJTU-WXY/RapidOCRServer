import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser

from .ocr_engine import OCREngine
from .router import router

MultiPartParser.max_part_size = 10 * 1024 * 1024   # 10 MB
MultiPartParser.max_file_size = 20 * 1024 * 1024   # 20 MB

sys.path.append(str(Path(__file__).resolve().parent.parent))

_ENV_CONFIG_PATH = "RAPIDOCR_CONFIG_PATH"
_ENV_IDLE_TIMEOUT = "RAPIDOCR_IDLE_TIMEOUT"


def _setup_logging() -> None:
    fmt = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        stream=sys.stdout,
        force=True,
    )


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    _setup_logging()

    config_path: str | None = os.environ.get(_ENV_CONFIG_PATH) or None

    idle_timeout_minutes: float = 0.0
    raw_timeout = os.environ.get(_ENV_IDLE_TIMEOUT, "0")
    try:
        idle_timeout_minutes = max(0.0, float(raw_timeout))
    except ValueError:
        logger.warning(
            "Invalid value for %s: %r — idle auto-unload disabled",
            _ENV_IDLE_TIMEOUT,
            raw_timeout,
        )

    app = FastAPI(
        title="RapidOCR Server",
        description="OCR recognition service powered by RapidOCR",
        version="0.1.0",
    )

    @app.on_event("startup")
    def startup_event():
        try:
            app.state.engine = OCREngine(
                config_path=config_path,
                idle_timeout_minutes=idle_timeout_minutes,
            )
            logger.info("OCR engine is ready (config_path=%s)", config_path)
        except RuntimeError as e:
            logger.critical("OCR engine initialization failed; cannot start service: %s", e)
            sys.exit(1)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            content = detail
        else:
            content = {"error": "HTTPException", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        field = " -> ".join(str(loc) for loc in first.get("loc", []))
        msg = first.get("msg", "validation error")
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "message": f"Field [{field}] {msg}",
                "detail": errors,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected server error occurred. Please try again later.",
                "detail": str(exc),
            },
        )

    app.include_router(router)

    return app


def main():
    parser = argparse.ArgumentParser("rapidocr_server")
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to RapidOCR config file")
    parser.add_argument("-l", "--listen", type=str, default="0.0.0.0",
                        help="Bind IP address")
    parser.add_argument("-p", "--port", type=int, default=9003,
                        help="Bind port")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Number of worker processes")
    parser.add_argument("-it", "--idle-timeout", type=float, default=None, help="Unload the OCR model after this many minutes of inactivity")
    
    args = parser.parse_args()

    if args.config:
        os.environ[_ENV_CONFIG_PATH] = args.config

    if args.idle_timeout is not None:
        os.environ[_ENV_IDLE_TIMEOUT] = str(args.idle_timeout)

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s %(levelname)s %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s %(message)s"

    uvicorn.run(
        "rapidocr_server.main:create_app",
        host=args.listen,
        port=args.port,
        reload=False,
        workers=args.workers,
        log_config=log_config,
        factory=True,
    )


if __name__ == "__main__":
    main()