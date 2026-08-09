"""Fabrique d'application FastAPI — configuration, middleware, handlers.

Le serveur est construit par :func:`create_app`, ce qui le rend testable
(isolation par app, injection d'une fabrique de moteur factice) et utilisable
à la fois par ``python main.py`` et par ``uvicorn scriptvault.api.app:create_app
--factory``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..batch_engine import BatchManager
from ..char_corrector import load_default_corrector
from ..config import Settings
from ..core_ocr import ImagePreprocessor, OCRBaseError, OCRInitError
from ..engines import EngineFactory, EngineManager
from ..exporter import ExportError
from ..pdf import PDFRasterError
from .routes import batches as batches_router
from .routes import export as export_router
from .routes import form as form_router
from .routes import health as health_router
from .routes import ocr as ocr_router

logger = logging.getLogger("scriptvault.api")


def _error_handler(status_code: int):
    """Construit un handler FastAPI renvoyant un JSON structuré."""

    def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "error",
                "detail": str(exc),
                "type": type(exc).__name__,
            },
        )

    return handler


def create_app(
    settings: Optional[Settings] = None,
    engine_factory: Optional[EngineFactory] = None,
) -> FastAPI:
    """Construit l'application API.

    Args:
        settings: Configuration (``None`` = variables d'environnement).
        engine_factory: Fabrique de moteur OCR injectable (tests).
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.preprocessor = ImagePreprocessor()
        app.state.engines = EngineManager(settings, engine_factory=engine_factory)
        await app.state.engines.start()
        app.state.batch = BatchManager(settings, preprocessor=app.state.preprocessor)
        await app.state.batch.init(app.state.engines)
        # Correcteur de niveau caractère (char-level) : le modèle lourd
        # (~50–80 Mo) déposé dans ``models/char_lm/`` est chargé ici, une
        # seule fois, pour toute la durée de vie du processus.
        app.state.char_corrector = load_default_corrector(settings.model_dir)
        logger.info(
            "API démarrée — langue=%r, threads=%d, concurrency=%d, "
            "correcteur=char-level(embarqué).",
            settings.lang,
            settings.cpu_threads or 0,
            settings.effective_max_concurrency,
        )
        yield
        await app.state.batch.close()
        await app.state.engines.close()

    app = FastAPI(
        title="ScriptVault OCR API",
        version=__version__,
        description=(
            "API locale de reconnaissance de texte (HTR TrOCR ONNX, CPU). "
            "Aucune donnée ne quitte la machine."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health_router.router)
    app.include_router(ocr_router.router)
    app.include_router(batches_router.router)
    app.include_router(form_router.router)
    app.include_router(export_router.router)

    # --- Handlers d'erreurs globaux -------------------------------------
    app.add_exception_handler(OCRInitError, _error_handler(503))
    app.add_exception_handler(OCRBaseError, _error_handler(400))
    app.add_exception_handler(PDFRasterError, _error_handler(400))
    app.add_exception_handler(ExportError, _error_handler(400))
    app.add_exception_handler(TimeoutError, _error_handler(504))

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "name": "ScriptVault OCR API",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app
