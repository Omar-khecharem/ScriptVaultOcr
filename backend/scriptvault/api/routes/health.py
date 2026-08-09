"""État du serveur et du pool de moteurs (``GET /api/health``)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request

from ... import __version__ as package_version
from ... import core_ocr
from ...config import Settings
from ...engines import EngineManager
from ...schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="État du serveur et du pool de moteurs",
)
async def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    engines: EngineManager = request.app.state.engines
    corrector = getattr(request.app.state, "char_corrector", None)
    pool: dict[str, Any] = await engines.health()
    return HealthResponse(
        version=package_version,
        engine_version=core_ocr.__version__,
        preloading=bool(pool["preloading"]),
        engines=pool["engines"],
        cpu_threads=(
            settings.cpu_threads
            if settings.cpu_threads > 0
            else min(8, os.cpu_count() or 4)
        ),
        max_concurrency=settings.effective_max_concurrency,
        lang=settings.lang,
        uptime_s=float(pool["uptime_s"]),
        corrections=(
            {
                "enabled": corrector is not None,
                "engine": "char-level-viterbi",
                "lexicon": False,
            }
            if corrector is not None
            else {"enabled": False, "engine": "none", "lexicon": False}
        ),
    )
