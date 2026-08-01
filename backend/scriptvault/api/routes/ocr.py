"""Endpoints OCR : fichier unique, lot, et santé du moteur.

Contrat:

* ``POST /api/ocr/single`` — un fichier image **ou PDF** (multipart). Le PDF
  est rastérisé page par page et chaque page est reconnue. Option
  ``preview=true`` : chaque page renvoie aussi l'image *telle qu'analysée*
  (PNG base64) — le client web dessine alors les boîtes sans risque de
  désalignement (le deskew peut changer les dimensions).
* ``POST /api/ocr/batch`` — plusieurs fichiers (multipart). Les fichiers en
  échec sont isolés (``status: "error"``), le lot poursuit son exécution.
* ``GET /api/health`` — état du serveur et du pool de moteurs.

Les échecs sont convertis en réponses HTTP par les handlers globaux
(:mod:`scriptvault.api.app`):

* :class:`OCRInitError` → ``503 Service Unavailable`` (moteur indisponible) ;
* :class:`OCRImageError` / ``PDFRasterError`` → ``400 Bad Request`` ;
* ``TimeoutError`` → ``504 Gateway Timeout``.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from ...config import Settings
from ...core_ocr import ImagePreprocessor, OCRBaseError, OCRImageError
from ...engines import EngineManager
from ...pdf import PDFRasterError, rasterize_pdf_bytes
from ...schemas import (
    OCRBatchItem,
    OCRBatchResponse,
    OCRItem,
    OCRPage,
    OCRResponse,
)

router = APIRouter(prefix="/api/ocr", tags=["ocr"])

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _read_upload(upload: UploadFile, settings: Settings) -> bytes:
    """Valide et lit un fichier uploadé (taille + extension)."""
    filename = upload.filename or "upload"
    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise OCRImageError(
            f"Format non supporté: {extension or '(sans extension)'}. "
            f"Formats acceptés: {', '.join(sorted(settings.allowed_extensions))}."
        )
    data = await upload.read()
    if len(data) == 0:
        raise OCRImageError(f"Fichier vide: {filename!r}")
    if len(data) > settings.max_file_bytes:
        raise OCRImageError(
            f"Fichier trop volumineux ({len(data) / 1e6:.1f} Mo, "
            f"limite {settings.max_file_mb} Mo)."
        )
    return data


def _mean_confidence(items: list[dict[str, Any]]) -> float:
    """Confiance moyenne des lignes (0.0 si aucune ligne)."""
    if not items:
        return 0.0
    scores = [float(item.get("confidence", 0.0)) for item in items]
    return round(sum(scores) / len(scores), 4)


def _joined_text(items: list[dict[str, Any]]) -> str:
    """Texte plein d'une page : lignes séparées par des sauts."""
    return "\n".join(str(item.get("text", "")) for item in items).strip()


def _encode_preview(image: np.ndarray) -> Optional[str]:
    """Encode l'image analysée en PNG base64 (data URL)."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


async def _predict_page(
    engines: EngineManager,
    preprocessor: ImagePreprocessor,
    image: np.ndarray,
    lang: Optional[str],
    use_preprocess: bool,
    page_number: int,
    started: float,
    with_preview: bool,
) -> OCRPage:
    """Prétraite, infère et normalise une page (image ou page de PDF).

    Le prétraitement est appliqué ici (et non dans le moteur) afin de pouvoir
    renvoyer l'image exactement analysée : les boîtes OCR sont ainsi toujours
    alignées avec la prévisualisation.
    """
    if use_preprocess:
        processed = await asyncio.to_thread(
            preprocessor.preprocess, image, binarize=True
        )
    else:
        processed = image
    items = await engines.predict_array(processed, lang=lang, preprocess=False)
    height, width = processed.shape[:2]
    return OCRPage(
        page=page_number,
        width=int(width),
        height=int(height),
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
        text=_joined_text(items),
        confidence=_mean_confidence(items),
        items=[
            OCRItem(
                text=str(item.get("text", "")),
                confidence=float(item.get("confidence", 0.0)),
                box=item.get("box") or [],
            )
            for item in items
        ],
        preview=_encode_preview(processed) if with_preview else None,
    )


async def _process_file(
    engines: EngineManager,
    preprocessor: ImagePreprocessor,
    data: bytes,
    filename: str,
    lang: Optional[str],
    preprocess: Optional[bool],
    with_preview: bool,
    settings: Settings,
) -> OCRResponse:
    """Traite un fichier : images directes, PDF rastérisé page par page."""
    extension = Path(filename).suffix.lower()
    use_preprocess = settings.preprocess if preprocess is None else preprocess
    started = time.perf_counter()

    if extension == ".pdf":
        images = await asyncio.to_thread(rasterize_pdf_bytes, data)
        pages = [
            await _predict_page(
                engines,
                preprocessor,
                image,
                lang,
                use_preprocess,
                index,
                started,
                with_preview,
            )
            for index, image in enumerate(images, start=1)
        ]
    else:
        image = await asyncio.to_thread(preprocessor.read_image_bytes, data)
        pages = [
            await _predict_page(
                engines,
                preprocessor,
                image,
                lang,
                use_preprocess,
                1,
                started,
                with_preview,
            )
        ]

    total_ms = round((time.perf_counter() - started) * 1000.0, 2)
    confidences = [page.confidence for page in pages if page.items]
    return OCRResponse(
        file=filename,
        pages=pages,
        text="\n\n".join(page.text for page in pages if page.text),
        confidence=(
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        ),
        elapsed_ms=total_ms,
    )


def _state(request: Request) -> tuple[Settings, EngineManager, ImagePreprocessor]:
    """Accès rapide à l'état de l'application."""
    return (
        request.app.state.settings,
        request.app.state.engines,
        request.app.state.preprocessor,
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post(
    "/single",
    response_model=OCRResponse,
    summary="Reconnaître le texte d'une image ou d'un PDF",
)
async def ocr_single(
    request: Request,
    file: UploadFile = File(..., description="Image (PNG, JPG, TIFF, WebP) ou PDF"),
    lang: Optional[str] = Form(
        None, description="Langue des modèles (défaut: serveur)"
    ),
    preprocess: Optional[bool] = Form(
        None, description="Pipeline OpenCV (CLAHE/deskew/binarize), défaut: serveur"
    ),
    preview: bool = Form(
        False,
        description="Inclure l'image analysée (PNG base64) pour un overlay aligné",
    ),
) -> OCRResponse:
    """OCR d'un fichier unique : lignes, confiances, boîtes, (pré)visualisation."""
    settings, engines, preprocessor = _state(request)
    data = await _read_upload(file, settings)
    return await _process_file(
        engines,
        preprocessor,
        data,
        file.filename or "upload",
        lang,
        preprocess,
        preview,
        settings,
    )


@router.post(
    "/batch",
    response_model=OCRBatchResponse,
    summary="Reconnaître le texte de plusieurs fichiers en un lot",
)
async def ocr_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    lang: Optional[str] = Form(None),
    preprocess: Optional[bool] = Form(None),
) -> OCRBatchResponse:
    """Lot OCR : chaque fichier est traité dans la limite de concurrence.

    Les échecs sont isolés par fichier : le lot continue jusqu'au bout.
    """
    settings, engines, preprocessor = _state(request)
    started = time.perf_counter()

    validated: list[tuple[UploadFile, bytes, Optional[str]]] = []
    for upload in files:
        try:
            validated.append((upload, await _read_upload(upload, settings), None))
        except OCRImageError as exc:
            validated.append((upload, b"", str(exc)))

    semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))

    async def process(
        upload: UploadFile, data: bytes, error: Optional[str]
    ) -> OCRBatchItem:
        name = upload.filename or "upload"
        if error is not None:
            return OCRBatchItem(file=name, status="error", error=error)
        async with semaphore:
            try:
                result = await _process_file(
                    engines,
                    preprocessor,
                    data,
                    name,
                    lang,
                    preprocess,
                    with_preview=False,
                    settings=settings,
                )
                return OCRBatchItem(file=name, status="ok", result=result)
            except (OCRBaseError, PDFRasterError) as exc:
                return OCRBatchItem(file=name, status="error", error=str(exc))
            except TimeoutError as exc:
                return OCRBatchItem(file=name, status="error", error=str(exc))

    items = await asyncio.gather(
        *(process(upload, data, error) for upload, data, error in validated)
    )
    ok = sum(1 for item in items if item.status == "ok")
    return OCRBatchResponse(
        total=len(items),
        ok=ok,
        errors=len(items) - ok,
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
        items=items,
    )
