"""Endpoints de traitement par lots (workflow entreprise).

Contrat:

* ``POST /api/batches`` — dépose un lot de fichiers (multipart). Les fichiers
  sont écrits en zone de travail puis traités en tâche de fond : l'API répond
  immédiatement avec le résumé du lot (progression consultable).
* ``GET /api/batches`` — historique des lots (résumés légers).
* ``GET /api/batches/{id}`` — progression, compteurs, confiance moyenne.
* ``GET /api/batches/{id}/files`` — liste paginable et filtrable (recherche).
* ``GET /api/batches/{id}/files/{file_id}`` — détail complet d'un fichier
  (pages OCR + formulaire structuré).
* ``GET /api/batches/{id}/files/{file_id}/preview?page=N`` — aperçu PNG de la
  page *telle qu'analysée* (overlay aligné côté web).
* ``POST /api/batches/{id}/cancel`` — annule proprement le lot.
* ``DELETE /api/batches/{id}`` — supprime le lot (mémoire + zone de travail).
* ``PATCH /api/batches/{id}/files/{file_id}/form`` — corrige manuellement les
  valeurs d'un formulaire (les corrections sont reprises dans l'export Excel).
* ``GET /api/batches/{id}/export.xlsx`` — export Excel de toutes les données.

Le gestionnaire :class:`~scriptvault.batch_engine.BatchManager` est attaché
à ``app.state.batch`` par la fabrique d'application (voir
:mod:`scriptvault.api.app`).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response

from ...batch_engine import BatchManager
from ...config import Settings
from ...core_ocr import ImagePreprocessor, OCRImageError
from ...engines import EngineManager
from ...excel_exporter import (
    ExcelDocument,
    ExcelError,
    ExcelField,
    ExcelPage,
    export_excel,
)
from ...schemas import FormOverrideRequest
from .ocr import _read_upload

router = APIRouter(prefix="/api/batches", tags=["batches"])

_STATUS_LABELS = {
    "pending": "En attente",
    "processing": "En traitement",
    "done": "Terminé",
    "error": "Erreur",
    "cancelled": "Annulé",
}


def _state(
    request: Request,
) -> tuple[Settings, EngineManager, ImagePreprocessor, BatchManager]:
    """Accès à l'état de l'application."""
    return (
        request.app.state.settings,
        request.app.state.engines,
        request.app.state.preprocessor,
        request.app.state.batch,
    )


def _filter_files(files: list[Any], q: str) -> list[Any]:
    """Filtre par nom de fichier ou libellé de statut (insensible à la casse)."""
    needle = q.strip().lower()
    if not needle:
        return files
    return [
        f
        for f in files
        if needle in f.file_name.lower()
        or needle in _STATUS_LABELS.get(f.status, f.status).lower()
    ]


def _form_to_excel_fields(
    form: Any, overrides: Optional[dict[str, str]] = None
) -> list[ExcelField]:
    """Extrait les zones clés/valeurs du formulaire post-analyse.

    ``overrides`` — corrections manuelles de l'utilisateur (UI) : elles
    remplacent les valeurs lues par l'OCR avant l'export Excel.
    """
    if form is None or not getattr(form, "fields", None):
        return []
    output: list[ExcelField] = []
    for field in form.fields:
        text = (overrides or {}).get(field.key, "") or (field.value or "").strip()
        if not text:
            continue
        output.append(
            ExcelField(
                text=text,
                confidence=float(field.confidence or 0.0),
                label=str(field.key),
            )
        )
    return output


def _build_excel(job: Any) -> bytes:
    """Construit le classeur ``.xlsx`` complet depuis les résultats en mémoire."""
    documents: list[ExcelDocument] = []
    for file in job.files:
        if file.status != "done":
            continue
        pages = []
        for page in file.pages:
            page_no = int(page.get("page", 0))
            pages.append(
                ExcelPage(
                    page=page_no,
                    text=str(page.get("text", "")),
                    confidence=float(file.confidence or 0.0),
                    fields=_form_to_excel_fields(
                        page.get("form"), file.form_overrides.get(page_no)
                    ),
                )
            )
        documents.append(ExcelDocument(filename=file.file_name, pages=pages))
    if not documents:
        raise ExcelError("Aucun fichier traité à exporter dans ce lot.")
    return export_excel(documents)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post(
    "",
    status_code=201,
    summary="Déposer un lot de fichiers (traitement en arrière-plan)",
)
async def create_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    name: str = Form("", description="Nom du lot (optionnel)"),
    lang: Optional[str] = Form(None, description="Langue des modèles OCR"),
    preprocess: Optional[bool] = Form(None, description="Pipeline OpenCV"),
) -> dict[str, Any]:
    """Dépose un lot : réponse immédiate, traitement asynchrone."""
    settings, _, _, batch = _state(request)
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")
    if len(files) > 10_000:
        raise HTTPException(
            status_code=400, detail="Lot trop volumineux (max 10 000 fichiers)."
        )

    uploaded: list[tuple[str, bytes]] = []
    failures: list[dict[str, Any]] = []
    for upload in files:
        filename = upload.filename or "upload"
        try:
            data = await _read_upload(upload, settings)
            uploaded.append((filename, data))
        except OCRImageError as exc:
            failures.append({"name": filename, "error": str(exc)})
    if not uploaded:
        raise HTTPException(status_code=400, detail="Aucun fichier valide.")

    job = batch.create_job(
        uploaded,
        name=name,
        lang=lang or settings.lang,
        preprocess=settings.preprocess if preprocess is None else preprocess,
        storage_root=Path(settings.storage_root),
    )
    payload: dict[str, Any] = {"job": job.summary()}
    if failures:
        payload["rejected"] = failures
    return payload


@router.get("", summary="Historique des lots (résumés légers)")
async def list_batches(request: Request) -> dict[str, Any]:
    _, _, _, batch = _state(request)
    return {"jobs": batch.list_jobs()}


@router.get("/{job_id}", summary="Progression et statistiques d'un lot")
async def get_batch(request: Request, job_id: str) -> dict[str, Any]:
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    return job.summary()


@router.get("/{job_id}/files", summary="Synthèses des fichiers (paginable)")
async def list_batch_files(
    request: Request,
    job_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    q: str = Query("", max_length=128, description="Filtre nom/statut"),
) -> dict[str, Any]:
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    filtered = _filter_files(job.files, q)
    total = len(filtered)
    start = (page - 1) * page_size
    slice_items = filtered[start : start + page_size]
    return {
        "job": job.summary(),
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [f.to_summary() for f in slice_items],
    }


@router.get("/{job_id}/files/{file_id}", summary="Détail complet d'un fichier")
async def get_file_detail(
    request: Request, job_id: str, file_id: str
) -> dict[str, Any]:
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    file = next((f for f in job.files if f.file_id == file_id), None)
    if file is None:
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    summary = file.to_summary()
    summary["pages"] = jsonable_encoder(file.pages)
    summary["overrides"] = {
        str(page): values for page, values in file.form_overrides.items()
    }
    return summary


@router.patch(
    "/{job_id}/files/{file_id}/form",
    summary="Corriger manuellement le formulaire d'une page",
)
async def update_form_overrides(
    request: Request, job_id: str, file_id: str, payload: FormOverrideRequest
) -> dict[str, Any]:
    """Enregistre les valeurs corrigées par l'utilisateur (export Excel inclus)."""
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    file = next((f for f in job.files if f.file_id == file_id), None)
    if file is None:
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    if payload.page < 1 or payload.page > len(file.pages):
        raise HTTPException(status_code=404, detail="Page introuvable.")

    cleaned = {
        key: (value or "").strip() for key, value in payload.values.items() if key
    }
    overrides = file.form_overrides.setdefault(payload.page, {})
    for key, value in cleaned.items():
        if value:
            overrides[key] = value
        else:
            overrides.pop(key, None)
    if not file.form_overrides.get(payload.page):
        file.form_overrides.pop(payload.page, None)
    return {
        "page": payload.page,
        "overrides": dict(file.form_overrides.get(payload.page, {})),
    }


@router.get(
    "/{job_id}/files/{file_id}/preview",
    summary="Aperçu PNG d'une page (telle qu'analysée)",
)
async def file_preview(
    request: Request,
    job_id: str,
    file_id: str,
    page: int = Query(1, ge=1),
) -> dict[str, Any]:
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    file = next((f for f in job.files if f.file_id == file_id), None)
    if file is None:
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    if page < 1 or page > len(file.pages):
        raise HTTPException(status_code=404, detail="Page introuvable.")
    data = await file.get_preview(
        page - 1, batch._preprocessor, binarize=job.preprocess
    )
    if data is None:
        raise HTTPException(status_code=404, detail="Aperçu indisponible.")
    return {"page": int(file.pages[page - 1]["page"]), "preview": data}


@router.post("/{job_id}/cancel", summary="Annuler un lot en cours")
async def cancel_batch(request: Request, job_id: str) -> dict[str, Any]:
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    batch.cancel_job(job)
    return job.summary()


@router.delete("/{job_id}", summary="Supprimer un lot (mémoire + disque)")
async def delete_batch(request: Request, job_id: str) -> dict[str, Any]:
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    batch.remove_job(job_id)
    return {"deleted": job_id}


@router.get("/{job_id}/export.xlsx", summary="Export Excel des données du lot")
async def export_batch(request: Request, job_id: str) -> Response:
    _, _, _, batch = _state(request)
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Lot introuvable.")
    try:
        data = await asyncio.to_thread(_build_excel, job)
    except ExcelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": (
                f'attachment; filename="scriptvault-lot-{job.job_id[:8]}.xlsx"'
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )
