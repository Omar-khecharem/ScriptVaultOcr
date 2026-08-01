"""Endpoint d'export des textes corrigés (TXT / DOCX / PDF).

Le client web télécharge le document généré côté serveur : la partie web
reste légère (aucune dépendance bureautique en JavaScript) et l'implémentation
est partagée avec le desktop (:mod:`scriptvault.exporter`).
"""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, Response

from ...exporter import ExportError, export_text
from ...schemas import ExportRequest

router = APIRouter(prefix="/api", tags=["export"])

_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "pdf": "application/pdf",
}

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]")


def _build_filename(requested: str | None, format_name: str) -> str:
    """Normalise le nom de fichier demandé (aucun chemin, aucun séparateur)."""
    if requested:
        stem = _SAFE_FILENAME.sub("_", requested).strip()
        if stem:
            return (
                stem
                if stem.lower().endswith(f".{format_name}")
                else f"{stem}.{format_name}"
            )
    return f"scriptvault-export.{format_name}"


@router.post(
    "/export",
    summary="Exporter un texte corrigé en TXT, DOCX ou PDF",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        400: {"description": "Texte vide ou format inconnu."},
    },
)
async def export_document(payload: ExportRequest) -> Response:
    """Sérialise ``payload.text`` dans le format demandé (pièce jointe)."""
    if not payload.text.strip():
        raise ExportError("Le texte à exporter est vide.")
    data = await asyncio.to_thread(export_text, payload.text, payload.format)
    return Response(
        content=data,
        media_type=_MEDIA_TYPES.get(payload.format, "application/octet-stream"),
        headers={
            "Content-Disposition": (
                'attachment; filename="'
                f'{_build_filename(payload.filename, payload.format)}"'
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )
