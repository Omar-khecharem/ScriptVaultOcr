"""Schémas Pydantic de l'API REST (contrat serveur ↔ web)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OCRItem(BaseModel):
    """Une ligne de texte détectée, avec sa position et sa confiance."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: list[list[int]] = Field(default_factory=list)


class OCRPage(BaseModel):
    """Résultat OCR d'une page (une page = une image pour les PDF)."""

    page: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    elapsed_ms: float = Field(ge=0.0)
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    items: list[OCRItem] = Field(default_factory=list)
    preview: Optional[str] = Field(
        default=None,
        description=(
            "PNG base64 (data URL) de l'image telle qu'analysée — boîtes "
            "parfaitement alignées. Uniquement si ``preview=true``."
        ),
    )


class OCRResponse(BaseModel):
    """Réponse complète pour un fichier traité."""

    file: str
    status: Literal["ok"] = "ok"
    pages: list[OCRPage] = Field(default_factory=list)
    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    elapsed_ms: float = Field(default=0.0, ge=0.0)


class OCRBatchItem(BaseModel):
    """Résultat d'un fichier au sein d'un lot (échec isolé, lot poursuivi)."""

    file: str
    status: Literal["ok", "error"]
    error: Optional[str] = None
    result: Optional[OCRResponse] = None


class OCRBatchResponse(BaseModel):
    """Réponse du traitement par lots."""

    total: int = Field(ge=0)
    ok: int = Field(ge=0)
    errors: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0.0)
    items: list[OCRBatchItem] = Field(default_factory=list)


class ExportRequest(BaseModel):
    """Demande d'export d'un texte corrigé."""

    format: Literal["txt", "docx", "pdf"]
    text: str = Field(default="", max_length=2_000_000)
    filename: Optional[str] = Field(default=None, max_length=255)


class HealthResponse(BaseModel):
    """État du serveur et du pool de moteurs."""

    status: Literal["ok"] = "ok"
    name: str = "ScriptVault OCR API"
    version: str
    engine_version: str
    preloading: bool = False
    engines: dict[str, dict[str, object]] = Field(default_factory=dict)
    cpu_threads: int
    max_concurrency: int
    lang: str
    uptime_s: float
