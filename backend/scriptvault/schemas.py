"""Schémas Pydantic de l'API REST (contrat serveur ↔ web)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class OCRItem(BaseModel):
    """Une ligne de texte détectée, avec sa position et sa confiance."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: list[list[int]] = Field(default_factory=list)
    label: Optional[str] = Field(
        default=None,
        description=(
            "Clé du champ du formulaire quand l'item provient de la lecture "
            "par zones (ex. ``nom``, ``cin``) — sinon ``None``."
        ),
    )


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


class FormSection(str, Enum):
    """Bloc du formulaire-type (gabarit feuille d'examen)."""

    CONCOURS = "concours"
    CANDIDAT = "candidat"
    CODIFICATION = "codification"


SECTION_LABELS: dict[FormSection, str] = {
    FormSection.CONCOURS: "Concours & Session",
    FormSection.CANDIDAT: "Informations du Candidat",
    FormSection.CODIFICATION: "Codification & Traçabilité Administrative",
}


class FieldStatus(str, Enum):
    """Statut de confiance d'un champ du formulaire dynamique.

    * ``valid``   — confiance > 85 % et règles métier OK (vert) ;
    * ``warning`` — confiance marginale 70–85 % mais format OK (orange) ;
    * ``error``   — confiance < 70 % OU règle métier violée (rouge) ;
    * ``empty``   — champ du gabarit non détecté par l'OCR (neutre).
    """

    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    EMPTY = "empty"


class FormFieldResult(BaseModel):
    """Un champ clé/valeur extrait du formulaire, avec son statut."""

    key: str  # ex: "cin", "nom", "date_naissance"
    label: str  # ex: "N° C.I.N ou N° du passeport"
    value: str  # ex: "09728320"
    confidence: float = Field(ge=0.0, le=1.0)  # ex: 0.94
    status: FieldStatus  # error -> affiché en ROUGE par les clients
    error_message: Optional[str] = (
        None  # ex: "Format CIN invalide (8 chiffres attendus)"
    )
    bounding_box: Optional[list[list[int]]] = None
    section: FormSection = FormSection.CANDIDAT
    section_label: str = ""
    corrections: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Trace de la correction au niveau des caractères, si la valeur "
            "brute OCR a été déconfondue : ``{original, kind, confidence}``. "
            "Aucun lexique statique n'est consulté."
        ),
    )


class FormAnalyzeRequest(BaseModel):
    """Entrée du moteur de post-traitement : les items OCR bruts d'une page."""

    file_name: str = Field(default="", max_length=255)
    items: list[OCRItem] = Field(default_factory=list)


class FormOverrideRequest(BaseModel):
    """Correction manuelle d'un formulaire (récupérée par la table Excel)."""

    page: int = Field(ge=1, description="Numéro de page (1-based).")
    values: dict[str, str] = Field(
        default_factory=dict,
        max_length=200,
        description="Clés de champs (ex. ``nom``) associées à leur valeur corrigée.",
    )


class AnalyzedFormResponse(BaseModel):
    """Formulaire structuré clé/valeur avec alertes de confiance."""

    file_name: str
    is_form: bool
    global_confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)
    fields: list[FormFieldResult] = Field(default_factory=list)


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
    corrections: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "État du correcteur de niveau caractère : ``{enabled, engine}`` "
            "(aucun lexique n'est plus embarqué)."
        ),
    )
