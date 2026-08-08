"""Endpoint de post-traitement : analyse de formulaire clé/valeur local.

Contrat:

* ``POST /api/form/analyze`` — reçoit les items OCR bruts d'une page
  (``{text, confidence, box}``) et renvoie le formulaire structuré avec le
  statut de chaque champ (``valid`` / ``warning`` / ``error``), la confiance
  globale et le temps de post-traitement (budget < 30 ms).

Aucune inférence ni aucun appel réseau : le client (desktop ou web) peut
enchaîner OCR → analyse en quelques millisecondes, ou ré-analyser un
résultat déjà conservé.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...form_analyzer import LocalFormAnalyzer
from ...schemas import AnalyzedFormResponse, FormAnalyzeRequest

router = APIRouter(prefix="/api/form", tags=["form"])

#: L'analyseur est sans état : une seule instance partagée, thread-safe.
_ANALYZER = LocalFormAnalyzer()


@router.post(
    "/analyze",
    response_model=AnalyzedFormResponse,
    summary="Structurer les items OCR en formulaire clé/valeur validé",
)
async def form_analyze(payload: FormAnalyzeRequest) -> AnalyzedFormResponse:
    """Transforme les lignes OCR en champs typés et classe leur niveau de risque.

    Le corps attendu est le tableau ``items`` produit par ``/api/ocr/single``
    (une page) : ``[{"text", "confidence", "box"}, ...]``.
    """
    raw_items = [
        {"text": item.text, "confidence": item.confidence, "box": item.box}
        for item in payload.items
    ]
    return _ANALYZER.analyze_page(raw_items, file_name=payload.file_name)
