"""Export des textes reconnus en TXT / DOCX / PDF (couche texte).

Service partagé entre l'API HTTP et le desktop : même implémentation,
mêmes formats, zéro duplication.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate


class ExportError(RuntimeError):
    """Échec de génération d'un document exporté."""


def export_text(text: str, format_name: str) -> bytes:
    """Sérialise ``text`` dans le format demandé.

    Args:
        text: Contenu à exporter.
        format_name: ``"txt"``, ``"docx"`` ou ``"pdf"``.

    Returns:
        Octets du document généré.

    Raises:
        ExportError: Format inconnu ou échec de génération.
    """
    if format_name == "txt":
        return text.encode("utf-8")
    if format_name == "docx":
        return _export_docx(text)
    if format_name == "pdf":
        return _export_pdf(text)
    raise ExportError(f"Format d'export inconnu: {format_name!r}")


def _export_docx(text: str) -> bytes:
    """Génère un document Word (.docx) en mémoire."""
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise ExportError("python-docx est requis: pip install python-docx") from exc

    import io

    try:
        document = Document()
        for line in text.splitlines():
            document.add_paragraph(line)
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()
    except Exception as exc:
        raise ExportError(f"Génération DOCX échouée: {exc}") from exc


def _export_pdf(text: str) -> bytes:
    """Génère un PDF avec couche texte sélectionnable, en mémoire."""
    import io

    try:
        styles = getSampleStyleSheet()
        body = styles["BodyText"]
        body.fontSize = 11
        story = [
            Paragraph(escape(line) or "&nbsp;", body) for line in text.splitlines()
        ]
        buffer = io.BytesIO()
        SimpleDocTemplate(
            buffer,
            pagesize=A4,
            title="ScriptVault OCR",
            author="ScriptVault OCR",
        ).build(story)
        return buffer.getvalue()
    except Exception as exc:
        raise ExportError(f"Génération PDF échouée: {exc}") from exc
