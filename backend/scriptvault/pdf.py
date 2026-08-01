"""Rasterisation PDF → images numpy (service partagé serveur/desktop)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("scriptvault.pdf")


class PDFRasterError(RuntimeError):
    """Échec d'ouverture ou de rendu d'un document PDF."""


def rasterize_pdf_bytes(data: bytes, dpi: int = 160) -> list[np.ndarray]:
    """Convertit chaque page d'un PDF en image BGR 8 bits.

    Args:
        data: Octets bruts du fichier PDF.
        dpi: Résolution de rendu (160 dpi ≈ 2200 px pour une page A4).

    Returns:
        Liste ordonnée d'images ``(H, W, 3)``, une par page.

    Raises:
        PDFRasterError: PDF illisible, chiffré ou vide.
    """
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise PDFRasterError(
            "PyMuPDF est requis pour le support des PDF: pip install PyMuPDF"
        ) from exc

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFRasterError(f"PDF illisible: {exc}") from exc

    pages: list[np.ndarray] = []
    try:
        for index, page in enumerate(document, start=1):
            try:
                pix = page.get_pixmap(dpi=dpi)
            except Exception as exc:
                raise PDFRasterError(
                    f"Rendu de la page {index} impossible: {exc}"
                ) from exc
            if pix.n == 0:
                continue
            array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
            pages.append(array)
            logger.debug("Page %d rastérisée: %dx%d", index, pix.width, pix.height)
    finally:
        document.close()

    if not pages:
        raise PDFRasterError("Le PDF ne contient aucune page rendable.")
    return pages
