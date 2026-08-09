"""Tests unitaires du moteur OCR — sans chargement des modèles ONNX (import léger)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from scriptvault import __version__ as scriptvault_version
from scriptvault import core_ocr
from scriptvault.core_ocr import (
    ImagePreprocessor,
    OCRBaseError,
    OCRImageError,
    OCRInitError,
)


def _text_like_image(angle_deg: float = 0.0) -> np.ndarray:
    """Image synthétique « lignes de texte » (rectangles noirs sur fond blanc)."""
    h, w = 300, 800
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in (80, 140, 200, 260):
        cv2.rectangle(img, (40, y), (w - 60, y + 24), (0, 0, 0), -1)
    if angle_deg:
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        img = cv2.warpAffine(
            img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
        )
    return img


# --------------------------------------------------------------------------- #
# Versions & hiérarchie d'exceptions
# --------------------------------------------------------------------------- #
def test_version():
    assert core_ocr.__version__ == scriptvault_version


def test_exception_hierarchy():
    assert issubclass(OCRInitError, OCRBaseError)
    assert issubclass(OCRImageError, OCRBaseError)


# --------------------------------------------------------------------------- #
# Prétraitement
# --------------------------------------------------------------------------- #
def test_preprocess_output_is_bgr8():
    img = _text_like_image()
    out = ImagePreprocessor().preprocess(img)
    assert out.shape == (300, 800, 3)
    assert out.dtype == np.uint8


def test_resize_applies_max_side_limit():
    """Le redimensionnement OCR est réellement appliqué (gain de vitesse majeur)."""
    big = np.full((2400, 1200, 3), 255, dtype=np.uint8)
    for y in range(200, 2400, 150):
        cv2.rectangle(big, (60, y), (1100, y + 26), (0, 0, 0), -1)
    out = ImagePreprocessor(max_side_len=1024).preprocess(big)
    assert max(out.shape[:2]) <= 1024
    assert out.shape[1] / out.shape[0] == pytest.approx(0.5, abs=0.01)
    assert out.dtype == np.uint8


def test_resize_skips_small_images():
    """Une image déjà plus petite que la limite n'est jamais agrandie."""
    img = _text_like_image()
    small = img[::2, ::2]
    out = ImagePreprocessor(max_side_len=1024).preprocess(small)
    assert out.shape[:2] == small.shape[:2]


def test_deskew_detects_rotation():
    applied = 7.0
    img = _text_like_image(angle_deg=applied)
    pre, meta = ImagePreprocessor()._preprocess_impl(
        img, denoise=False, clahe=False, deskew=True, binarize=False
    )
    angle = float(meta["deskew_angle"])
    assert abs(angle) >= 2.0, f"inclinaison non détectée: {angle}"
    assert abs(angle + applied) < 4.0, f"angle incohérent: {angle}"


def test_deskew_ignores_straight_image():
    """Régression: minAreaRect renvoie 90° (OpenCV >= 4.10) sur un document
    droit — la correction ne doit pas déclencher de rotation parasite."""
    img = _text_like_image()
    _, meta = ImagePreprocessor()._preprocess_impl(
        img, denoise=False, clahe=False, deskew=True, binarize=False
    )
    assert meta["deskew_angle"] == 0.0


def test_binarize_normalizes_polarity():
    h, w = 200, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)  # fond noir
    cv2.putText(
        img, "TEXTE", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3
    )
    out = ImagePreprocessor().preprocess(img)
    assert out.mean() > 127, "la binarisation doit ramener le fond clair"


def test_preprocess_file_metadata(tmp_path: Path):
    path = tmp_path / "scan.png"
    cv2.imwrite(str(path), _text_like_image())
    pre, meta = ImagePreprocessor().preprocess_file(path)
    assert meta["path"] == str(path)
    assert meta["size"] == (800, 300)
    assert meta["deskew_angle"] == 0.0


def test_read_image_unicode_path(tmp_path: Path):
    path = tmp_path / "scanné_001.png"  # accent : chemin Unicode
    # cv2.imwrite ne supporte pas les chemins Unicode (Windows) : écrire via
    # imencode + pathlib pour tester la lecture de notre pipeline.
    path.write_bytes(cv2.imencode(".png", _text_like_image())[1].tobytes())
    img = ImagePreprocessor().read_image(path)
    assert img.shape[:2] == (300, 800)


def test_read_image_missing_raises(tmp_path: Path):
    with pytest.raises(OCRImageError):
        ImagePreprocessor().read_image(tmp_path / "absent.png")


# --------------------------------------------------------------------------- #
# Zones manuscrites : pipeline adaptatif
# --------------------------------------------------------------------------- #
def _handwritten_image() -> np.ndarray:
    """Image « copie scannée » : texte sombre, fond taché + ombre en coin."""
    h, w = 320, 700
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.putText(
        img, "Didi Elloumi", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (45, 45, 45), 3
    )
    cv2.rectangle(img, (0, 0), (220, 90), (150, 155, 160), -1)  # ombre
    return img


def test_preprocess_handwritten_outputs_bgr8():
    img = _handwritten_image()
    out = ImagePreprocessor().preprocess_handwritten(img)
    assert out.shape[:2] == img.shape[:2]
    assert out.dtype == np.uint8
    assert out.shape[2] == 3


def test_preprocess_handwritten_normalizes_dark_background():
    """Un fond sombre (ombre) est ramené vers un fond clair: l'encre seule
    doit rester sombre — moyenne du résultat nettement au-dessus de 127."""
    img = _handwritten_image()
    out = ImagePreprocessor().preprocess_handwritten(img)
    assert out.mean() > 127.0


def test_preprocess_handwritten_grayscale_input():
    """Entrée en niveau de gris (2D) acceptée et sortie BGR."""
    gray = cv2.cvtColor(_handwritten_image(), cv2.COLOR_BGR2GRAY)
    out = ImagePreprocessor().preprocess_handwritten(gray)
    assert out.ndim == 3 and out.shape[2] == 3


def test_preprocess_handwritten_components_optional():
    img = _handwritten_image()
    base = ImagePreprocessor().preprocess_handwritten(img, binarize=False)
    assert base.dtype == np.uint8
    stroked = ImagePreprocessor().preprocess_handwritten(
        img, structural=True, binarize=False
    )
    assert stroked.shape == base.shape


def test_preprocess_handwritten_preserves_ink():
    """L'encre sombre (le texte) ne doit pas être effacée par la suppression
    d'ombre: il reste toujours du noir significatif sur l'image finale."""
    img = _handwritten_image()
    out = ImagePreprocessor().preprocess_handwritten(img)
    assert (out < 100).sum() > 500
