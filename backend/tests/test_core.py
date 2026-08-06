"""Tests unitaires du moteur OCR — sans PaddlePaddle (import léger)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from scriptvault import __version__ as scriptvault_version
from scriptvault import core_ocr
from scriptvault.core_ocr import (
    ImagePreprocessor,
    LocalOCREngine,
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
# Normalisation des sorties (parseurs 2.x / 3.x)
# --------------------------------------------------------------------------- #
def test_poly_to_box_variants():
    proc = LocalOCREngine
    assert proc._poly_to_box([10, 20, 30, 20, 30, 40, 10, 40]) == [
        [10, 20],
        [30, 20],
        [30, 40],
        [10, 40],
    ]
    quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    box = proc._poly_to_box(quad)
    assert len(box) == 4 and box[0] == [0, 0]
    assert proc._poly_to_box([]) == []
    assert proc._poly_to_box("invalide") == []


class _ResultLike:
    """Reproduit ``paddlex.inference.pipelines.ocr.result.OCRResult``:
    dict-like avec ``get()`` mais sans attributs ``__dict__``."""

    def __init__(self, data):
        object.__setattr__(self, "_data", data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getattr__(self, name):
        raise AttributeError(name)


def _parse_instance():
    """Instance légère de LocalOCREngine (sans PaddlePaddle) pour les
    parseurs purs."""
    return object.__new__(LocalOCREngine)


def test_parse_v3_dictlike():
    raw = [
        _ResultLike(
            {
                "rec_texts": ["Bonjour", "Monde"],
                "rec_scores": [0.99, 0.85],
                "rec_polys": [
                    [[0, 0], [50, 0], [50, 10], [0, 10]],
                    [[0, 20], [40, 20], [40, 30], [0, 30]],
                ],
            }
        )
    ]
    entries = _parse_instance()._parse_v3(raw)
    assert [e["text"] for e in entries] == ["Bonjour", "Monde"]
    assert entries[0]["confidence"] == pytest.approx(0.99)
    assert entries[0]["box"] == [[0, 0], [50, 0], [50, 10], [0, 10]]


def test_parse_v3_plain_dict_and_dt_polys():
    raw = [
        {
            "rec_texts": ["X"],
            "rec_scores": [1.0],
            "dt_polys": [[[1, 1], [2, 1], [2, 2], [1, 2]]],
        }
    ]
    entries = _parse_instance()._parse_v3(raw)
    assert entries[0]["text"] == "X"
    assert entries[0]["box"] == [[1, 1], [2, 1], [2, 2], [1, 2]]


def test_parse_v3_empty():
    assert _parse_instance()._parse_v3([None, {}]) == []


def test_parse_v2_nested():
    box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    raw = [[[box, ("Hello", 0.99)]], [[box, ("World", 0.8)]]]
    entries = _parse_instance()._parse_v2(raw)
    assert [e["text"] for e in entries] == ["Hello", "World"]
    assert entries[1]["confidence"] == pytest.approx(0.8)
