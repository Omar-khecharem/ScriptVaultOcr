"""Tests du moteur HTR ONNX (TrOCR) : tokenizer Unigram, pré-traitement,
détecteur de lignes et contrat OCR de :class:`LocalOCREngine`.

Les tests d'inférence réelle sont conditionnés à la présence des modèles
(``models/trocr/``) : la suite reste verte en CI sans téléchargement.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from scriptvault.core_ocr import LocalOCREngine, OCRInitError
from scriptvault.htr_engine import (
    HandwrittenLineDetector,
    TrOcrEngine,
    TrOcrTokenizer,
    preprocess_trocr,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TROCR_DIR = PROJECT_ROOT / "models" / "trocr"
HAVE_MODELS = (TROCR_DIR / "encoder_model_quantized.onnx").exists() and (
    TROCR_DIR / "decoder_model_merged_quantized.onnx"
).exists()


@pytest.fixture(scope="module")
def tokenizer() -> TrOcrTokenizer:
    return TrOcrTokenizer(TROCR_DIR / "tokenizer.json")


def _text_image(lines: list[str]) -> np.ndarray:
    """Image synthétique : plusieurs lignes de texte sombre sur fond clair."""
    h, w = 240, 700
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (30, 60 + index * 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (30, 30, 30),
            3,
        )
    return img


# --------------------------------------------------------------------------- #
# Tokenizer Unigram
# --------------------------------------------------------------------------- #
def test_tokenizer_special_ids(tokenizer: TrOcrTokenizer):
    assert tokenizer.bos_id == 0
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3


def test_tokenizer_roundtrip(tokenizer: TrOcrTokenizer):
    ids = tokenizer.encode_with_special("Didi Elloumi")
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "Didi Elloumi"


def test_tokenizer_accents_roundtrip(tokenizer: TrOcrTokenizer):
    """Les accents français doivent survivre au roundtrip (Unigram ▁)."""
    ids = tokenizer.encode("Élodie")
    decoded = tokenizer.decode(ids)
    assert decoded.lower() == "élodie"


def test_tokenizer_unknown_falls_back_unk(tokenizer: TrOcrTokenizer):
    """Un caractère hors vocabulaire ne fait pas planter l'encodage."""
    ids = tokenizer.encode("¥©ñ")
    assert ids  # segments produits (unk éventuellement)
    assert all(0 <= i < len(tokenizer.inverse) for i in ids)


def test_tokenizer_encode_empty(tokenizer: TrOcrTokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode(tokenizer.encode("   ")) == ""


# --------------------------------------------------------------------------- #
# Pré-traitement TrOCR
# --------------------------------------------------------------------------- #
def test_preprocess_trocr_shape():
    img = _text_image(["abc"])
    tensor = preprocess_trocr(img)
    assert tensor.shape == (1, 3, 384, 384)
    assert tensor.dtype == np.float32
    assert -1.0 <= tensor.min() <= 1.0
    assert -1.0 <= tensor.max() <= 1.0


def test_preprocess_trocr_grayscale():
    gray = cv2.cvtColor(_text_image(["abc"]), cv2.COLOR_BGR2GRAY)
    assert preprocess_trocr(gray).shape == (1, 3, 384, 384)


# --------------------------------------------------------------------------- #
# Détecteur de lignes
# --------------------------------------------------------------------------- #
def test_detector_finds_lines():
    img = _text_image(["Premiere ligne", "Deuxieme ligne", "Troisieme ligne"])
    lines = HandwrittenLineDetector().detect_lines(img)
    assert len(lines) >= 3
    ys = sorted(line.y0 for line in lines)
    assert ys[1] - ys[0] > 20  # lignes séparées verticalement


def test_detector_margins_inside_image():
    img = _text_image(["X"])
    lines = HandwrittenLineDetector().detect_lines(img)
    assert lines
    line = lines[0]
    assert 0 <= line.x0 <= line.x1 <= img.shape[1]
    assert 0 <= line.y0 <= line.y1 <= img.shape[0]
    box = line.box
    assert len(box) == 4
    assert box[0] == [float(line.x0), float(line.y0)]


def test_detector_blank_page():
    blank = np.full((200, 400, 3), 255, dtype=np.uint8)
    assert HandwrittenLineDetector().detect_lines(blank) == []


# --------------------------------------------------------------------------- #
# Moteur : chargement + contrat (nécessite les modèles)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_MODELS, reason="modèles TrOCR ONNX absents")
def test_engine_contract_predict_array():
    """Le contrat public (text/confidence/box) est préservé avec le HTR."""
    eng = LocalOCREngine(
        cpu_threads=4,
        preprocess_kwargs={"binarize": False},
    )
    try:
        assert eng._backend_name == "htr"
        assert eng.is_ready
        items = eng.predict_array(_text_image(["hello"]), preprocess=False)
        for item in items:
            assert isinstance(item["text"], str)
            assert 0.0 <= item["confidence"] <= 1.0
            box = item["box"]
            assert len(box) == 4
            assert all(len(point) == 2 for point in box)
    finally:
        eng.close()


@pytest.mark.skipif(not HAVE_MODELS, reason="modèles TrOCR ONNX absents")
def test_engine_default_resolves_project_models(tmp_path: Path):
    """Sans ``model_dir``, le moteur localise ``<projet>/models/trocr``."""
    eng = LocalOCREngine(cpu_threads=2, preprocess_kwargs={"binarize": False})
    try:
        assert eng._backend_name == "htr"
        assert eng._htr._encoder is not None
    finally:
        eng.close()


@pytest.mark.skipif(not HAVE_MODELS, reason="modèles TrOCR ONNX absents")
def test_engine_roi_labels_and_offsets():
    """ROI : chaque item porte ``label`` + ``box`` dans le repère page, et une
    zone sans encre est totalement ignorée (pas d'inférence inutile)."""
    eng = LocalOCREngine(cpu_threads=2, preprocess_kwargs={"binarize": True})
    try:
        img = _text_image(["LIGNE UNE"])
        h, w = img.shape[:2]
        rois = {
            "haut": (0.0, 0.0, 1.0, 0.5),
            "vide": (0.0, 0.55, 1.0, 0.95),
        }
        pages = eng.predict_pages_bytes(
            cv2.imencode(".png", img)[1].tobytes(), rois=rois, scan_barcode=False
        )
        items = pages[0]["items"]
        assert any(item["label"] == "haut" for item in items)
        assert not any(item["label"].startswith("vide") for item in items)
        for item in items:
            if item["label"] == "haut":
                assert item["box"] is not None
                assert all(len(point) == 2 for point in item["box"])
    finally:
        eng.close()


@pytest.mark.skipif(HAVE_MODELS, reason="modèles présents: le défaut pollue l'erreur")
def test_engine_missing_models_instructive(tmp_path: Path):
    """Sans modèles, l'initialisation échoue avec un message actionnable."""
    with pytest.raises(OCRInitError, match="download_trocr_models"):
        LocalOCREngine(model_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Smoke real model (déterminisme du greedy)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_MODELS, reason="modèles TrOCR ONNX absents")
def test_greedy_deterministic():
    eng = TrOcrEngine(TROCR_DIR, threads=4)
    try:
        img = _text_image(["test"])
        a = eng.recognize(img)
        b = eng.recognize(img)
        assert a == b
    finally:
        eng.close()
