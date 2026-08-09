"""Tests unitaires du correcteur de niveau caractère (sans lexique)."""

from __future__ import annotations

import json

import pytest
from scriptvault.char_corrector import (
    CharCorrector,
    CharPrediction,
    default_char_corrector,
    load_default_corrector,
)


@pytest.fixture(scope="module")
def corrector() -> CharCorrector:
    return CharCorrector()


# --------------------------------------------------------------------------- #
# Décodage — champs texte
# --------------------------------------------------------------------------- #
def test_text_digit_to_letter(corrector: CharCorrector):
    result: CharPrediction = corrector.correct("D1d1", "text")
    assert result.changed is True
    assert result.value == "Didi"


def test_text_zero_to_o(corrector: CharCorrector):
    result: CharPrediction = corrector.correct("D0di", "text")
    assert result.value == "DOdi"


def test_text_clean_word_unchanged(corrector: CharCorrector):
    result: CharPrediction = corrector.correct("Eya", "text")
    assert result.changed is False
    assert result.value == "Eya"


def test_text_untouched_without_confusion(corrector: CharCorrector):
    """Un mot sans chiffre/confusion classique n'est jamais modifié."""
    assert corrector.correct("Tnus", "text").value == "Tnus"
    assert corrector.correct("Elloom", "text").value == "Elloom"


def test_digit_field_letter_to_digit(corrector: CharCorrector):
    result: CharPrediction = corrector.correct("O92832O", "digit")
    assert result.value == "0928320"


def test_digit_field_mixed_ocr(corrector: CharCorrector):
    result: CharPrediction = corrector.correct("04550842", "digit")
    assert result.changed is False
    assert result.value == "04550842"


def test_empty_and_short_inputs():
    c = default_char_corrector()
    assert c.correct("", "text").value == ""
    assert c.correct("  ", "digit").value == ""


# --------------------------------------------------------------------------- #
# Construction / chargement de modèle
# --------------------------------------------------------------------------- #
def test_from_path(tmp_path) -> None:
    model = {
        "bigram": {"th": 1.0, "en": 0.5},
        "confusions": {"text": {"1": ["i", "I"]}},
        "unigram": {"e": 2.0},
    }
    path = tmp_path / "char_lm_test.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    loaded = CharCorrector.from_path(path)
    assert loaded.decode  # construction OK
    assert loaded.correct("D1d1", "text").value == "Didi"


def test_load_any_falls_back_to_embedded(tmp_path) -> None:
    corrector = load_default_corrector(tmp_path)
    assert isinstance(corrector, CharCorrector)
    assert corrector.decode("O", "digit").value == "0"


def test_load_any_picks_model_file(tmp_path) -> None:
    path = tmp_path / "char_lm_alpha.json"
    path.write_text(
        json.dumps({"bigram": {"ot": 100.0}}), encoding="utf-8"
    )
    loaded = load_default_corrector(tmp_path)
    assert isinstance(loaded, CharCorrector)


def test_invalid_json_is_ignored(tmp_path) -> None:
    path = tmp_path / "char_lm_broken.json"
    path.write_text("{pas du json", encoding="utf-8")
    assert isinstance(load_default_corrector(tmp_path), CharCorrector)


# --------------------------------------------------------------------------- #
# Décodage par mots / segments
# --------------------------------------------------------------------------- #
def test_compound_segment_preserves_separators(corrector: CharCorrector):
    result: CharPrediction = corrector.decode("D1 d1", "text")
    assert result.value == "Di di"


def test_decode_empty(corrector: CharCorrector):
    assert corrector.decode("", "text").value == ""


def test_confidence_in_unit_range(corrector: CharCorrector):
    assert 0.0 <= corrector.decode("D1d1", "text").confidence <= 1.0
