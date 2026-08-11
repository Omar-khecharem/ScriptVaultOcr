"""Tests unitaires du lecteur VLM local (``scriptvault.vlm_reader``).

Aucun serveur Ollama/llama.cpp ni modèle n'est requis : le transport HTTP est
simulé par un client factice injecté (``client_factory``). Les chemins de
repli d'urgence (timeout, HTTP, JSON invalide) sont couverts sans réseau.

L'intégration pipeline est testée sur
:func:`scriptvault.image_processing._re_recognize_handwritten` (routage
label-aware du crop manuscrit vers le lecteur VLM).
"""

from __future__ import annotations

import asyncio
import base64
from typing import Callable

import cv2
import numpy as np
import pytest
from scriptvault.image_processing import FieldBand, _re_recognize_handwritten
from scriptvault.vlm_reader import (
    LocalVLMReader,
    VLMConfig,
    VLMResultError,
    build_system_prompt,
    build_user_prompt,
    encode_image_base64,
    parse_band_grid_json,
    parse_vlm_json,
    sanitize_vlm_text,
)


# --------------------------------------------------------------------------- #
# Doubles de test : client HTTP factice
# --------------------------------------------------------------------------- #
class FakeResponse:
    """Réponse HTTP factice (``status_code``, ``json()``)."""

    def __init__(self, status_code: int = 200, payload: object = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("corps non JSON")
        return self._payload

    def __bool__(self) -> bool:
        return self.status_code < 400


def _vlm_response(text: str, status_code: int = 200) -> FakeResponse:
    """Réponse ``/api/chat`` standard d'Ollama."""
    return FakeResponse(status_code, {"message": {"content": text}})


class FakeClient:
    """Client HTTP factice : réponses programmables + capture des requêtes.

    ``responses`` est consommé séquentiellement (la dernière réponse sert de
    défaut). Une entrée peut être une exception (transport en échec), un
    :class:`FakeResponse`, ou un entier (statut HTTP seul).
    """

    def __init__(self, responses: list[object] | None = None) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[tuple[str, dict]] = []
        self.responses: list[object] = list(responses or [])
        self._index = 0
        self.closed = False

    def _next(self) -> object:
        if not self.responses:
            return _vlm_response('{"text": "defaut", "confidence": 0.9}')
        if self._index < len(self.responses):
            item = self.responses[self._index]
            self._index += 1
            return item
        return self.responses[-1]

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.posts.append((url, kwargs))
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, int):
            return FakeResponse(item, None)
        assert isinstance(item, FakeResponse)
        return item

    async def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.gets.append((url, kwargs))
        return FakeResponse(200, {"models": [{"name": "qwen2.5vl:7b"}]})

    async def aclose(self) -> None:
        self.closed = True


def _reader(
    responses: list[object] | None = None,
    *,
    fallback: Callable[[np.ndarray], tuple[str, float]] | None = None,
    timeout_s: float = 2.0,
) -> tuple[LocalVLMReader, FakeClient]:
    """Construit un lecteur VLM branché sur un client factice."""
    client = FakeClient(responses)
    config = VLMConfig(timeout_s=timeout_s)
    reader = LocalVLMReader(
        config=config, fallback=fallback, client_factory=lambda: client
    )
    return reader, client


def _htr_fallback(crop: np.ndarray) -> tuple[str, float]:
    return ("Repli Trocr", 0.61)


_htr_fallback.name = "htr"  # type: ignore[attr-defined]


def _ink_crop(text: str = "Didi Elloumi") -> np.ndarray:
    """Crop synthétique « manuscrit » (texte sombre sur fond clair)."""
    img: np.ndarray = np.full((120, 420, 3), 245, dtype=np.uint8)
    cv2.putText(
        img, text, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2
    )
    return img


# --------------------------------------------------------------------------- #
# Encodage base64 du crop
# --------------------------------------------------------------------------- #
def test_encode_image_base64_roundtrip():
    crop = _ink_crop()
    encoded = encode_image_base64(crop)
    assert isinstance(encoded, str) and encoded
    raw = base64.b64decode(encoded)
    assert raw.startswith(b"\xff\xd8")  # magic JPEG
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == crop.shape[:2]


def test_encode_image_base64_downscales_oversized_crop():
    big = np.full((2400, 1200, 3), 255, dtype=np.uint8)
    encoded = encode_image_base64(big, max_side=512)
    raw = base64.b64decode(encoded)
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 512


def test_encode_image_base64_grayscale_and_errors():
    gray = cv2.cvtColor(_ink_crop(), cv2.COLOR_BGR2GRAY)
    assert encode_image_base64(gray)
    with pytest.raises(ValueError):
        encode_image_base64(np.zeros((0, 0, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Prompts contextuels (Stratégie de lecture directe)
# --------------------------------------------------------------------------- #
def test_system_prompt_etablissement_contient_acronymes():
    prompt = build_system_prompt("etablissement")
    for acronym in ("IPEIN", "FST", "ISSATSO"):
        assert acronym in prompt


def test_system_prompt_nom_contient_lexique_tunisien():
    assert "TRABELSI" in build_system_prompt("nom")
    assert "TRABELSI" in build_system_prompt("prenom")


def test_system_prompt_toujours_contrainte_json_strict():
    for field in ("nom", "prenom", "etablissement"):
        prompt = build_system_prompt(field)
        assert '"text"' in prompt
        assert "confidence" in prompt


def test_user_prompt_nomme_le_champ():
    assert "Prénom" in build_user_prompt("prenom")
    assert "Établissement" in build_user_prompt("etablissement")


# --------------------------------------------------------------------------- #
# Parsing strict du JSON de sortie
# --------------------------------------------------------------------------- #
def test_parse_vlm_json_valide():
    assert parse_vlm_json('{"text": "Didi", "confidence": 0.92}') == ("Didi", 0.92)


def test_parse_vlm_json_fences_markdown():
    raw = '```json\n{"text": "IPEIN", "confidence": 0.9}\n```'
    assert parse_vlm_json(raw) == ("IPEIN", 0.9)


def test_parse_vlm_json_bruit_autour():
    raw = 'Voici le résultat : {"text": "Salma", "confidence": 0.88}. Fin.'
    assert parse_vlm_json(raw) == ("Salma", 0.88)


def test_parse_vlm_json_sans_json_leve():
    with pytest.raises(VLMResultError):
        parse_vlm_json("je ne lis aucune écriture sur cette zone")


def test_parse_vlm_json_champ_text_vide_leve():
    with pytest.raises(VLMResultError):
        parse_vlm_json('{"text": "", "confidence": 0.9}')


def test_parse_vlm_json_confidence_hors_borne_leve():
    with pytest.raises(VLMResultError):
        parse_vlm_json('{"text": "x", "confidence": 1.7}')
    with pytest.raises(VLMResultError):
        parse_vlm_json('{"text": "x", "confidence": -0.1}')


# --------------------------------------------------------------------------- #
# Normalisation selon la contrainte du champ
# --------------------------------------------------------------------------- #
def test_sanitize_etablissement_rapproche_acronyme():
    assert sanitize_vlm_text("issatso", "etablissement") == "ISSATSO"
    assert sanitize_vlm_text("f.s.t", "etablissement") == "FST"


def test_sanitize_etablissement_conserve_inconnu():
    assert sanitize_vlm_text("ZARZIS-X", "etablissement") == "ZARZISX"


def test_sanitize_nom_capitalise_et_nettoye():
    assert sanitize_vlm_text("did i  elloumi!", "nom") == "Did I Elloumi"
    assert sanitize_vlm_text("amira   ben-salah.", "prenom") == "Amira Ben Salah"


# --------------------------------------------------------------------------- #
# Lecture VLM — chemin nominal
# --------------------------------------------------------------------------- #
def test_read_handwritten_crop_happy_path():
    reader, client = _reader(
        [_vlm_response('{"text": "Didi Elloumi", "confidence": 0.92}')]
    )
    try:
        result = asyncio.run(reader.read_handwritten_crop(_ink_crop(), "nom"))
        assert result["source"] == "vlm"
        assert result["engine"] == "vlm:qwen2.5vl:7b"
        assert result["text"] == "Didi Elloumi"
        assert result["confidence"] == 0.92
        assert result["field_type"] == "nom"
        assert result["latency_ms"] >= 0.0
        # Le payload part bien avec l'image encodée + le prompt contextuel.
        url, payload = client.posts[0]
        assert url.endswith("/api/chat")
        assert payload["json"]["model"] == "qwen2.5vl:7b"
        assert payload["json"]["format"] == "json"
        assert payload["json"]["stream"] is False
        messages = payload["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert "TRABELSI" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["images"][0]
    finally:
        reader.close()


def test_read_handwritten_crop_etablissement_normalise():
    reader, _ = _reader([_vlm_response('{"text": "ipein", "confidence": 0.9}')])
    try:
        result = asyncio.run(
            reader.read_handwritten_crop(_ink_crop("IPEIN"), "etablissement")
        )
        assert result["text"] == "IPEIN"
        assert result["field_type"] == "etablissement"
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Repli d'urgence : timeout strict (2 s) puis HTTP puis JSON invalide
# --------------------------------------------------------------------------- #
class _SlowClient(FakeClient):
    """Client factice qui ne répond jamais (délai > timeout)."""

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.posts.append((url, kwargs))
        await asyncio.sleep(30)
        return _vlm_response("{}")  # pragma: no cover - jamais atteint


def test_read_handwritten_crop_timeout_bascule_sur_repli():
    reader = LocalVLMReader(
        config=VLMConfig(timeout_s=0.1),
        fallback=_htr_fallback,
        client_factory=lambda: _SlowClient(),
    )
    try:
        result = asyncio.run(reader.read_handwritten_crop(_ink_crop(), "prenom"))
        assert result["source"] == "fallback"
        assert result["engine"] == "htr"
        assert result["text"] == "Repli Trocr"
        assert result["confidence"] == 0.61
    finally:
        reader.close()


def test_read_handwritten_crop_http_error_bascule_sur_repli():
    reader, _ = _reader([_vlm_response("{}", status_code=503)], fallback=_htr_fallback)
    try:
        result = asyncio.run(reader.read_handwritten_crop(_ink_crop(), "nom"))
        assert result["source"] == "fallback"
        assert result["engine"] == "htr"
        assert result["text"] == "Repli Trocr"
    finally:
        reader.close()


def test_read_handwritten_crop_transport_error_bascule_sur_repli():
    reader, _ = _reader([RuntimeError("connexion refusée")], fallback=_htr_fallback)
    try:
        result = asyncio.run(reader.read_handwritten_crop(_ink_crop(), "nom"))
        assert result["source"] == "fallback"
        assert result["text"] == "Repli Trocr"
    finally:
        reader.close()


def test_read_handwritten_crop_json_invalide_bascule_sur_repli():
    reader, _ = _reader([_vlm_response("aucune écriture lisible")], fallback=_htr_fallback)
    try:
        result = asyncio.run(reader.read_handwritten_crop(_ink_crop(), "nom"))
        assert result["source"] == "fallback"
        assert result["text"] == "Repli Trocr"
    finally:
        reader.close()


def test_read_handwritten_crop_repli_vide_sans_fallback():
    reader, _ = _reader([RuntimeError("down")], fallback=None)
    try:
        result = asyncio.run(reader.read_handwritten_crop(_ink_crop(), "nom"))
        assert result["source"] == "fallback"
        assert result["engine"] == "aucun"
        assert result["text"] == ""
        assert result["confidence"] == 0.0
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Batch : 3 champs manuscrits en parallèle (asyncio.gather)
# --------------------------------------------------------------------------- #
def test_batch_trois_champs_paralleles_avec_repli_par_champ():
    responses = [
        _vlm_response('{"text": "Didi", "confidence": 0.95}'),
        RuntimeError("timeout serveur"),
        _vlm_response('{"text": "ISSATSO", "confidence": 0.89}'),
    ]
    reader, _ = _reader(responses, fallback=_htr_fallback)
    crops = [_ink_crop("Didi"), _ink_crop("X"), _ink_crop("ISSATSO")]
    fields = ["nom", "prenom", "etablissement"]
    try:
        results = asyncio.run(reader.read_handwritten_crops_batch(crops, fields))
        assert len(results) == 3
        assert results[0]["source"] == "vlm"
        assert results[0]["text"] == "Didi"
        assert results[1]["source"] == "fallback"
        assert results[1]["engine"] == "htr"
        assert results[2]["source"] == "vlm"
        assert results[2]["text"] == "ISSATSO"
        # L'ordre des champs est préservé.
        assert [r["field_type"] for r in results] == fields
    finally:
        reader.close()


def test_batch_tailles_differentes_leve_valueerror():
    reader, _ = _reader()
    try:
        with pytest.raises(ValueError):
            asyncio.run(
                reader.read_handwritten_crops_batch([_ink_crop()], ["nom", "prenom"])
            )
    finally:
        reader.close()


def test_batch_timeout_global_borne():
    """Avec un timeout de 0.05 s, le lot entier reste < 1 s (parallélisme)."""
    reader = LocalVLMReader(
        config=VLMConfig(timeout_s=0.05),
        fallback=_htr_fallback,
        client_factory=lambda: _SlowClient(),
    )
    crops = [_ink_crop() for _ in range(3)]
    try:
        results = asyncio.run(
            reader.read_handwritten_crops_batch(crops, ["nom", "prenom", "etablissement"])
        )
        assert all(r["source"] == "fallback" for r in results)
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Pont synchrone (intégration pipeline core_ocr)
# --------------------------------------------------------------------------- #
def test_sync_read_handwritten_crop_depuis_thread():
    reader, _ = _reader(
        [_vlm_response('{"text": "Salma Gharbi", "confidence": 0.9}')]
    )
    try:
        result = reader.sync_read_handwritten_crop(_ink_crop("Salma"), "prenom")
        assert result["text"] == "Salma Gharbi"
        assert result["source"] == "vlm"
    finally:
        reader.close()


def test_sync_read_handwritten_crop_repli_si_timeout():
    reader = LocalVLMReader(
        config=VLMConfig(timeout_s=0.1),
        fallback=_htr_fallback,
        client_factory=lambda: _SlowClient(),
    )
    try:
        result = reader.sync_read_handwritten_crop(_ink_crop(), "nom")
        assert result["source"] == "fallback"
        assert result["text"] == "Repli Trocr"
    finally:
        reader.close()


def test_lecteur_ferme_bascule_gracieusement_sur_repli():
    """Un lecteur fermé ne lève jamais : il dégrade sur le repli (ou vide)."""
    reader, _ = _reader([], fallback=_htr_fallback)
    reader.close()
    result = asyncio.run(reader.read_handwritten_crop(_ink_crop(), "nom"))
    assert result["source"] == "fallback"
    assert result["text"] == "Repli Trocr"


# --------------------------------------------------------------------------- #
# Disponibilité du serveur local
# --------------------------------------------------------------------------- #
def test_is_available_depuis_client_factice():
    reader, client = _reader()
    try:
        assert asyncio.run(reader.is_available()) is True
        assert client.gets[0][0].endswith("/api/tags")
    finally:
        reader.close()


def test_is_available_false_si_transport_en_erreur():
    class _BoomClient(FakeClient):
        async def get(self, url: str, **kwargs: object) -> FakeResponse:
            raise RuntimeError("serveur VLM éteint")

    reader = LocalVLMReader(client_factory=lambda: _BoomClient())
    try:
        assert asyncio.run(reader.is_available()) is False
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Configuration depuis l'environnement
# --------------------------------------------------------------------------- #
def test_vlm_config_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCRIPTVAULT_VLM_URL", "http://127.0.0.1:11434/")
    monkeypatch.setenv("SCRIPTVAULT_VLM_MODEL", "qwen2vl:7b")
    monkeypatch.setenv("SCRIPTVAULT_VLM_TIMEOUT_S", "1.5")
    monkeypatch.setenv("SCRIPTVAULT_VLM_JSON", "0")
    config = VLMConfig.from_env()
    assert config.base_url == "http://127.0.0.1:11434"  # slash final retiré
    assert config.model == "qwen2vl:7b"
    assert config.timeout_s == 1.5
    assert config.json_mode is False


def test_vlm_config_env_invalide_reste_aux_defauts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCRIPTVAULT_VLM_TIMEOUT_S", "abc")
    config = VLMConfig.from_env()
    assert config.timeout_s == 2.0


# --------------------------------------------------------------------------- #
# Intégration pipeline : routage label-aware du crop manuscrit
# --------------------------------------------------------------------------- #
def _fake_form_page() -> np.ndarray:
    return np.full((800, 1000, 3), 255, dtype=np.uint8)


def _band() -> FieldBand:
    return FieldBand(y0=100, y1=120, dots_x0=300, dots_x1=900, y_center=110)


def _labeled_item(text: str = "Nom:.D") -> dict:
    return {
        "label": "nom",
        "text": text,
        "confidence": 0.9,
        "box": [[300, 100], [900, 100], [900, 120], [300, 120]],
    }


def test_re_recognize_handwritten_route_label_vers_lecteur():
    """Le lecteur VLM reçoit bien (crop, field_type) — prompt contextuel."""
    calls: list[tuple[str, np.ndarray]] = []

    def reader(crop: np.ndarray, field_type: str) -> tuple[str, float]:
        calls.append((field_type, crop))
        return ("Didi Elloumi", 0.95)

    image = _fake_form_page()
    pairs = [(_band(), image[80:140, 290:950])]
    items = [_labeled_item()]
    out = _re_recognize_handwritten(image, pairs, items, reader)
    assert len(calls) == 1
    assert calls[0][0] == "nom"
    assert calls[0][1].shape[0] > 0
    assert out[0]["text"] == "Didi Elloumi"
    assert out[0]["confidence"] == 0.95


def test_re_recognize_handwritten_ignore_champ_hors_liste():
    calls: list[str] = []

    def reader(crop: np.ndarray, field_type: str) -> tuple[str, float]:
        calls.append(field_type)
        return ("Ignore", 0.9)

    image = _fake_form_page()
    pairs = [(_band(), image[80:140, 290:950])]
    items = [_labeled_item()]
    _re_recognize_handwritten(image, pairs, items, reader, handwritten_fields=("nom",))
    assert calls == ["nom"]


def test_re_recognize_handwritten_lecture_faible_conservee():
    """Confiance < 0.5 : la lecture composite reste en place."""

    def reader(crop: np.ndarray, field_type: str) -> tuple[str, float]:
        return ("brouillon", 0.3)

    image = _fake_form_page()
    pairs = [(_band(), image[80:140, 290:950])]
    items = [_labeled_item()]
    out = _re_recognize_handwritten(image, pairs, items, reader)
    assert out[0]["text"] == "Nom:.D"


def test_re_recognize_handwritten_champ_absent_ignore():
    calls: list[str] = []

    def reader(crop: np.ndarray, field_type: str) -> tuple[str, float]:
        calls.append(field_type)
        return ("X", 0.9)

    image = _fake_form_page()
    pairs = [(_band(), image[80:140, 290:950])]
    items = [{"label": "prenom", "text": "P:.", "confidence": 0.9, "box": []}]
    out = _re_recognize_handwritten(image, pairs, items, reader, handwritten_fields=("nom",))
    assert calls == []
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# Grille de bandes : lecture du formulaire en UN appel VLM
# --------------------------------------------------------------------------- #
def test_parse_band_grid_json_valide():
    raw = '{"rows": [{"row": 1, "text": "Nom : Didi Elloumi", "confidence": 0.95}, {"row": 2, "text": "Prénom : Salma", "confidence": 0.9}]}'
    assert parse_band_grid_json(raw, 1, 2) == [
        (1, "Nom : Didi Elloumi", 0.95),
        (2, "Prénom : Salma", 0.9),
    ]


def test_parse_band_grid_json_tolere_fences_et_bruit():
    raw = 'Voici le résultat :\n```json\n{"rows": [{"row": 1, "text": "x", "confidence": 0.8}]}\n```'
    assert parse_band_grid_json(raw, 1, 1) == [(1, "x", 0.8)]


def test_parse_band_grid_json_ignore_lignes_hors_bornes():
    raw = '{"rows": [{"row": 0, "text": "hors"}, {"row": 9, "text": "hors2"}, {"row": 3, "text": "ok", "confidence": 0.7}]}'
    assert parse_band_grid_json(raw, 1, 4) == [(3, "ok", 0.7)]


def test_parse_band_grid_json_vide_leve():
    with pytest.raises(VLMResultError):
        parse_band_grid_json("aucune écriture", 1, 3)
    with pytest.raises(VLMResultError):
        parse_band_grid_json('{"rows": []}', 1, 3)
    with pytest.raises(VLMResultError):
        parse_band_grid_json('{"pas_rows": 1}', 1, 3)


def test_read_form_band_grid_happy_path():
    grid = _ink_crop("Grille")
    payload = '{"rows": [{"row": 1, "text": "Nom : Didi Elloumi", "confidence": 0.95}, {"row": 2, "text": "Prénom : Salma", "confidence": 0.9}]}'
    reader, client = _reader([_vlm_response(payload)])
    try:
        result = asyncio.run(reader.read_form_band_grid(grid, 1, 2))
        assert result == [
            (1, "Nom : Didi Elloumi", 0.95),
            (2, "Prénom : Salma", 0.9),
        ]
        url, payload = client.posts[0]
        assert url.endswith("/api/chat")
        assert payload["json"]["model"] == "qwen2.5vl:7b"
        assert payload["json"]["format"] == "json"
        messages = payload["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert "numérotées" in messages[0]["content"]
        assert "1 à 2" in messages[1]["content"]
        assert messages[1]["images"][0]
    finally:
        reader.close()


def test_read_form_band_grid_borne_unique():
    grid = _ink_crop("G")
    reader, _ = _reader(
        [_vlm_response('{"rows": [{"row": 7, "text": "Date : 12/03/2005", "confidence": 0.8}]}')]
    )
    try:
        result = asyncio.run(reader.read_form_band_grid(grid, 7, 7))
        assert result == [(7, "Date : 12/03/2005", 0.8)]
    finally:
        reader.close()


def test_read_form_band_grid_timeout_retourne_none():
    reader = LocalVLMReader(
        config=VLMConfig(grid_timeout_s=0.1),
        client_factory=lambda: _SlowClient(),
    )
    try:
        result = asyncio.run(reader.read_form_band_grid(_ink_crop(), 1, 2))
        assert result is None
    finally:
        reader.close()


def test_read_form_band_grid_json_invalide_retourne_none():
    reader, _ = _reader([_vlm_response("je ne lis rien")])
    try:
        result = asyncio.run(reader.read_form_band_grid(_ink_crop(), 1, 2))
        assert result is None
    finally:
        reader.close()


def test_sync_read_form_band_grid():
    reader, _ = _reader(
        [_vlm_response('{"rows": [{"row": 1, "text": "Nom : Didi Elloumi", "confidence": 0.9}]}')]
    )
    try:
        result = reader.sync_read_form_band_grid(_ink_crop(), 1, 1)
        assert result == [(1, "Nom : Didi Elloumi", 0.9)]
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Construction de la grille + intégration dans _transcribe_band_rows
# --------------------------------------------------------------------------- #
def test_build_band_grid_decoupe_et_numérote():
    from scriptvault.image_processing import _GRID_MARGIN_W, _build_band_grid

    rows = [_ink_crop("A") for _ in range(20)]
    grids = _build_band_grid(rows, 0)
    assert len(grids) == 2  # 16 + 4
    first_grid, first, last = grids[0]
    assert (first, last) == (1, 16)
    second_grid, first2, last2 = grids[1]
    assert (first2, last2) == (17, 20)
    assert first_grid.shape[1] > _GRID_MARGIN_W
    assert second_grid.shape[0] > 0


def test_transcribe_band_rows_grille_vlm_labelise():
    from scriptvault.image_processing import FieldBand, _transcribe_band_rows

    def recognize_crop(crop: np.ndarray) -> list[dict]:
        return []  # jamais appelé : la grille VLM prend le relais

    def grid_reader(
        grid: np.ndarray, first: int, last: int
    ) -> list[tuple[int, str, float]]:
        return [
            (first, "Nom : ...Didi Elloumi...", 0.95),
            (first + 1, "Prénom : ...Salma...", 0.9),
            (first + 2, "Établissement d'origine : ISSATSO", 0.88),
        ]

    image = _fake_form_page()
    pairs = [
        (FieldBand(y0=100, y1=120, dots_x0=300, dots_x1=900, y_center=110), image[80:140, 20:980]),
        (FieldBand(y0=200, y1=220, dots_x0=300, dots_x1=900, y_center=210), image[180:240, 20:980]),
        (FieldBand(y0=300, y1=320, dots_x0=300, dots_x1=900, y_center=310), image[280:340, 20:980]),
    ]
    out = _transcribe_band_rows(
        image, pairs, recognize_crop, None, band_grid_reader=grid_reader
    )
    labels = {item["label"]: item["text"] for item in out}
    assert labels.get("nom") == "Nom : ...Didi Elloumi..."
    assert labels.get("prenom") == "Prénom : ...Salma..."
    assert labels.get("etablissement") == "Établissement d'origine : ISSATSO"
    for item in out:
        box = item["box"]
        assert len(box) == 4
        assert all(len(pt) == 2 for pt in box)


def test_transcribe_band_rows_grille_echo_on_repli_trocr():
    from scriptvault.image_processing import FieldBand, _transcribe_band_rows

    calls: list[str] = []

    def recognize_crop(crop: np.ndarray) -> list[dict]:
        calls.append("trocr")
        return []

    def grid_reader(grid: np.ndarray, first: int, last: int) -> None:
        return None  # VLM indisponible → repli TrOCR composite

    image = _fake_form_page()
    pairs = [
        (FieldBand(y0=100, y1=120, dots_x0=300, dots_x1=900, y_center=110), image[80:140, 20:980]),
    ]
    out = _transcribe_band_rows(
        image, pairs, recognize_crop, None, band_grid_reader=grid_reader
    )
    assert calls == ["trocr"]
    assert out == []


def _three_bands(image: np.ndarray):
    from scriptvault.image_processing import FieldBand

    return [
        (FieldBand(y0=100, y1=120, dots_x0=300, dots_x1=900, y_center=110), image[80:140, 20:980]),
        (FieldBand(y0=200, y1=220, dots_x0=300, dots_x1=900, y_center=210), image[180:240, 20:980]),
        (FieldBand(y0=300, y1=320, dots_x0=300, dots_x1=900, y_center=310), image[280:340, 20:980]),
    ]


def test_transcribe_band_rows_rangee_manquee_relue():
    """Une rangée absente de la grille VLM est relue en mono-ligne agrandie."""
    from scriptvault.image_processing import _transcribe_band_rows

    def recognize_crop(crop: np.ndarray) -> list[dict]:
        return []

    def grid_reader(
        grid: np.ndarray, first: int, last: int
    ) -> list[tuple[int, str, float]]:
        if first == last:
            return [(1, "Prénom : ...Salma...", 0.88)]
        return [
            (first, "Nom : ...Didi Elloumi...", 0.95),
            (first + 2, "Établissement d'origine : ISSATSO", 0.88),
        ]

    image = _fake_form_page()
    out = _transcribe_band_rows(
        image, _three_bands(image), recognize_crop, None, band_grid_reader=grid_reader
    )
    labels = {item["label"]: item["text"] for item in out}
    assert labels.get("nom") == "Nom : ...Didi Elloumi..."
    assert labels.get("prenom") == "Prénom : ...Salma..."
    assert labels.get("etablissement") == "Établissement d'origine : ISSATSO"


def test_transcribe_band_rows_libelle_sans_valeur_relu_manuscrit():
    """« Nom : » sans valeur → la zone de valeur est relue par le lecteur dédié."""
    from scriptvault.image_processing import _transcribe_band_rows

    def recognize_crop(crop: np.ndarray) -> list[dict]:
        return []

    def grid_reader(
        grid: np.ndarray, first: int, last: int
    ) -> list[tuple[int, str, float]]:
        return [(first, "Nom :", 0.9), (first + 1, "Prénom : ...Salma...", 0.9)]

    def handwritten_reader(crop: np.ndarray, field: str) -> tuple[str, float]:
        assert field == "nom"
        return ("Didi Elloumi", 0.92)

    image = _fake_form_page()
    out = _transcribe_band_rows(
        image,
        _three_bands(image),
        recognize_crop,
        None,
        band_grid_reader=grid_reader,
        handwritten_reader=handwritten_reader,
        handwritten_fields=("nom", "prenom", "etablissement"),
    )
    labels = {item["label"]: item["text"] for item in out}
    assert labels.get("nom") == "Didi Elloumi"


def test_transcribe_band_rows_rangee_vide_relue_par_grille():
    """Une rangée retournée vide par la grille VLM est récupérée (mono-ligne)."""
    from scriptvault.image_processing import _transcribe_band_rows

    def recognize_crop(crop: np.ndarray) -> list[dict]:
        return []

    def grid_reader(
        grid: np.ndarray, first: int, last: int
    ) -> list[tuple[int, str, float]]:
        if first == last:
            return [(1, "Concours : Physique & Chimie", 0.91)]
        return [
            (first, "Nom : ...Didi Elloumi...", 0.95),
            (first + 1, "Concours :", 0.6),
            (first + 2, "Établissement d'origine : ISSATSO", 0.88),
        ]

    image = _fake_form_page()
    out = _transcribe_band_rows(
        image, _three_bands(image), recognize_crop, None, band_grid_reader=grid_reader
    )
    labels = {item["label"]: item["text"] for item in out}
    assert labels.get("concours") == "Concours : Physique & Chimie"


def test_transcribe_band_rows_pas_de_relecture_si_lecture_complete():
    """Des rangées complètes ne déclenchent aucune relecture superflue."""
    from scriptvault.image_processing import _transcribe_band_rows

    def recognize_crop(crop: np.ndarray) -> list[dict]:
        return []

    def grid_reader(
        grid: np.ndarray, first: int, last: int
    ) -> list[tuple[int, str, float]]:
        return [
            (first, "Nom : ...Didi Elloumi...", 0.9),
            (first + 1, "Prénom : ...Salma...", 0.9),
            (first + 2, "Établissement d'origine : ISSATSO", 0.9),
        ]

    reads: list[str] = []

    def handwritten_reader(crop: np.ndarray, field: str) -> tuple[str, float]:
        reads.append(field)
        return ("XXX", 0.99)

    image = _fake_form_page()
    out = _transcribe_band_rows(
        image,
        _three_bands(image),
        recognize_crop,
        None,
        band_grid_reader=grid_reader,
        handwritten_reader=handwritten_reader,
        handwritten_fields=("nom", "prenom", "etablissement"),
    )
    labels = {item["label"]: item["text"] for item in out}
    assert labels.get("nom") == "Nom : ...Didi Elloumi..."
    assert labels.get("prenom") == "Prénom : ...Salma..."
    assert reads == []


def test_transcribe_band_rows_relecture_valeur_seule_ecartee():
    """Une relecture « libellé seul » (sans valeur) est écartée, pas gardée."""
    from scriptvault.image_processing import _transcribe_band_rows

    def recognize_crop(crop: np.ndarray) -> list[dict]:
        return []

    def grid_reader(
        grid: np.ndarray, first: int, last: int
    ) -> list[tuple[int, str, float]]:
        if first == last:
            return [(1, "Prénom :", 0.6)]
        return [
            (first, "Nom : ...Didi Elloumi...", 0.95),
            (first + 1, "Prénom :", 0.6),
            (first + 2, "Établissement d'origine : ISSATSO", 0.88),
        ]

    image = _fake_form_page()
    out = _transcribe_band_rows(
        image, _three_bands(image), recognize_crop, None, band_grid_reader=grid_reader
    )
    labels = {item["label"]: item["text"] for item in out}
    assert labels.get("nom") == "Nom : ...Didi Elloumi..."
    assert labels.get("etablissement") == "Établissement d'origine : ISSATSO"
    assert "prenom" not in labels


# --------------------------------------------------------------------------- #
# Pré-chargement du modèle + options de performance (num_ctx / num_gpu)
# --------------------------------------------------------------------------- #
def test_chat_payload_inclut_options_perf():
    reader, client = _reader([_vlm_response('{"text": "x", "confidence": 0.9}')])
    try:
        asyncio.run(reader._chat([{"role": "user", "content": "bonjour"}]))
        payload = client.posts[0][1]["json"]
        options = payload["options"]
        assert options["num_ctx"] == 8192
        assert options["num_gpu"] == 99
        assert payload["keep_alive"] == "30m"
    finally:
        reader.close()


def test_chat_payload_options_desactivables():
    client = FakeClient([_vlm_response('{"text": "x", "confidence": 0.9}')])
    config = VLMConfig(num_ctx=None, num_gpu=None, keep_alive="")
    reader = LocalVLMReader(config=config, client_factory=lambda: client)
    try:
        asyncio.run(reader._chat([{"role": "user", "content": "bonjour"}]))
        payload = client.posts[0][1]["json"]
        assert "num_ctx" not in payload["options"]
        assert "num_gpu" not in payload["options"]
        assert "keep_alive" not in payload
    finally:
        reader.close()


def test_warm_up_charge_le_modele():
    reader, _ = _reader([_vlm_response('{"text": "ok"}')])
    try:
        assert reader.warm_up() is True
    finally:
        reader.close()


def test_warm_up_echoue_sans_serveur():
    reader = LocalVLMReader(
        config=VLMConfig(timeout_s=0.5),
        client_factory=lambda: FakeClient([RuntimeError("connexion refusée")]),
    )
    try:
        assert reader.warm_up() is False
    finally:
        reader.close()

