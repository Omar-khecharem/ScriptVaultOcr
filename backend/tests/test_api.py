"""Tests de l'API REST (FastAPI TestClient) avec un moteur OCR factice.

Le moteur PaddlePaddle n'est pas requis : une fabrique factice est injectée
dans :func:`scriptvault.api.app.create_app`, ce qui permet de tester le
contrat HTTP complet (validation, formats, erreurs) en isolation.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scriptvault.api.app import create_app
from scriptvault.config import Settings


def _fake_engine_factory():
    """Moteur factice : retourne une ligne fixe, sans PaddlePaddle."""

    class FakeEngine:
        cpu_threads = 4
        is_ready = True

        def predict_array(self, image, *, preprocess=True):
            assert isinstance(image, np.ndarray)
            return [
                {
                    "text": "Bonjour",
                    "confidence": 0.99,
                    "box": [[0, 0], [50, 0], [50, 10], [0, 10]],
                },
                {
                    "text": "Monde",
                    "confidence": 0.85,
                    "box": [[0, 20], [40, 20], [40, 30], [0, 30]],
                },
            ]

        predict_bytes = predict_array

        def close(self):
            pass

    return FakeEngine()


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        preload=False,
        max_concurrency=1,
        timeout_ms=5000,
        max_file_mb=5,
        preprocess=True,
    )


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    app = create_app(settings=settings, engine_factory=_fake_engine_factory)
    with TestClient(app) as test_client:
        yield test_client


def _sample_image(tmp_path: Path, name: str = "scan.png") -> Path:
    image = np.full((300, 800, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Texte", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    path = tmp_path / name
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    return path


# --------------------------------------------------------------------------- #
# Racine & santé
# --------------------------------------------------------------------------- #
def test_root_metadata(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ScriptVault OCR API"
    assert data["docs"] == "/docs"


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["lang"] == "en"
    assert data["max_concurrency"] == 1
    assert data["preloading"] is False
    assert data["corrections"]["enabled"] is True
    assert data["corrections"]["lexicon"] is False


# --------------------------------------------------------------------------- #
# OCR fichier unique
# --------------------------------------------------------------------------- #
def test_ocr_single_image(client: TestClient, tmp_path: Path):
    path = _sample_image(tmp_path)
    with path.open("rb") as handle:
        response = client.post(
            "/api/ocr/single",
            files={"file": ("scan.png", handle, "image/png")},
            data={"preview": "true"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["file"] == "scan.png"
    assert len(data["pages"]) == 1
    page = data["pages"][0]
    assert page["page"] == 1
    assert page["width"] > 0 and page["height"] > 0
    assert page["text"] == "Bonjour\nMonde"
    assert page["confidence"] == pytest.approx(0.92, abs=0.01)
    assert page["items"][0]["box"][0] == [0, 0]
    assert page["preview"].startswith("data:image/png;base64,")
    assert data["confidence"] == pytest.approx(0.92, abs=0.01)


def test_ocr_single_without_preview(client: TestClient, tmp_path: Path):
    path = _sample_image(tmp_path)
    with path.open("rb") as handle:
        response = client.post(
            "/api/ocr/single",
            files={"file": ("scan.png", handle, "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["pages"][0]["preview"] is None


def test_ocr_single_rejects_unsupported_format(client: TestClient):
    response = client.post(
        "/api/ocr/single",
        files={"file": ("doc.txt", b"pas une image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Format non supporté" in response.json()["detail"]


def test_ocr_single_rejects_oversized_file(client: TestClient):
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * (6 * 1024 * 1024)
    response = client.post(
        "/api/ocr/single",
        files={"file": ("big.png", payload, "image/png")},
    )
    assert response.status_code == 400
    assert "trop volumineux" in response.json()["detail"]


def test_ocr_single_rejects_empty_file(client: TestClient):
    response = client.post(
        "/api/ocr/single",
        files={"file": ("vide.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "vide" in response.json()["detail"]


def test_ocr_single_rejects_corrupted_image(client: TestClient):
    response = client.post(
        "/api/ocr/single",
        files={"file": ("corrompu.png", b"pas du png", "image/png")},
    )
    assert response.status_code == 400


def test_ocr_single_accepts_pdf(client: TestClient, tmp_path: Path):
    """Un PDF minimal (une page) doit être rastérisé puis reconnu."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF non installé")
    document = fitz.open()
    page = document.new_page(width=300, height=300)
    page.insert_text((30, 60), "Hello", fontsize=24)
    pdf_bytes = document.tobytes()
    document.close()
    response = client.post(
        "/api/ocr/single",
        files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["pages"]) >= 1


# --------------------------------------------------------------------------- #
# OCR par lots
# --------------------------------------------------------------------------- #
def test_ocr_batch_mixed(client: TestClient, tmp_path: Path):
    good = _sample_image(tmp_path, "ok.png")
    with good.open("rb") as handle:
        response = client.post(
            "/api/ocr/batch",
            files=[
                ("files", ("ok.png", handle, "image/png")),
                ("files", ("mauvais.png", b"nope", "image/png")),
            ],
        )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["ok"] == 1
    assert data["errors"] == 1
    statuses = {item["file"]: item["status"] for item in data["items"]}
    assert statuses == {"ok.png": "ok", "mauvais.png": "error"}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_export_txt(client: TestClient):
    response = client.post(
        "/api/export", json={"format": "txt", "text": "Bonjour le monde"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "attachment" in response.headers["content-disposition"]
    assert response.content == "Bonjour le monde".encode("utf-8")


def test_export_pdf(client: TestClient):
    response = client.post(
        "/api/export", json={"format": "pdf", "text": "Couche texte"}
    )
    assert response.status_code == 200
    assert response.content[:4] == b"%PDF"


def test_export_docx(client: TestClient):
    response = client.post(
        "/api/export", json={"format": "docx", "text": "Document Word"}
    )
    assert response.status_code == 200
    assert response.content[:2] == b"PK"


def test_export_empty_text_rejected(client: TestClient):
    response = client.post("/api/export", json={"format": "txt", "text": "   "})
    assert response.status_code == 400


def test_export_invalid_format_rejected(client: TestClient):
    response = client.post("/api/export", json={"format": "html", "text": "x"})
    assert response.status_code == 422


def test_export_filename_sanitized(client: TestClient):
    """La pièce jointe ne doit jamais contenir de séparateurs de chemin."""
    response = client.post(
        "/api/export",
        json={"format": "txt", "text": "x", "filename": "../../etc/passwd"},
    )
    disposition = response.headers["content-disposition"]
    assert "/" not in disposition and "\\" not in disposition
    assert "attachment" in disposition
    assert ".txt" in disposition


# --------------------------------------------------------------------------- #
# Analyse de formulaire (post-traitement)
# --------------------------------------------------------------------------- #
def test_form_analyze_valid_document(client: TestClient):
    """Items OCR réels -> formulaire structuré, champs validés."""
    response = client.post(
        "/api/form/analyze",
        json={
            "file_name": "scan.png",
            "items": [
                {
                    "text": "Nom :",
                    "confidence": 0.98,
                    "box": [[50, 100], [260, 100], [260, 134], [50, 134]],
                },
                {
                    "text": "Didi",
                    "confidence": 0.96,
                    "box": [[280, 100], [520, 100], [520, 134], [280, 134]],
                },
                {
                    "text": "N° CIN :",
                    "confidence": 0.97,
                    "box": [[50, 160], [260, 160], [260, 194], [50, 194]],
                },
                {
                    "text": "09728320",
                    "confidence": 0.95,
                    "box": [[280, 160], [520, 160], [520, 194], [280, 194]],
                },
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_name"] == "scan.png"
    assert data["is_form"] is True
    assert data["processing_time_ms"] < 30.0
    fields = {field["key"]: field for field in data["fields"]}
    assert fields["nom"]["value"] == "Didi"
    assert fields["nom"]["status"] == "valid"
    assert fields["cin"]["status"] == "valid"


def test_form_analyze_flags_errors(client: TestClient):
    """CIN illisible (lettres non-corrigeables) + incohérence Série/Identifiant."""
    rows = [
        ("N° CIN :", "K97283ZK"),
        ("Série :", "514"),
        ("Identifiant :", "615001"),
    ]
    items = []
    for row, (label, value) in enumerate(rows):
        y0 = 100 + row * 60
        items.append(
            {
                "text": label,
                "confidence": 0.98,
                "box": [[50, y0], [260, y0], [260, y0 + 34], [50, y0 + 34]],
            }
        )
        items.append(
            {
                "text": value,
                "confidence": 0.90,
                "box": [[280, y0], [520, y0], [520, y0 + 34], [280, y0 + 34]],
            }
        )
    response = client.post(
        "/api/form/analyze", json={"file_name": "scan.png", "items": items}
    )
    assert response.status_code == 200
    data = response.json()
    fields = {field["key"]: field for field in data["fields"]}
    assert fields["cin"]["status"] == "error"
    assert "lettres" in (fields["cin"]["error_message"] or "")
    assert fields["identifiant"]["status"] == "error"
    assert fields["serie"]["status"] == "error"


def test_form_analyze_rejects_empty_items(client: TestClient):
    response = client.post(
        "/api/form/analyze", json={"file_name": "x.png", "items": []}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_form"] is False
    assert data["fields"] == []


def test_form_analyze_rejects_invalid_payload(client: TestClient):
    response = client.post(
        "/api/form/analyze",
        json={"file_name": "x.png", "items": [{"text": 123, "confidence": "oops"}]},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Timeout d'inférence
# --------------------------------------------------------------------------- #
def test_ocr_timeout_maps_to_504(tmp_path: Path):
    """Un moteur qui dépasse le délai doit produire un 504 Gateway Timeout."""

    class SlowEngine:
        is_ready = True

        def predict_array(self, image, *, preprocess=True):
            time.sleep(1.0)
            return []

        predict_bytes = predict_array

        def close(self):
            pass

    app = create_app(
        settings=Settings(preload=False, timeout_ms=100, max_concurrency=1),
        engine_factory=lambda: SlowEngine(),
    )
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    with TestClient(app) as slow_client:
        response = slow_client.post(
            "/api/ocr/single",
            files={"file": ("t.png", buffer.tobytes(), "image/png")},
        )
    assert response.status_code == 504
