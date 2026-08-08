"""Tests du workflow de traitement par lots (API /api/batches).

Le moteur PaddlePaddle n'est pas requis : une fabrique factice est injectée.
Les lots sont traités en tâche de fond — les tests sondent la progression
jusqu'à épuisement (le moteur factice est quasi instantané).
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
    """Moteur factice : retourne deux lignes fixes, sans PaddlePaddle."""

    class FakeEngine:
        cpu_threads = 4
        is_ready = True

        def predict_array(self, image, *, preprocess=True):
            assert isinstance(image, np.ndarray)
            return [
                {
                    "text": "Nom DUPONT",
                    "confidence": 0.99,
                    "box": [[0, 0], [50, 0], [50, 10], [0, 10]],
                },
                {
                    "text": "Prénom Jean",
                    "confidence": 0.85,
                    "box": [[0, 20], [40, 20], [40, 30], [0, 30]],
                },
            ]

        predict_bytes = predict_array

        def close(self):
            pass

    return FakeEngine()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        preload=False,
        max_concurrency=1,
        timeout_ms=5000,
        max_file_mb=5,
        preprocess=True,
        storage_root=str(tmp_path / "storage"),
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


def _wait_done(client: TestClient, job_id: str, timeout: float = 8.0) -> dict:
    """Sonde le lot jusqu'à son terme (done / cancelled / error)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/batches/{job_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] in ("done", "cancelled", "error"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"Lot {job_id} toujours en cours après {timeout}s.")


def _create_batch(
    client: TestClient, images: list[Path], name: str = "Lot test"
) -> dict:
    files = [
        ("files", (image.name, image.read_bytes(), "image/png")) for image in images
    ]
    response = client.post("/api/batches", files=files, data={"name": name})
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# Création & progression
# --------------------------------------------------------------------------- #
def test_create_batch_returns_job_summary(client: TestClient, tmp_path: Path):
    image = _sample_image(tmp_path)
    payload = _create_batch(client, [image])
    job = payload["job"]
    assert job["name"] == "Lot test"
    assert job["counts"]["total"] == 1
    assert job["status"] in ("pending", "processing", "done")

    summary = _wait_done(client, job["id"])
    assert summary["status"] == "done"
    assert summary["counts"]["done"] == 1
    assert summary["counts"]["error"] == 0
    assert summary["avg_confidence"] > 0.0


def test_batch_files_paginated_and_filtered(client: TestClient, tmp_path: Path):
    images = [_sample_image(tmp_path, f"scan_{i}.png") for i in range(3)]
    job = _create_batch(client, images)["job"]
    _wait_done(client, job["id"])

    response = client.get(
        f"/api/batches/{job['id']}/files", params={"page": 1, "page_size": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["items"]) == 2
    assert all(item["status"] == "done" for item in data["items"])

    page_two = client.get(
        f"/api/batches/{job['id']}/files", params={"page": 2, "page_size": 2}
    ).json()
    assert len(page_two["items"]) == 1

    search = client.get(
        f"/api/batches/{job['id']}/files", params={"q": "scan_2"}
    ).json()
    assert search["total"] == 1
    assert search["items"][0]["name"] == "scan_2.png"


def test_file_detail_contains_pages_and_form(client: TestClient, tmp_path: Path):
    image = _sample_image(tmp_path)
    job = _create_batch(client, [image])["job"]
    _wait_done(client, job["id"])

    listing = client.get(f"/api/batches/{job['id']}/files").json()
    file_id = listing["items"][0]["id"]

    detail = client.get(f"/api/batches/{job['id']}/files/{file_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["status"] == "done"
    assert payload["pages"], "au moins une page OCR"
    page = payload["pages"][0]
    assert page["page"] == 1
    assert page["width"] > 0 and page["height"] > 0
    assert page["text"]
    assert "form" in page
    assert page["form"]["fields"] is not None


def test_file_preview_returns_png_data_url(client: TestClient, tmp_path: Path):
    image = _sample_image(tmp_path)
    job = _create_batch(client, [image])["job"]
    _wait_done(client, job["id"])
    file_id = client.get(f"/api/batches/{job['id']}/files").json()["items"][0]["id"]

    response = client.get(f"/api/batches/{job['id']}/files/{file_id}/preview")
    assert response.status_code == 200
    preview = response.json()["preview"]
    assert preview.startswith("data:image/jpeg;base64,")

    missing = client.get(
        f"/api/batches/{job['id']}/files/{file_id}/preview", params={"page": 99}
    )
    assert missing.status_code == 404


def test_preview_persisted_once_on_disk(client: TestClient, tmp_path: Path):
    """Les aperçus JPEG sont écrits au moment de l'analyse et lus ensuite du
    disque : aucun re-prétraitement à chaque consultation."""
    image = _sample_image(tmp_path)
    job = _create_batch(client, [image])["job"]
    _wait_done(client, job["id"])

    file_id = client.get(f"/api/batches/{job['id']}/files").json()["items"][0]["id"]
    first = client.get(f"/api/batches/{job['id']}/files/{file_id}/preview")
    assert first.status_code == 200
    assert first.json()["preview"].startswith("data:image/jpeg;base64,")

    previews_dir = tmp_path / "storage" / "jobs" / job["id"] / "previews"
    assert previews_dir.is_dir()
    assert list(previews_dir.glob("*.jpg")), "l'aperçu doit être persisté sur disque"


def test_batch_export_excel(client: TestClient, tmp_path: Path):
    image = _sample_image(tmp_path)
    job = _create_batch(client, [image])["job"]
    _wait_done(client, job["id"])

    response = client.get(f"/api/batches/{job['id']}/export.xlsx")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert response.content.startswith(b"PK")


def test_cancel_and_delete_batch(client: TestClient, tmp_path: Path):
    image = _sample_image(tmp_path)
    job = _create_batch(client, [image])["job"]

    cancelled = client.post(f"/api/batches/{job['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["cancelled"] is True

    deleted = client.delete(f"/api/batches/{job['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == job["id"]

    assert client.get(f"/api/batches/{job['id']}").status_code == 404


def test_history_lists_jobs(client: TestClient, tmp_path: Path):
    image = _sample_image(tmp_path)
    first = _create_batch(client, [image], "Lot A")["job"]
    second = _create_batch(client, [image], "Lot B")["job"]
    jobs = client.get("/api/batches").json()["jobs"]
    ids = [entry["id"] for entry in jobs]
    assert first["id"] in ids and second["id"] in ids


# --------------------------------------------------------------------------- #
# Validation des entrées
# --------------------------------------------------------------------------- #
def test_create_batch_rejects_empty(client: TestClient):
    response = client.post("/api/batches", files=[])
    assert response.status_code in (400, 422)


def test_create_batch_rejects_all_invalid_files(client: TestClient, tmp_path: Path):
    bad = tmp_path / "note.txt"
    bad.write_text("pas une image")
    response = client.post(
        "/api/batches",
        files=[("files", ("note.txt", bad.read_bytes(), "text/plain"))],
    )
    assert response.status_code == 400


def test_create_batch_isolates_invalid_files(client: TestClient, tmp_path: Path):
    image = _sample_image(tmp_path)
    bad = tmp_path / "note.txt"
    bad.write_text("pas une image")
    response = client.post(
        "/api/batches",
        files=[
            ("files", ("scan.png", image.read_bytes(), "image/png")),
            ("files", ("note.txt", bad.read_bytes(), "text/plain")),
        ],
    )
    assert response.status_code == 201
    payload = response.json()
    assert len(payload["rejected"]) == 1
    assert payload["job"]["counts"]["total"] == 1


def test_unknown_job_returns_404(client: TestClient):
    assert client.get("/api/batches/nope").status_code == 404
    assert client.get("/api/batches/nope/files").status_code == 404
    assert client.post("/api/batches/nope/cancel").status_code == 404
    assert client.delete("/api/batches/nope").status_code == 404
    assert client.get("/api/batches/nope/export.xlsx").status_code == 404
