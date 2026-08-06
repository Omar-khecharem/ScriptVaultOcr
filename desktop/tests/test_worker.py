"""Tests unitaires du worker de traitement par lots (PySide6)."""

import os
import time

import pytest
import worker_thread
from worker_thread import BatchWorker, CancellationToken, CancelledError

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeEngine:
    """Moteur factice: compte les appels, renvoie une ligne statique."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, path):
        self.calls += 1
        return [{"text": "ok", "confidence": 1.0}]


def _wait_until(worker: BatchWorker, timeout: float = 15.0) -> None:
    """Attend la fin du thread en pompant les événements Qt."""
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])
    deadline = time.monotonic() + timeout
    while worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def test_version():
    assert worker_thread.__version__ == "1.0.0"


def test_cancellation_token_lifecycle():
    token = CancellationToken()
    assert not token.is_cancelled
    token.cancel()
    assert token.is_cancelled
    with pytest.raises(CancelledError):
        token.check()
    token.reset()
    assert not token.is_cancelled
    token.check()  # ne doit pas lever


def test_mean_confidence():
    assert BatchWorker._mean_confidence([]) == 0.0
    assert BatchWorker._mean_confidence(
        [{"confidence": 0.5}, {"confidence": 1.0}]
    ) == pytest.approx(0.75)
    assert BatchWorker._mean_confidence([{"text": "sans confiance"}]) == 0.0


def test_submit_immediately_after_start_is_not_lost(tmp_path):
    """Régression: un submit() lancé juste après start() ne doit pas être
    perdu (l'ancienne boucle pouvait se terminer avant la soumission)."""
    engine = _FakeEngine()
    worker = BatchWorker(engine, batch_size=2, gc_after_each=False)
    paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    worker.start()
    worker.submit(paths)
    _wait_until(worker)
    assert not worker.isRunning()
    assert worker.state == "finished"
    assert engine.calls == 2
    assert worker.processed == 2
    worker.shutdown()
