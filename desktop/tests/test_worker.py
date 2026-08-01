"""Tests unitaires du worker de traitement par lots (PySide6)."""

import pytest
import worker_thread
from worker_thread import BatchWorker, CancellationToken, CancelledError


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
