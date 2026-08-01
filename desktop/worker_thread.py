"""Worker multithread de traitement OCR par lots (PySide6 + core_ocr).

Ce module fournit un worker basé sur ``QThread`` qui consomme une file
d'attente bornée de chemins d'images et produit, en temps réel, des signaux
Qt : progression, résultats et erreurs.

Modèle de concurrence:

* L'inférence PaddleOCR n'est **pas thread-safe** : une seule instance de
  :class:`LocalOCREngine` est utilisée, exclusivement depuis le thread du
  worker. C'est pourquoi le module repose sur ``QThread`` plutôt qu'un pool
  de threads concurrents.
* La file d'attente est **bornée** (``queue.Queue(maxsize=...)``) : le
  producteur (GUI ou CLI) ne peut pas inonder la RAM, la saturation
  rétro-agissant sur l'appel ``submit()`` (backpressure).
* Traitement par **lots** : les éléments sont consommés par paquets de
  ``batch_size``, chaque image étant libérée immédiatement après traitement.
* **Mémoire** : ``gc.collect()`` est déclenché après chaque image (désactivable
  via ``gc_after_each=False``) et le buffer de résultats est borné.
* **Annulation** : jeton d'annulation coopératif et thread-safe
  (:class:`CancellationToken`), vérifié entre chaque élément. Une inférence
  en cours n'est pas interrompue en plein appel bloquant (limite inhérente à
  PaddleOCR), mais le traitement s'arrête au plus vite.

.. warning::
    **Garde la référence du worker vivante** tant que le thread est en cours
    d'exécution. Si l'objet Python est détruit (fin de scope, GC) alors que
    le QThread tourne encore, Qt appelle ``abort()`` et le processus se
    termine (code ``0xC0000409`` sur Windows). Conservez le worker dans une
    variable de portée large, ou avec l'application comme parent Qt.

Exemple d'utilisation::

    from core_ocr import LocalOCREngine
    from worker_thread import BatchWorker

    engine = LocalOCREngine(lang="fr")
    worker = BatchWorker(engine, batch_size=4)
    # Connecter TOUS les slots avant start() pour ne manquer aucun signal.
    worker.progress_updated.connect(lambda n: print("progression:", n))
    worker.result_ready.connect(lambda r: print(r["text"]))
    worker.error_occurred.connect(lambda e: print("ERREUR:", e))
    worker.batch_finished.connect(app.quit)
    worker.start()
    worker.submit(["scan_1.png", "scan_2.png", "scan_3.png"])
    app.exec()
    worker.shutdown()
    engine.close()
"""

from __future__ import annotations

import gc
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Deque, Optional, Sequence

try:
    from PySide6.QtCore import QThread, Signal
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PySide6 est requis. Installez-le avec: pip install PySide6"
    ) from exc

from scriptvault.core_ocr import OCRBaseError

__version__ = "1.0.0"
__all__ = [
    "BatchWorker",
    "CancellationToken",
    "CancelledError",
]

PathLike = str | os.PathLike[str]
Record = dict[str, Any]


# --------------------------------------------------------------------------- #
# Jeton d'annulation
# --------------------------------------------------------------------------- #
class CancelledError(Exception):
    """Levée lorsqu'une opération est annulée via le jeton d'annulation."""


class CancellationToken:
    """Jeton d'annulation coopératif, thread-safe.

    Mécanisme standard de signalisation d'annulation : chaque thread observant
    le jeton vérifie ``is_cancelled`` (ou appelle :meth:`check`) à des points
    de synchronisation sûrs, puis abandonne proprement. Aucune ressource n'est
    conservée par le jeton ; le seul état est un ``threading.Event``.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Demande l'annulation de toutes les opérations observatrices."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Vrai si l'annulation a été demandée."""
        return self._event.is_set()

    def check(self) -> None:
        """Lève :class:`CancelledError` si l'annulation a été demandée.

        Raises:
            CancelledError: Si le jeton est annulé.
        """
        if self._event.is_set():
            raise CancelledError("Opération annulée.")

    def reset(self) -> None:
        """Réarme le jeton (uniquement hors traitement, cf. ``reset_for_batch``)."""
        with self._lock:
            self._event.clear()


class _Stats:
    """Compteurs du worker, protégés par un verrou (multi-thread)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total: int = 0
        self.pending: int = 0
        self.processed: int = 0
        self.failed: int = 0
        self.batch_done: int = 0


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class BatchWorker(QThread):
    """Worker multithread de traitement OCR par lots.

    Consomme une file bornée de chemins d'images, les traite via
    :class:`core_ocr.LocalOCREngine` par paquets de ``batch_size`` et émet
    des signaux Qt en temps réel.

    Signaux émis (depuis le thread du worker, livrés dans le thread de
    l'objet — typiquement le thread GUI — grâce aux connexions queued) :

    * ``progress_updated(int)`` : nombre cumulé d'images traitées avec succès.
    * ``result_ready(dict)`` : un résultat par image, au format::

        {
            "index": int,
            "path": str,
            "status": "ok",
            "elapsed_ms": float,
            "text": str,
            "confidence": float,
            "results": [{"text", "confidence", "box"}, ...],
        }

    * ``error_occurred(str)`` : message d'erreur (l'image en échec est
      ignorée, le traitement continue).
    * ``batch_processed(int)`` : nombre de lots terminés.
    * ``batch_finished(int)`` : total d'images traitées, à la fin du run.
    * ``state_changed(str)`` : ``"idle"``, ``"running"``, ``"finished"``,
      ``"cancelled"`` ou ``"failed"``.
    """

    progress_updated = Signal(int)
    result_ready = Signal(dict)
    error_occurred = Signal(str)
    batch_processed = Signal(int)
    batch_finished = Signal(int)
    state_changed = Signal(str)

    def __init__(
        self,
        engine: Any,
        batch_size: int = 4,
        queue_maxsize: int = 64,
        buffer_size: int = 200,
        gc_after_each: bool = True,
        parent: Optional[Any] = None,
    ) -> None:
        """Initialise le worker.

        Args:
            engine: Instance de :class:`core_ocr.LocalOCREngine`. Elle ne
                doit être utilisée par aucun autre thread.
            batch_size: Nombre d'images traitées par lot (>= 1).
            queue_maxsize: Capacité maximale de la file d'attente. Une valeur
                basse limite l'empreinte RAM du producteur (backpressure).
            buffer_size: Nombre maximal de résultats conservés en mémoire
                (``0`` désactive le buffer). Au-delà, les plus anciens sont
                évincés.
            gc_after_each: Exécute ``gc.collect()`` après chaque image
                traitée pour limiter la pression mémoire.
            parent: Objet parent Qt optionnel.
        """
        super().__init__(parent)
        if batch_size < 1:
            raise ValueError(f"batch_size doit être >= 1, reçu {batch_size}.")
        if queue_maxsize < 1:
            raise ValueError(f"queue_maxsize doit être >= 1, reçu {queue_maxsize}.")

        self._engine = engine
        self._batch_size = int(batch_size)
        self._gc_after_each = bool(gc_after_each)
        self._queue: queue.Queue[PathLike] = queue.Queue(maxsize=int(queue_maxsize))
        self._token = CancellationToken()
        self._stats = _Stats()
        self._results: Deque[Record] = deque(maxlen=max(0, int(buffer_size)))
        self._state = "idle"
        self._next_index = 0

    # ------------------------------------------------------------------ #
    # API producteur (thread principal)
    # ------------------------------------------------------------------ #
    def submit(self, paths: Sequence[PathLike]) -> int:
        """Ajoute des chemins d'images à la file d'attente.

        Le put est borné (timeout de 0,5 s par tentative) : si la file est
        pleine et qu'aucun thread consommateur n'est actif, une
        :class:`RuntimeError` est levée au lieu de bloquer indéfiniment
        (évite tout deadlock producteur/consommateur).

        Args:
            paths: Séquence de chemins d'images à traiter.

        Returns:
            Le nombre d'éléments acceptés dans la file.

        Raises:
            RuntimeError: Si le worker est annulé/arrêté, ou si la file est
                pleine alors que le worker n'est pas en cours d'exécution
                (appeler :meth:`start` avant de soumettre).
        """
        if self._token.is_cancelled:
            raise RuntimeError(
                "Impossible de soumettre des images: le worker est annulé ou arrêté."
            )
        accepted = 0
        for path in paths:
            item = os.fspath(path)
            while True:
                try:
                    self._queue.put(item, timeout=0.5)
                    break
                except queue.Full:
                    if not self.isRunning():
                        raise RuntimeError(
                            "File d'attente pleine et worker non actif: "
                            "démarrez le worker (start()) avant de soumettre "
                            f"plus de {self._queue.maxsize} éléments."
                        ) from None
            accepted += 1
        with self._stats.lock:
            self._stats.total += accepted
            self._stats.pending += accepted
        return accepted

    def submit_one(self, path: PathLike) -> int:
        """Ajoute une image unique à la file (voir :meth:`submit`)."""
        return self.submit([path])

    def cancel(self) -> None:
        """Demande l'annulation immédiate et coopérative du traitement.

        Le worker s'arrête au plus vite (avant l'élément suivant). Appeler
        ensuite :meth:`shutdown` pour attendre la fin du thread et libérer
        la mémoire.
        """
        self._token.cancel()

    def shutdown(self, timeout_ms: int = 10000) -> None:
        """Annule, attend la fin du thread et libère les ressources.

        Args:
            timeout_ms: Délai maximal d'attente de fin du thread (ms).
        """
        self._token.cancel()
        if self.isRunning():
            self.wait(max(0, int(timeout_ms)))
        self._clear_queue()
        if self.isFinished():
            self._queue = queue.Queue(maxsize=self._queue.maxsize)
            self._results.clear()
        self._set_state("stopped")
        gc.collect()

    def reset_for_batch(self) -> None:
        """Réinitialise le worker pour une nouvelle série de traitements.

        .. note::
            Les événements de signal déjà postés par la série précédente
            (avant le reset) peuvent encore être livrés ensuite. Si les slots
            ne distinguent pas les séries (ex. ``batch_finished``), filtrez
            les événements obsolètes (identifiant de lot, compteur, ...).

        Raises:
            RuntimeError: Si le thread est encore en cours d'exécution.
        """
        if self.isRunning():
            raise RuntimeError(
                "Impossible de réinitialiser un worker en cours d'exécution."
            )
        with self._stats.lock:
            self._stats = _Stats()
        self._queue = queue.Queue(maxsize=self._queue.maxsize)
        self._results.clear()
        self._token.reset()
        self._next_index = 0
        self._set_state("idle")

    # ------------------------------------------------------------------ #
    # Propriétés
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        """État courant: ``idle``, ``running``, ``finished``, ``cancelled``,
        ``failed`` ou ``stopped``."""
        return self._state

    @property
    def cancellation_token(self) -> CancellationToken:
        """Le jeton d'annulation associé au worker."""
        return self._token

    @property
    def total_submitted(self) -> int:
        """Nombre total d'images soumises depuis le dernier reset."""
        with self._stats.lock:
            return self._stats.total

    @property
    def processed(self) -> int:
        """Nombre d'images traitées avec succès."""
        with self._stats.lock:
            return self._stats.processed

    @property
    def failed(self) -> int:
        """Nombre d'images en échec."""
        with self._stats.lock:
            return self._stats.failed

    @property
    def pending(self) -> int:
        """Nombre d'images encore en attente dans la file."""
        with self._stats.lock:
            return self._stats.pending

    @property
    def batch_count(self) -> int:
        """Nombre de lots terminés."""
        with self._stats.lock:
            return self._stats.batch_done

    @property
    def is_busy(self) -> bool:
        """Vrai si le worker est en cours d'exécution ou a des éléments
        en attente."""
        with self._stats.lock:
            return self.isRunning() or self._stats.pending > 0

    def get_results(self) -> list[Record]:
        """Copie des résultats conservés dans le buffer (borné)."""
        with self._stats.lock:
            return list(self._results)

    # ------------------------------------------------------------------ #
    # Implémentation du thread
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Boucle principale: draine la file par lots et émet les signaux."""
        self._set_state("running")
        try:
            while not self._token.is_cancelled:
                with self._stats.lock:
                    drained = self._queue.empty() and self._stats.pending == 0
                if drained:
                    break
                batch = self._drain_batch()
                if not batch:
                    time.sleep(0.02)
                    continue
                for path, index in batch:
                    if self._token.is_cancelled:
                        break
                    self._process_item(path, index)
                with self._stats.lock:
                    self._stats.batch_done += 1
                    done = self._stats.batch_done
                self.batch_processed.emit(done)
        except Exception as exc:  # garde-fou global du thread
            self.error_occurred.emit(f"Erreur fatale du worker: {exc}")
            self._set_state("failed")
        else:
            self._set_state("cancelled" if self._token.is_cancelled else "finished")
        finally:
            with self._stats.lock:
                final = self._stats.processed
            self.batch_finished.emit(final)
            if self._gc_after_each:
                gc.collect()

    def _drain_batch(self) -> list[tuple[PathLike, int]]:
        """Prélève jusqu'à ``batch_size`` éléments dans la file."""
        batch: list[tuple[PathLike, int]] = []
        deadline = time.monotonic() + 0.1
        while len(batch) < self._batch_size and time.monotonic() < deadline:
            try:
                path = self._queue.get(timeout=0.05)
            except queue.Empty:
                break
            self._queue.task_done()
            with self._stats.lock:
                self._stats.pending -= 1
                index = self._next_index
                self._next_index += 1
            batch.append((path, index))
        return batch

    def _process_item(self, path: PathLike, index: int) -> None:
        """Traite une image: inférence, signal de résultat, libération mémoire."""
        started = time.perf_counter()
        try:
            items = self._engine.predict(path)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record: Record = {
                "index": index,
                "path": os.fspath(path),
                "status": "ok",
                "elapsed_ms": round(elapsed_ms, 2),
                "text": " ".join(item.get("text", "") for item in items).strip(),
                "confidence": self._mean_confidence(items),
                "results": items,
            }
            self._store_result(record)
            self.result_ready.emit(record)
            with self._stats.lock:
                self._stats.processed += 1
                processed = self._stats.processed
            self.progress_updated.emit(processed)
        except OCRBaseError as exc:
            with self._stats.lock:
                self._stats.failed += 1
            self.error_occurred.emit(f"[#{index}] {path}: {exc}")
        except Exception as exc:
            with self._stats.lock:
                self._stats.failed += 1
            self.error_occurred.emit(
                f"[#{index}] {path}: erreur inattendue ({type(exc).__name__}): {exc}"
            )
        finally:
            if self._gc_after_each:
                gc.collect()

    def _store_result(self, record: Record) -> None:
        """Conserve le résultat dans le buffer borné (éviction FIFO)."""
        with self._stats.lock:
            self._results.append(record)

    @staticmethod
    def _mean_confidence(items: Sequence[dict[str, Any]]) -> float:
        """Confiance moyenne des lignes détectées (0.0 si aucune)."""
        if not items:
            return 0.0
        total = 0.0
        count = 0
        for item in items:
            score = item.get("confidence")
            if isinstance(score, (int, float)):
                total += float(score)
                count += 1
        return round(total / count, 4) if count else 0.0

    def _set_state(self, state: str) -> None:
        """Met à jour l'état et émet ``state_changed``."""
        self._state = state
        self.state_changed.emit(state)

    def _clear_queue(self) -> int:
        """Vide la file des éléments restants (après annulation)."""
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            drained += 1
        with self._stats.lock:
            self._stats.pending = max(0, self._stats.pending - drained)
            self._stats.total -= drained
        return drained


# --------------------------------------------------------------------------- #
# CLI de démonstration
# --------------------------------------------------------------------------- #
def main() -> int:
    """Ligne de commande: ``python worker_thread.py img1.png img2.png ...``

    Exécute un traitement par lots complet avec progression en temps réel,
    annulation via Ctrl+C et affichage des résultats au format texte.
    """
    import argparse
    import signal
    import sys

    from PySide6.QtCore import QCoreApplication
    from scriptvault.core_ocr import LocalOCREngine

    parser = argparse.ArgumentParser(
        description="Traitement OCR par lots, multithread (PySide6)."
    )
    parser.add_argument("images", nargs="+", help="Images à traiter.")
    parser.add_argument("--lang", default="en", help="Langue des modèles OCR.")
    parser.add_argument(
        "--batch", type=int, default=4, help="Taille de lot (images/lot)."
    )
    parser.add_argument(
        "--queue", type=int, default=64, help="Capacité de la file d'attente."
    )
    parser.add_argument("--threads", type=int, default=0, help="Threads CPU.")
    parser.add_argument("--model-dir", default=None, help="Dossier des modèles.")
    parser.add_argument(
        "--no-gc",
        action="store_true",
        help="Désactive gc.collect() après chaque image.",
    )
    args = parser.parse_args()

    app = QCoreApplication(sys.argv)

    engine = LocalOCREngine(
        lang=args.lang, model_dir=args.model_dir, cpu_threads=args.threads
    )
    worker = BatchWorker(
        engine,
        batch_size=args.batch,
        queue_maxsize=args.queue,
        gc_after_each=not args.no_gc,
    )

    def on_result(record: Record) -> None:
        print(
            f"[#{record['index']}] {record['path']}: {record['text']!r} "
            f"(conf={record['confidence']:.3f}, {record['elapsed_ms']} ms)"
        )

    def on_error(message: str) -> None:
        print(f"ERREUR: {message}", file=sys.stderr)

    def on_progress(count: int) -> None:
        print(f"\rProgression: {count} image(s) traitée(s)...", end="", flush=True)

    def on_state(state: str) -> None:
        print(f"\r[état: {state}]")

    def on_finished(count: int) -> None:
        print(
            f"\nTerminé: {count} image(s) réussie(s), "
            f"{worker.failed} en erreur, {worker.batch_count} lot(s)."
        )

    worker.result_ready.connect(on_result)
    worker.error_occurred.connect(on_error)
    worker.progress_updated.connect(on_progress)
    worker.state_changed.connect(on_state)
    worker.batch_finished.connect(on_finished)
    worker.batch_finished.connect(app.quit)

    def handle_sigint(*_args: Any) -> None:
        print("\nAnnulation demandée (Ctrl+C)...")
        worker.cancel()
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)

    worker.submit(args.images)
    worker.start()
    exit_code = app.exec()

    worker.shutdown()
    engine.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
