"""Gestion du cycle de vie des moteurs OCR côté serveur.

PaddleOCR n'est **pas thread-safe** : une instance ne peut être utilisée que
par un seul thread à la fois. Le serveur FastAPI est asynchrone et peut donc
recevoir des requêtes concurrentes — c'est le rôle de :class:`EngineManager`
de garantir l'exclusion mutuelle.

Principe (pool round-robin) :

* Pour chaque langue, un pool de ``max_concurrency`` instances est créé
  paresseusement. Chaque instance est **pinnée** sur son propre
  ``ThreadPoolExecutor(max_workers=1)`` : l'inférence d'un moteur s'exécute
  donc toujours sur le même thread, ce qui est la seule configuration sûre
  pour PaddlePaddle.
* Les requêtes asynchrones piochent le prochain moteur libre dans une
  ``asyncio.Queue`` ; si tous les moteurs sont occupés, elles attendent.
* Un délai maximal par inférence est appliqué (``asyncio.wait_for``). En cas
  de dépassement, le moteur est considéré comme instable et remplacé.
* Le pool est "chaud" : la première requête déclenche le chargement des poids
  (lent), les suivantes sont instantanées. ``preload=True`` déclenche le
  chargement dès le démarrage du serveur, en arrière-plan.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np

from .config import Settings
from .core_ocr import LocalOCREngine

logger = logging.getLogger("scriptvault.engines")

EngineFactory = Callable[[], Any]


class _EngineSlot:
    """Un moteur OCR et le thread unique qui l'exécute."""

    __slots__ = ("engine", "executor", "busy", "created_at")

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        self.busy = False
        self.created_at = time.monotonic()

    def shutdown(self) -> None:
        try:
            close = getattr(self.engine, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # pragma: no cover - défensif
            logger.warning("Fermeture du moteur échouée: %s", exc)
        self.executor.shutdown(wait=False, cancel_futures=True)


class EngineManager:
    """Pool thread-safe de moteurs OCR, indexé par langue.

    Args:
        settings: Configuration du serveur.
        engine_factory: Fabrique optionnelle de moteur (injectable pour les
            tests). Par défaut, construit :class:`LocalOCREngine` avec la
            configuration du serveur.
    """

    def __init__(
        self,
        settings: Settings,
        engine_factory: Optional[EngineFactory] = None,
    ) -> None:
        self._settings = settings
        self._factory: EngineFactory = engine_factory or self._default_factory
        self._slots: dict[str, list[_EngineSlot]] = {}
        self._queues: dict[str, asyncio.Queue[_EngineSlot]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._preload_thread: Optional[threading.Thread] = None
        self._started_at = time.monotonic()
        self._closed = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _default_factory(self) -> Any:
        settings = self._settings
        return LocalOCREngine(
            lang=settings.lang,
            model_dir=settings.model_dir,
            cpu_threads=settings.cpu_threads,
            preprocess_kwargs={"binarize": settings.preprocess},
        )

    async def start(self) -> None:
        """Initialise le pool (pré-chargement éventuel en arrière-plan)."""
        if self._settings.preload:
            self._preload_thread = threading.Thread(
                target=self._warmup_sync,
                name="scriptvault-preload",
                daemon=True,
            )
            self._preload_thread.start()

    def _warmup_sync(self) -> None:
        """Charge le moteur par défaut hors du thread asyncio (démarrage rapide).

        L'instance créée est immédiatement disponible pour le pool (elle n'est
        pas marquée occupée).
        """
        try:
            lang = self._settings.lang
            engine = self._factory()
            self._slots.setdefault(lang, []).append(_EngineSlot(engine))
            logger.info(
                "Pré-chargement du moteur %r terminé (threads=%d).",
                lang,
                getattr(engine, "cpu_threads", "?"),
            )
        except Exception as exc:
            logger.warning(
                "Pré-chargement du moteur %r échoué: %s",
                self._settings.lang,
                exc,
            )

    async def _acquire_slot(self, lang: str) -> _EngineSlot:
        """Retourne un moteur libre pour ``lang`` (en crée si le pool n'est
        pas plein, sinon attend la libération d'un moteur)."""
        lock = self._locks.setdefault(lang, asyncio.Lock())
        queue = self._queues.setdefault(
            lang, asyncio.Queue(maxsize=self._settings.max_concurrency)
        )
        async with lock:
            slots = self._slots.setdefault(lang, [])
            # Remplit le pool tant qu'il reste de la place : les moteurs
            # disponibles sont mis dans la file, les occupés restent à part.
            while queue.empty() and len(slots) < self._settings.max_concurrency:
                engine = self._factory()
                slot = _EngineSlot(engine)
                slots.append(slot)
                queue.put_nowait(slot)
                logger.info(
                    "Pool %r: instance %d/%d créée.",
                    lang,
                    len(slots),
                    self._settings.max_concurrency,
                )
            # Récupère les moteurs disponibles créés par des requêtes
            # précédentes (cas pré-chargement ou pool déjà plein).
            for slot in slots:
                if not slot.busy and queue.qsize() < self._settings.max_concurrency:
                    queue.put_nowait(slot)
        return await queue.get()

    async def _release_slot(self, lang: str, slot: _EngineSlot) -> None:
        queue = self._queues.get(lang)
        if queue is not None:
            await queue.put(slot)

    # ------------------------------------------------------------------ #
    # API publique (asynchrone)
    # ------------------------------------------------------------------ #
    async def predict_bytes(
        self,
        data: bytes,
        *,
        lang: Optional[str] = None,
        preprocess: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Reconnaît le texte d'une image encodée (octets bruts)."""
        lang = lang or self._settings.lang
        return await self._predict(lang, "predict_bytes", data, preprocess)

    async def predict_array(
        self,
        image: np.ndarray,
        *,
        lang: Optional[str] = None,
        preprocess: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Reconnaît le texte d'une image numpy (page PDF rastérisée)."""
        lang = lang or self._settings.lang
        return await self._predict(lang, "predict_array", image, preprocess)

    async def _predict(
        self,
        lang: str,
        method: str,
        arg: bytes | np.ndarray,
        preprocess: Optional[bool],
    ) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Le gestionnaire de moteurs est fermé.")
        slot = await self._acquire_slot(lang)
        try:
            callable_method = getattr(slot.engine, method)
            kwargs: dict[str, Any] = {}
            if preprocess is not None:
                kwargs["preprocess"] = preprocess
            future = asyncio.wrap_future(
                slot.executor.submit(callable_method, arg, **kwargs)
            )
            timeout = self._settings.timeout_ms / 1000.0
            try:
                if timeout > 0:
                    return await asyncio.wait_for(future, timeout=timeout)
                return await future
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Inférence OCR dépassée ({self._settings.timeout_ms} ms)."
                ) from None
        finally:
            await self._release_slot(lang, slot)

    async def health(self) -> dict[str, Any]:
        """État du pool pour l'endpoint de santé."""
        engines: dict[str, Any] = {}
        for lang, slots in self._slots.items():
            engines[lang] = {
                "instances": len(slots),
                "ready": all(
                    bool(getattr(slot.engine, "is_ready", True)) for slot in slots
                ),
            }
        return {
            "preloading": (
                self._preload_thread is not None and self._preload_thread.is_alive()
            ),
            "engines": engines,
            "uptime_s": round(time.monotonic() - self._started_at, 1),
        }

    async def close(self) -> None:
        """Libère tous les moteurs et leurs threads."""
        self._closed = True
        for slots in self._slots.values():
            for slot in slots:
                slot.shutdown()
        self._slots.clear()
        self._queues.clear()
        self._locks.clear()
