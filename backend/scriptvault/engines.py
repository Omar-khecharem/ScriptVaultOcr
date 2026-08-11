"""Gestion du cycle de vie des moteurs OCR côté serveur — scalable & Air-Gapped.

Les sessions ONNX (TrOCR HTR) sont thread-safe : le mode par défaut est donc
un :class:`ProcessPoolExecutor` de ``settings.workers`` workers (0 = 1 seul,
borné au nombre de cœurs). Chaque worker porte **son propre moteur OCR** ; le
Warm-start est effectué à l'initialisation : les poids des modèles sont
chargés en mémoire dès le démarrage du serveur, depuis le répertoire local
``models/trocr/`` — fonctionnement 100 % hors-ligne, sans dépendance cloud.

**Mode ``thread`` (repli)** : activé automatiquement si la fabrique de moteur
injectée n'est pas picklable (cas des tests), ou si
``SCRIPTVAULT_USE_PROCESSES=false``. Un pool round-robin de slots, chaque
slot étant pinné sur un ``ThreadPoolExecutor(max_workers=1)``.

Les obstacles : parallélisme = nombre de workers ; les requêtes asynchrones
sont mises en file et une limite de temps par inférence est appliquée
(``timeout_ms`` → ``TimeoutError`` → HTTP 504).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import multiprocessing
import os
import pickle
import platform
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional

from .config import Settings
from .core_ocr import (
    ImagePreprocessor,
    LocalOCREngine,
    OCRInitError,
    OCRResultItem,
    PageResult,
    ROIProfile,
    make_page_result,
)

logger = logging.getLogger("scriptvault.engines")

EngineFactory = Callable[[], Any]

# Plafond par défaut du nombre de workers OCR (mémoire ~1-2 Go par instance).
_WORKER_CAP = 8

# --------------------------------------------------------------------------- #
# État global du processus worker (un seul moteur par worker)
# --------------------------------------------------------------------------- #
_WORKER_ENGINE: Any = None
_WORKER_INIT_ERROR: Optional[BaseException] = None


def _worker_init(factory: EngineFactory) -> None:
    """Initialiseur de worker : construit le moteur et force le warm-up.

    Exécuté dans le contexte du worker par ``ProcessPoolExecutor``
    (``initializer``). Le warm-up matérialise les poids des modèles en RAM —
    c'est le Warm-start.
    """
    global _WORKER_ENGINE, _WORKER_INIT_ERROR
    _WORKER_INIT_ERROR = None
    try:
        _WORKER_ENGINE = factory()
        warm_up = getattr(_WORKER_ENGINE, "warm_up", None)
        if callable(warm_up):
            warm_up()
        logger.info(
            "Worker %d prêt: moteur %r chargé en mémoire.",
            os.getpid(),
            type(_WORKER_ENGINE).__name__,
        )
    except Exception as exc:
        _WORKER_INIT_ERROR = exc
        _WORKER_ENGINE = None
        logger.error("Échec du warm-up worker %d: %s", os.getpid(), exc)


def _worker_warmup() -> str:
    """Tâche barrière : confirme que le worker a initialisé son moteur."""
    if _WORKER_ENGINE is None:
        return f"worker-{os.getpid()}-vide"
    return f"worker-{os.getpid()}-{type(_WORKER_ENGINE).__name__}"


def _worker_call(method: str, arg: Any, kwargs: dict[str, Any]) -> Any:
    """Routeur principal : appelle ``engine.method(arg, **kwargs)`` dans le worker."""
    if _WORKER_INIT_ERROR is not None:
        raise OCRInitError(
            f"Worker OCR non initialisé: {_WORKER_INIT_ERROR}"
        ) from _WORKER_INIT_ERROR
    if _WORKER_ENGINE is None:
        raise OCRInitError("Worker OCR non initialisé.")
    fn = getattr(_WORKER_ENGINE, method, None)
    if fn is None:
        raise RuntimeError(f"Le moteur du worker ne fournit pas {method!r}.")
    return fn(arg, **kwargs)


# --------------------------------------------------------------------------- #
# Fabrique par défaut du moteur (module-level -> picklable, spawn-safe)
# --------------------------------------------------------------------------- #
def _default_engine_factory(settings: Settings) -> Any:
    """Construit le moteur OCR serveur (100 % local, weights du dossier models/).

    Le backend est choisi via ``settings.ocr_backend`` :

    * ``"paddle"`` — PaddleOCR PP-OCRv5 (qualité imprimé/français) ;
    * ``"htr"`` — TrOCR ONNX (repli sans Paddle) ;
    * ``"auto"`` (défaut) — Paddle si ``paddleocr`` est installé, sinon HTR.

    Si ``settings.vlm_enabled``, un :class:`~scriptvault.vlm_reader
    .LocalVLMReader` est attaché au moteur : les champs manuscrits du
    formulaire (Nom, Prénom, Établissement) sont lus en direct par le VLM
    local (Ollama/llama.cpp) avec repli TrOCR/PP-OCR automatique ; la
    détection globale et les champs imprimés/chiffres (CIN, Identifiant)
    restent sur PP-OCRv5/OpenCV.
    """
    backend = (settings.ocr_backend or "auto").lower()

    def _vlm_reader() -> Any:
        """Construit le lecteur VLM local si activé (``None`` sinon).

        Le modèle est pré-chargé dans Ollama en arrière-plan (``warm_up``) :
        sans cela, le premier formulaire d'un lot subit le chargement à froid
        (2-3 min) et peut dépasser le timeout de grille.
        """
        if not settings.vlm_enabled:
            return None
        try:
            from .vlm_reader import LocalVLMReader

            reader = LocalVLMReader()
            try:
                threading.Thread(
                    target=reader.warm_up,
                    name="scriptvault-vlm-warmup",
                    daemon=True,
                ).start()
            except Exception:  # pragma: no cover - défensif
                logger.debug("Pré-chauffage VLM non lancé.", exc_info=True)
            return reader
        except Exception as exc:
            logger.warning(
                "VLM local indisponible (%s) ; lecture manuscrite classique.", exc
            )
            return None

    if backend in ("paddle", "auto"):
        try:
            from .paddle_engine import PaddleOCREngine

            engine = PaddleOCREngine(
                lang=settings.lang,
                cpu_threads=settings.cpu_threads,
                max_side_len=settings.max_side_len or None,
                barcode=settings.barcode_enabled,
                barcode_budget_ms=float(settings.barcode_budget_ms),
                vlm_reader=_vlm_reader(),
            )
            if not getattr(engine, "is_ready", False):
                raise OCRInitError("PaddleOCR n'a pas pu initialiser ses modèles.")
            return engine
        except OCRInitError as exc:
            if backend == "paddle":
                raise
            logger.warning("Paddle indisponible (%s) ; repli backend HTR.", exc)
    return LocalOCREngine(
        lang=settings.lang,
        model_dir=settings.model_dir,
        cpu_threads=settings.cpu_threads,
        preprocess_kwargs={"binarize": settings.preprocess},
        max_side_len=settings.max_side_len or None,
        barcode=settings.barcode_enabled,
        barcode_budget_ms=float(settings.barcode_budget_ms),
        vlm_reader=_vlm_reader(),
    )


def _resolve_workers(settings: Settings) -> int:
    """Nombre de workers : ``SCRIPTVAULT_WORKERS`` si > 0, sinon 1.

    Chaque worker charge ses propres poids de modèles en RAM (~1-2 Go par
    instance) : le défaut est volontairement conservateur pour ne pas saturer
    la machine. Passez ``SCRIPTVAULT_WORKERS`` (plafond 8) pour paralléliser.
    """
    if settings.workers > 0:
        return max(1, min(settings.workers, _WORKER_CAP))
    return 1


def _is_picklable(obj: Any) -> bool:
    """Vrai si l'objet est sérialisable (requis pour le mode process)."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Repli thread (moteurs non picklables : tests) et dispatch de tâches
# --------------------------------------------------------------------------- #
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


def _thread_dispatch(engine: Any, method: str, arg: Any, kwargs: dict[str, Any]) -> Any:
    """Exécute une méthode sur le moteur du slot (mode thread).

    Repli : si le moteur (ex. fabrique factice) ne fournit pas
    ``predict_pages_bytes``, on décompose le fichier (TIFF multi-pages inclus)
    via :class:`ImagePreprocessor` et on appelle ``predict_array`` par page.
    """
    fn = getattr(engine, method, None)
    if fn is not None:
        return fn(arg, **kwargs)
    if method == "predict_pages_bytes":
        return _thread_predict_pages_fallback(
            engine, arg, bool(kwargs.get("preprocess", True))
        )
    raise AttributeError(f"Le moteur ne fournit pas la méthode {method!r}.")


def _thread_predict_pages_fallback(
    engine: Any, data: bytes, preprocess: bool
) -> list[PageResult]:
    """Repli multipage : lecture TIFF multi-pages + OCR ``predict_array`` par page."""
    preprocessor = ImagePreprocessor()
    images = preprocessor.read_pages_bytes(data)
    results: list[PageResult] = []
    for index, page in enumerate(images, start=1):
        started = time.perf_counter()
        processed = preprocessor.preprocess(page, binarize=True) if preprocess else page
        height, width = processed.shape[:2]
        items = engine.predict_array(processed, preprocess=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results.append(
            make_page_result(index, width, height, items, elapsed_ms, [], processed)
        )
    return results


# --------------------------------------------------------------------------- #
# Gestionnaire de pool
# --------------------------------------------------------------------------- #
class EngineManager:
    """Pool scalable de moteurs OCR, indexé par langue.

    Modes:
    * ``process`` — :class:`ProcessPoolExecutor`, un moteur par worker,
      Warm-start au démarrage, s'adapte au nombre de cœurs.
    * ``thread`` — slots round-robin pinnés sur leur propre thread (repli
      pour les fabriques non picklables / ``SCRIPTVAULT_USE_PROCESSES=false``).

    Args:
        settings: Configuration serveur.
        engine_factory: Fabrique optionnelle de moteur (injectable pour les
            tests). Doit être picklable pour le mode process.
    """

    def __init__(
        self,
        settings: Settings,
        engine_factory: Optional[EngineFactory] = None,
    ) -> None:
        self._settings = settings
        self._factory: EngineFactory = (
            engine_factory
            if engine_factory is not None
            else functools.partial(_default_engine_factory, settings)
        )
        self._use_processes = settings.use_processes and _is_picklable(self._factory)
        self._mode = "process" if self._use_processes else "thread"
        self._workers = _resolve_workers(settings)
        logger.info(
            "Mode de pool sélectionné: %s (workers=%d).",
            self._mode,
            self._workers,
        )

        self._pool: Optional[ProcessPoolExecutor] = None
        self._slots: dict[str, list[_EngineSlot]] = {}
        self._queues: dict[str, asyncio.Queue[_EngineSlot]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._preload_thread: Optional[threading.Thread] = None
        self._warmup_thread: Optional[threading.Thread] = None
        self._warmup_futures: list[Future[Any]] = []
        self._preloading = False
        self._started_at = time.monotonic()
        self._closed = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Initialise le pool (Warm-start des modèles au démarrage)."""
        if self._mode == "process":
            self._ensure_pool()
            return
        if self._settings.preload:
            self._preload_thread = threading.Thread(
                target=self._warmup_sync,
                name="scriptvault-preload",
                daemon=True,
            )
            self._preload_thread.start()

    def _ensure_pool(self) -> None:
        """Crée le pool process (une fois) et déclenche le Warm-start si configuré."""
        if self._pool is not None:
            return
        if platform.system() == "Windows":
            context: multiprocessing.context.BaseContext = multiprocessing.get_context(
                "spawn"
            )
        else:
            context = multiprocessing.get_context("fork")
        self._pool = ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(self._factory,),
        )
        if self._settings.preload:
            self._preloading = True
            self._warmup_thread = threading.Thread(
                target=self._warmup_pool,
                name="scriptvault-warmup",
                daemon=True,
            )
            self._warmup_thread.start()

    def _warmup_pool(self) -> None:
        """Pré-charge chaque worker (modèles en RAM) avant service des requêtes."""
        pool = self._pool
        if pool is None:
            self._preloading = False
            return
        try:
            futures = [pool.submit(_worker_warmup) for _ in range(self._workers)]
            self._warmup_futures = futures
            for future in futures:
                future.result(timeout=1800)
            self._preloading = False
            logger.info(
                "Warm-start terminé: %d worker(s) OCR prêts (modèles en RAM).",
                self._workers,
            )
        except Exception as exc:
            self._preloading = False
            logger.warning("Warm-start partiel (certains workers en échec): %s", exc)

    def _warmup_sync(self) -> None:
        """Mode thread : charge le moteur par défaut hors du loop asyncio."""
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

    # ------------------------------------------------------------------ #
    # Mode thread : pool round-robin
    # ------------------------------------------------------------------ #
    async def _acquire_slot(self, lang: str) -> _EngineSlot:
        """Retourne un moteur libre (en crée un si quota, sinon attend).

        Invariant : la file contient exactement les slots libres ; un slot
        occupé est retiré de la file pendant l'exécution, ce qui garantit
        qu'aucune requête n'obtient deux fois le même moteur simultanément.
        """
        lock = self._locks.setdefault(lang, asyncio.Lock())
        queue = self._queues.setdefault(
            lang, asyncio.Queue(maxsize=self._settings.effective_max_concurrency)
        )
        async with lock:
            slots = self._slots.setdefault(lang, [])
            if queue.empty() and len(slots) < self._settings.effective_max_concurrency:
                try:
                    engine = self._factory()
                except Exception as exc:
                    raise OCRInitError(
                        f"Échec de l'initialisation du moteur OCR ({lang}): {exc}"
                    ) from exc
                slot = _EngineSlot(engine)
                slots.append(slot)
                queue.put_nowait(slot)
                logger.info(
                    "Pool %r: instance %d/%d créée.",
                    lang,
                    len(slots),
                    self._settings.max_concurrency,
                )
        slot = await queue.get()
        slot.busy = True
        return slot

    async def _release_slot(self, lang: str, slot: _EngineSlot) -> None:
        slot.busy = False
        queue = self._queues.get(lang)
        if queue is not None:
            await queue.put(slot)

    # ------------------------------------------------------------------ #
    # Exécution des requêtes
    # ------------------------------------------------------------------ #
    async def _predict(
        self,
        lang: str,
        method: str,
        arg: Any,
        *,
        preprocess: Optional[bool] = None,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("Le gestionnaire de moteurs est fermé.")
        kwargs: dict[str, Any] = {}
        if preprocess is not None:
            kwargs["preprocess"] = preprocess
        if rois is not None:
            kwargs["rois"] = rois
        if scan_barcode is not None:
            kwargs["scan_barcode"] = scan_barcode

        if self._mode == "process":
            self._ensure_pool()
            pool = self._pool
            if pool is None:
                raise RuntimeError("Pool de workers indisponible.")
            future = asyncio.wrap_future(pool.submit(_worker_call, method, arg, kwargs))
            return await self._await_result(future, lang)
        slot = await self._acquire_slot(lang)
        try:
            future = asyncio.wrap_future(
                slot.executor.submit(_thread_dispatch, slot.engine, method, arg, kwargs)
            )
            return await self._await_result(future, lang)
        finally:
            await self._release_slot(lang, slot)

    async def _await_result(self, future: Any, lang: str) -> Any:
        timeout = self._settings.timeout_ms / 1000.0
        try:
            if timeout > 0:
                return await asyncio.wait_for(future, timeout=timeout)
            return await future
        except asyncio.TimeoutError:
            logger.warning(
                "Inférence OCR dépassée pour %r (%s ms).",
                lang,
                self._settings.timeout_ms,
            )
            raise TimeoutError(
                f"Inférence OCR dépassée ({self._settings.timeout_ms} ms)."
            ) from None

    # ------------------------------------------------------------------ #
    # API publique (asynchrone)
    # ------------------------------------------------------------------ #
    async def predict_bytes(
        self,
        data: bytes,
        *,
        lang: Optional[str] = None,
        preprocess: Optional[bool] = None,
    ) -> list[OCRResultItem]:
        """Reconnaît le texte d'une image encodée (première page si TIF)."""
        lang = lang or self._settings.lang
        return await self._predict(lang, "predict_bytes", data, preprocess=preprocess)

    async def predict_array(
        self,
        image: Any,
        *,
        lang: Optional[str] = None,
        preprocess: Optional[bool] = None,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
    ) -> list[OCRResultItem]:
        """Reconnaît le texte d'une image numpy (page PDF rastérisée).

        ``rois`` : mode formulaires — chaque zone est transcrite séparément
        (items portant ``label``), idéal pour la lecture de feuilles d'examen.
        """
        lang = lang or self._settings.lang
        return await self._predict(
            lang,
            "predict_array",
            image,
            preprocess=preprocess,
            rois=rois,
            scan_barcode=scan_barcode,
        )

    async def predict_pages_bytes(
        self,
        data: bytes,
        *,
        lang: Optional[str] = None,
        preprocess: Optional[bool] = None,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
    ) -> list[PageResult]:
        """Analyse **toutes** les pages d'un fichier image (TIF multi-pages).

        Lecture hybride : code-barres/QR locaux puis OCR global, ou recettes
        par zones d'intérêt (``rois``) si un profil de formulaire est fourni.
        Chaque :class:`PageResult` embarque l'image exactement analysée
        (prétraitée, zones masquées) — prête pour l'overlay web.
        """
        lang = lang or self._settings.lang
        return await self._predict(
            lang,
            "predict_pages_bytes",
            data,
            preprocess=preprocess,
            rois=rois,
            scan_barcode=scan_barcode,
        )

    # ------------------------------------------------------------------ #
    # État
    # ------------------------------------------------------------------ #
    async def health(self) -> dict[str, Any]:
        """État du pool pour l'endpoint de santé."""
        if self._mode == "process":
            engines: dict[str, Any] = {
                self._settings.lang: {
                    "instances": self._workers,
                    "ready": not self._preloading,
                    "mode": "process",
                }
            }
        else:
            engines = {}
            for lang, slots in self._slots.items():
                engines[lang] = {
                    "instances": len(slots),
                    "ready": all(
                        bool(getattr(slot.engine, "is_ready", True)) for slot in slots
                    ),
                    "mode": "thread",
                }
        return {
            "mode": self._mode,
            "preloading": self._preloading,
            "engines": engines,
            "workers": self._workers if self._mode == "process" else None,
            "uptime_s": round(time.monotonic() - self._started_at, 1),
        }

    async def close(self) -> None:
        """Libère tous les workers, moteurs et threads."""
        self._closed = True
        self._preloading = False
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:  # pragma: no cover - défensif
                logger.warning("Arrêt du pool en erreur: %s", exc)
            self._pool = None
        for slots in self._slots.values():
            for slot in slots:
                slot.shutdown()
        self._slots.clear()
        self._queues.clear()
        self._locks.clear()
