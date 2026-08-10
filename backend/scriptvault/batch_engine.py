"""Moteur de lots OCR asynchrones — Architecture Entreprise.

Permet d'absorber des volumes massifs de documents (TIF multi-pages, PDF,
images) sans bloquer l'API :

* fichiers traités en parallèle sous un sémaphore global (une inférence à
  la fois, CPU-safe) ;
* progression consultable : état ``pending / processing / done / error`` par
  fichier, compteurs globaux, annulation propre ;
* résultats paginables — le client n'interroge que la synthèse puis le
  détail d'un fichier précis ;
* aperçu à la demande (PNG tel qu'analysé) avec cache LRU par lot ;
* post-traitement intégré : analyse de formulaire clé/valeur par page.

Chaque :class:`BatchJob` est autonome (annulable, supprimable) et le
:class:`BatchManager` limite la concurrence globale à ``max_concurrency``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import Settings
from .core_ocr import ImagePreprocessor
from .engines import EngineManager
from .form_analyzer import analyze_form_items
from .pdf import rasterize_pdf_bytes

logger = logging.getLogger("scriptvault.batch")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

# Nombre max d'aperçus gardés en mémoire par lot (cache LRU).
_PREVIEW_CACHE_SIZE = 8

# Qualité JPEG des aperçus persistés (excellent rendu, ~10× plus léger que PNG).
_PREVIEW_JPEG_QUALITY = 85

PENDING = "pending"
PROCESSING = "processing"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"


def _sanitize_filename(name: str) -> str:
    """Nom de fichier sûr pour le stockage temporaire (aucun séparateur)."""
    stem = _SAFE_NAME.sub("_", (name or "file")).strip("._")
    return stem or "file"


def _mean_confidence(items: list[dict[str, Any]]) -> float:
    """Confiance moyenne des lignes OCR d'une page."""
    if not items:
        return 0.0
    scores = [float(item.get("confidence", 0.0)) for item in items]
    return round(sum(scores) / len(scores), 4)


def _joined_text(items: list[dict[str, Any]]) -> str:
    """Texte plein d'une page : lignes séparées par des sauts."""
    return "\n".join(str(item.get("text", "")) for item in items).strip()


def _encode_preview(image: np.ndarray) -> str:
    """Encode l'image analysée en PNG base64 (data URL)."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _encode_jpeg_bytes(data: bytes) -> str:
    """Encode des octets JPEG en data URL (lecture disque → réponse web)."""
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------------- #
# Un fichier du lot
# --------------------------------------------------------------------------- #
class BatchFile:
    """Un fichier soumis au lot, avec son cycle de vie et son résultat."""

    __slots__ = (
        "file_id",
        "file_name",
        "disk_path",
        "preview_dir",
        "status",
        "error",
        "pages",
        "confidence",
        "elapsed_ms",
        "preview_cache",
        "form_overrides",
    )

    def __init__(
        self, file_id: str, file_name: str, disk_path: Path, preview_dir: Path
    ) -> None:
        self.file_id = file_id
        self.file_name = file_name
        self.disk_path = disk_path
        self.preview_dir = preview_dir
        self.status = PENDING
        self.error: Optional[str] = None
        self.pages: list[dict[str, Any]] = []
        self.confidence = 0.0
        self.elapsed_ms = 0.0
        self.preview_cache: "OrderedDict[int, str]" = OrderedDict()
        #: Corrections manuelles du formulaire : ``{page: {clé: valeur}}``
        #: appliquées en dernier recours (UI) et reprises par l'export Excel.
        self.form_overrides: dict[int, dict[str, str]] = {}

    def to_summary(self) -> dict[str, Any]:
        """Métadonnées légères pour la liste paginée (sans contenu OCR)."""
        return {
            "id": self.file_id,
            "name": self.file_name,
            "status": self.status,
            "error": self.error,
            "pages": len(self.pages),
            "confidence": self.confidence,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    async def get_preview(
        self,
        page: int,
        preprocessor: ImagePreprocessor,
        *,
        binarize: bool = True,
    ) -> Optional[str]:
        """Aperçu de la page demandée, mis en cache (LRU par lot).

        Chemin rapide : les aperçus JPEG sont persistés **une seule fois** au
        moment de l'analyse (`preview_dir`) — la réponse est une simple
        lecture disque, sans re-décodage ni re-prétraitement de l'image.
        Repli : pipeline complet (recalcul), lui aussi persisté ensuite.
        """
        if page < 0 or page >= len(self.pages):
            return None
        cached = self.preview_cache.get(page)
        if cached is not None:
            return cached

        if self.preview_dir is not None:
            jpeg_path = self.preview_dir / f"{page + 1:04d}.jpg"
            if jpeg_path.exists():
                try:
                    preview = _encode_jpeg_bytes(jpeg_path.read_bytes())
                    self._cache_preview(page, preview)
                    return preview
                except OSError:
                    pass

        image = await asyncio.to_thread(
            self._load_page_image, page, preprocessor, binarize
        )
        if image is None:
            return None
        preview = _encode_preview(image)
        if self.preview_dir is not None:
            self._persist_preview(image, page)
        self._cache_preview(page, preview)
        return preview

    def _cache_preview(self, page: int, preview: str) -> None:
        """Écrase le cache LRU (les entrées les plus anciennes sortent)."""
        self.preview_cache[page] = preview
        while len(self.preview_cache) > _PREVIEW_CACHE_SIZE:
            self.preview_cache.popitem(last=False)

    def _persist_preview(self, image: np.ndarray, page: int) -> None:
        """Écrit l'aperçu JPEG sur disque (une seule fois par page)."""
        try:
            ok, buffer = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_JPEG_QUALITY]
            )
            if ok:
                self.preview_dir.mkdir(parents=True, exist_ok=True)
                jpeg_path = self.preview_dir / f"{page + 1:04d}.jpg"
                if not jpeg_path.exists():
                    jpeg_path.write_bytes(buffer.tobytes())
        except (cv2.error, OSError):
            pass

    def _load_page_image(
        self,
        page: int,
        preprocessor: ImagePreprocessor,
        binarize: bool,
    ) -> Optional[np.ndarray]:
        """Relit le fichier temporaire et ré-applique le pipeline page cible."""
        try:
            images = preprocessor.read_pages_bytes(self.disk_path.read_bytes())
        except Exception:
            return None
        if page < 0 or page >= len(images):
            return None
        return preprocessor.preprocess(images[page], binarize=binarize)


# --------------------------------------------------------------------------- #
# Un lot (job) de fichiers
# --------------------------------------------------------------------------- #
class BatchJob:
    """Un lot : liste de fichiers, progression, annulation, résultats."""

    __slots__ = (
        "job_id",
        "name",
        "files",
        "status",
        "lang",
        "preprocess",
        "created_at",
        "started_at",
        "finished_at",
        "cancelled",
        "manager",
        "task",
    )

    def __init__(
        self,
        job_id: str,
        name: str,
        *,
        lang: str,
        preprocess: bool,
        manager: "BatchManager",
    ) -> None:
        self.job_id = job_id
        self.name = name or "Lot"
        self.files: list[BatchFile] = []
        self.status = PENDING
        self.lang = lang
        self.preprocess = preprocess
        self.created_at = time.time()
        self.started_at: Optional[float] = self.created_at
        self.finished_at: Optional[float] = None
        self.cancelled = False
        self.manager = manager
        self.task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    def add_file(self, file: BatchFile) -> None:
        self.files.append(file)

    def counts(self) -> dict[str, int]:
        """Compteurs globaux du lot."""
        return {
            "total": len(self.files),
            "done": sum(1 for f in self.files if f.status == DONE),
            "error": sum(1 for f in self.files if f.status == ERROR),
            "processing": sum(1 for f in self.files if f.status == PROCESSING),
            "pending": sum(1 for f in self.files if f.status == PENDING),
        }

    def summary(self) -> dict[str, Any]:
        """État complet du lot (léger, sans contenu OCR)."""
        counts = self.counts()
        confidences = [f.confidence for f in self.files if f.status == DONE]
        started = self.started_at or time.time()
        elapsed_ms = round(((self.finished_at or time.time()) - started) * 1000.0, 2)
        return {
            "id": self.job_id,
            "name": self.name,
            "status": self.status,
            "cancelled": self.cancelled,
            "lang": self.lang,
            "preprocess": self.preprocess,
            "counts": counts,
            "avg_confidence": round(sum(confidences) / len(confidences), 4)
            if confidences
            else 0.0,
            "elapsed_ms": elapsed_ms,
            "created_at": round(started, 2),
            "finished_at": round(self.finished_at, 2) if self.finished_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - outil de debug
        return f"<BatchJob {self.job_id} {self.status} {len(self.files)} files>"


# --------------------------------------------------------------------------- #
# Gestionnaire de lots
# --------------------------------------------------------------------------- #
class BatchManager:
    """Registre des lots et ordonnanceur global (concurrence bornée).

    Le traitement est déclenché en tâche de fond : l'API reste immédiatement
    disponible pour interroger l'avancement, afficher des résultats partiels
    ou annuler le lot.
    """

    def __init__(
        self,
        settings: Settings,
        preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        self._settings = settings
        self._preprocessor = preprocessor or ImagePreprocessor()
        self._jobs: dict[str, BatchJob] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._engines: Optional[EngineManager] = None

    async def init(self, engines: EngineManager) -> None:
        """Borne la concurrence globale et attache le pool de moteurs."""
        self._engines = engines
        self._semaphore = asyncio.Semaphore(
            max(1, self._settings.effective_max_concurrency)
        )

    # ------------------------------------------------------------------ #
    def create_job(
        self,
        files: list[tuple[str, bytes]],
        *,
        name: str = "",
        lang: str,
        preprocess: bool,
        storage_root: Path,
    ) -> BatchJob:
        """Crée un lot : écrit les fichiers en zone de travail, puis démarre."""
        job_id = uuid.uuid4().hex
        job = BatchJob(
            job_id,
            name,
            lang=lang,
            preprocess=preprocess,
            manager=self,
        )
        job_dir = storage_root / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        for index, (filename, data) in enumerate(files, start=1):
            disk_path = job_dir / f"{index:05d}_{_sanitize_filename(filename)}"
            disk_path.write_bytes(data)
            job.add_file(
                BatchFile(
                    file_id=f"{job_id}-{index:05d}",
                    file_name=filename,
                    disk_path=disk_path,
                    preview_dir=job_dir / "previews" / f"{index:05d}",
                )
            )
        self._jobs[job_id] = job
        job.started_at = time.time()
        job.status = PROCESSING
        job.task = asyncio.create_task(self._run_job(job))
        return job

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return [job.summary() for job in reversed(self._jobs.values())]

    def cancel_job(self, job: BatchJob) -> None:
        """Annule un lot : les fichiers restants passent en ``cancelled``."""
        job.cancelled = True
        for file in job.files:
            if file.status == PENDING:
                file.status = CANCELLED
        job.status = CANCELLED
        job.finished_at = time.time()

    def remove_job(self, job_id: str) -> None:
        """Supprime un lot et sa zone de travail (mémoire + disque)."""
        job = self._jobs.pop(job_id, None)
        if job is None:
            return
        if job.task is not None:
            job.task.cancel()
        try:
            if job.files:
                job_dir = job.files[0].disk_path.parent
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
        except Exception as exc:  # pragma: no cover - défensif
            logger.warning("Nettoyage du lot %r échoué: %s", job_id, exc)

    async def close(self) -> None:
        """Annule et supprime tous les lots (arrêt serveur)."""
        for job_id in list(self._jobs):
            self.remove_job(job_id)

    # ------------------------------------------------------------------ #
    # Traitement
    # ------------------------------------------------------------------ #
    async def _run_job(self, job: BatchJob) -> None:
        """Traite les fichiers *pending* sous le sémaphore global."""
        try:
            pending = [f for f in job.files if f.status == PENDING]

            async def handle(file: BatchFile) -> None:
                if job.cancelled:
                    return
                semaphore = self._semaphore
                assert semaphore is not None
                async with semaphore:
                    if job.cancelled:
                        file.status = CANCELLED
                        return
                    file.status = PROCESSING
                    try:
                        await self._process_file(job, file)
                        file.status = DONE
                    except Exception as exc:
                        file.status = ERROR
                        file.error = str(exc) or type(exc).__name__
                        logger.warning(
                            "Fichier %r du lot %r en échec: %s",
                            file.file_name,
                            job.job_id,
                            exc,
                        )

            await asyncio.gather(*(handle(file) for file in pending))
        finally:
            if not job.cancelled:
                job.status = DONE
            job.finished_at = time.time()
            counts = job.counts()
            logger.info(
                "Lot %r terminé: %d fichiers (%d OK, %d erreurs).",
                job.job_id,
                len(job.files),
                counts["done"],
                counts["error"],
            )

    async def _process_file(self, job: BatchJob, file: BatchFile) -> None:
        """OCR d'un fichier complet + post-traitement formulaire par page."""
        started = time.perf_counter()
        data = file.disk_path.read_bytes()
        images = await asyncio.to_thread(self._decode_pages, data, file.file_name)
        pages: list[dict[str, Any]] = []
        for index, image in enumerate(images, start=1):
            page = await self._recognize_page(job, file, image, index)
            page["page"] = index
            pages.append(page)
        file.pages = pages
        confidences = [p["confidence"] for p in pages if p["items"]]
        file.confidence = (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        )
        file.elapsed_ms = (time.perf_counter() - started) * 1000.0

    def _decode_pages(self, data: bytes, name: str) -> list[np.ndarray]:
        """Décode toutes les pages : PDF rastérisé, image directe (TIF inclus)."""
        if name.lower().endswith(".pdf"):
            return rasterize_pdf_bytes(data)
        return self._preprocessor.read_pages_bytes(data)

    async def _recognize_page(
        self, job: BatchJob, file: BatchFile, image: np.ndarray, page_number: int
    ) -> dict[str, Any]:
        """Prétraite, infère, analyse le formulaire et persiste l'aperçu."""
        if job.preprocess:
            processed = await asyncio.to_thread(
                self._preprocessor.preprocess, image, binarize=True
            )
        else:
            processed = image
        engines = self._engines
        assert engines is not None
        rois = self._settings.roi_profile or None
        if rois is not None:
            items = await engines.predict_array(
                image,
                lang=job.lang,
                preprocess=True,
                rois=rois,
                scan_barcode=False,
            )
        else:
            items = await engines.predict_array(
                image, lang=job.lang, preprocess=True
            )
        height, width = image.shape[:2]
        form = analyze_form_items(
            items,
            file_name=job.name,
            image=image,
            include_placeholders=True,
        )
        self._persist_page_preview(file, processed, page_number)
        return {
            "page": 0,
            "width": int(width),
            "height": int(height),
            "text": _joined_text(items),
            "confidence": _mean_confidence(items),
            "items": items,
            "form": form,
        }

    @staticmethod
    def _persist_page_preview(
        file: BatchFile, processed: np.ndarray, page_number: int
    ) -> None:
        """JPEG de la page *telle qu'analysée*, généré une fois à l'analyse."""
        if file.preview_dir is None:
            return
        try:
            ok, buffer = cv2.imencode(
                ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_JPEG_QUALITY]
            )
            if not ok:
                return
            file.preview_dir.mkdir(parents=True, exist_ok=True)
            jpeg_path = file.preview_dir / f"{page_number:04d}.jpg"
            if not jpeg_path.exists():
                jpeg_path.write_bytes(buffer.tobytes())
        except (cv2.error, OSError) as exc:
            logger.warning("Aperçu JPEG non persisté (%s): %s", page_number, exc)
