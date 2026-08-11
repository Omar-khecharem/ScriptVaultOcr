"""Moteur OCR PaddleOCR (PP-OCRv5 mobile, CPU) — lecture fiable hors-ligne.

Ce moteur remplace le backend HTR TrOCR par défaut pour les documents
**imprimés** (feuilles d'examen, formulaires) : PaddleOCR lit le texte
imprimé français beaucoup plus fidèlement que TrOCR-small-handwritten
(sortie ``"in the United States's"`` constante sans la correction du
décodeur, charabia même corrigé).

Caractéristiques :

* PP-OCRv5 mobile (détection + reconnaissance), CPU pur, ``enable_mkldnn=False``
  (l'accélération oneDNN crashe avec paddlepaddle 3.3.1 / PIR — erreur
  ``ConvertPirAttribute2RuntimeAttribute`` documentée).
* Le réglage ``max_side_len`` (défaut 1400) est appliqué **avant** Paddle :
  le document est réduit à cette taille, ce qui divise par ~4 le coût de la
  détection sans perte de qualité (86-95 s → ~13 s sur la feuille de test).
* API identique à :class:`core_ocr.LocalOCREngine` :
  ``predict_bytes`` / ``predict_array`` / ``predict_pages_bytes``, sortie
  structurée ``[{"text", "confidence", "box"}]``.
* Les modèles sont téléchargés au premier appel (cachés dans
  ``~/.paddlex/official_models``) — fonctionnement 100 % hors-ligne ensuite.

Thread-safety : le pool de moteurs crée une instance Paddle par processus
worker (mode process) ou par slot (mode thread) — jamais de partage entre
threads.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np

from .core_ocr import (
    BarcodeResult,
    BarcodeScanner,
    ImagePreprocessor,
    OCRBaseError,
    OCRImageError,
    OCRInferenceError,
    OCRInitError,
    OCRResultItem,
    PageResult,
    ROIProfile,
    frac_to_px,
    make_page_result,
    mask_barcodes,
)
from .image_processing import has_form_structure, read_exam_form_zones

PathLike = Union[str, os.PathLike[str]]

__all__ = ["PaddleOCREngine"]

logger = logging.getLogger("scriptvault.paddle_engine")

#: Modèles PP-OCRv5 mobiles (compromis vitesse/qualité sur CPU).
_DEFAULT_DET_MODEL = "PP-OCRv5_mobile_det"
_DEFAULT_REC_MODEL = "PP-OCRv5_mobile_rec"
_DEFAULT_MAX_SIDE = 1400

#: Note de sortie analytics (client/API).
ENGINE_BACKEND_NAME = "paddle"


def _as_ocr_error(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Convertit toute erreur Paddle/OpenCV en erreur OCR typée."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except (OCRBaseError, MemoryError):
            raise
        except cv2.error as exc:
            raise OCRImageError(f"Erreur OpenCV dans {fn.__name__}: {exc}") from exc
        except Exception as exc:
            raise OCRInferenceError(
                f"Échec de {fn.__name__}: {type(exc).__name__}: {exc}"
            ) from exc

    return wrapper  # type: ignore[return-value]


def _box_from_poly(poly: Any) -> list[list[int]]:
    """Convertit un polynôme Paddle ``(N,2)`` en liste ``[[x, y], ...]``."""
    try:
        points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    except (ValueError, TypeError):
        return []
    return [[int(round(float(p[0]))), int(round(float(p[1])))] for p in points]


def _items_from_page(page: Any) -> list[OCRResultItem]:
    """Extrait les items OCR d'un résultat PaddleOCR (une page)."""
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []
    polys = page.get("dt_polys") or []
    items: list[OCRResultItem] = []
    for text, score, poly in zip(texts, scores, polys):
        text = str(text).strip() if text is not None else ""
        if not text:
            continue
        items.append(
            {
                "text": text,
                "confidence": float(score) if score is not None else 0.0,
                "box": _box_from_poly(poly),
            }
        )
    return items


class PaddleOCREngine:
    """Moteur OCR PaddleOCR v5 (CPU) — même interface que :class:`core_ocr.LocalOCREngine`.

    Args:
        lang: Langue PaddleOCR (``"fr"`` recommandé pour les feuilles
            d'examen françaises).
        model_dir: Ignoré (conservé pour compatibilité d'interface — les
            poids Paddle vivent dans ``~/.paddlex`` géré par paddleocr).
        cpu_threads: Threads CPU (Paddle/OpenMP ; ``0`` = auto).
        preprocess_kwargs: Conservés pour compatibilité d'interface (Paddle
            consomme l'image brute).
        preprocessor: Instance :class:`ImagePreprocessor` (injection tests).
        max_side_len: Réduction de la plus grande dimension avant le réseau
            Paddle (défaut 1420 — sweet spot vitesse/qualité mesuré).
        barcode: Active le scanner local de codes-barres/QR OpenCV.
        barcode_budget_ms / barcode_max_preview_side: réglages du scan.
        det_model_name / rec_model_name: Modèles PP-OCR à utiliser.

    Raises:
        OCRInitError: ``paddleocr``/``paddlepaddle`` absents de l'environnement.
    """

    def __init__(
        self,
        lang: str = "fr",
        model_dir: Optional[PathLike] = None,
        cpu_threads: int = 0,
        preprocess_kwargs: Optional[dict[str, Any]] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        logger: Optional[logging.Logger] = None,
        *,
        max_side_len: Optional[int] = None,
        barcode: bool = True,
        barcode_budget_ms: float = 15.0,
        barcode_max_preview_side: int = 1000,
        det_model_name: str = _DEFAULT_DET_MODEL,
        rec_model_name: str = _DEFAULT_REC_MODEL,
        vlm_reader: Optional[Any] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("scriptvault.paddle_engine.engine")
        self.lang = lang
        self._max_side_len = (
            max_side_len if max_side_len and max_side_len > 0 else _DEFAULT_MAX_SIDE
        )
        self._preprocessor = preprocessor or ImagePreprocessor(
            logger=self.logger, max_side_len=self._max_side_len
        )
        self._preprocess_kwargs = dict(preprocess_kwargs or {})
        self._ready = False
        self._barcode_scanner = (
            BarcodeScanner(
                time_budget_ms=barcode_budget_ms,
                max_preview_side=barcode_max_preview_side,
                logger=self.logger,
            )
            if barcode
            else None
        )
        self._det_model_name = det_model_name
        self._rec_model_name = rec_model_name
        self._ocr: Any = None
        self._backend_name = ENGINE_BACKEND_NAME
        self.cpu_threads = cpu_threads if cpu_threads > 0 else (os.cpu_count() or 4)
        self._htr: Any = None
        self._htr_lock = threading.Lock()

        # --- Champs manuscrits : VLM local (direct) sinon TrOCR (repli) --- #
        self._vlm_reader: Optional[Any] = vlm_reader
        self._handwritten_fields: tuple[str, ...] = ("nom", "prenom")
        self._handwritten_reader: Any = None
        self._band_grid_reader: Any = None
        if vlm_reader is not None:
            if getattr(vlm_reader, "fallback", None) is None:
                vlm_reader.fallback = self._htr_recognize
            self._handwritten_reader = self._vlm_handwritten_read
            self._handwritten_fields = ("nom", "prenom", "etablissement")
            # Lecture grille : toutes les bandes sont lues par le VLM (comme
            # Gemini) quand l'OCR local lit mal les bandes ; repli TrOCR sinon.
            self._band_grid_reader = self._vlm_band_grid_read
            self.logger.info(
                "Lecture manuscrite routée vers le VLM local (%s).",
                getattr(getattr(vlm_reader, "config", None), "model", "?"),
            )
        self.warm_up()

    # ------------------------------------------------------------------ #
    # Cycle de vie
    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        return self._ready

    def _import_paddle(self) -> Any:
        """Initialise l'OCR (une seule fois, paresseux).

        Préférence : la pipette PP-OCRv5 **ONNX Runtime** (:mod:`.onnx_ocr`)
        quand ``models/paddle_onnx/`` contient les exports officiels — ~10×
        plus rapide que paddlex sur CPU (paddlepaddle 3.x sans oneDNN).
        Repli : l'objet ``PaddleOCR`` de paddlex (même interface).
        """
        if self._ocr is not None:
            return self._ocr
        try:
            from .onnx_ocr import OnnxPaddleOCR, _default_model_dir

            if os.path.exists(os.path.join(_default_model_dir(), "PP-OCRv5_mobile_det.onnx")):
                self._ocr = OnnxPaddleOCR(
                    cpu_threads=self.cpu_threads, logger=self.logger
                )
                self._backend_name = "ppocrv5-onnx"
                return self._ocr
        except OCRInitError as exc:
            self.logger.warning("Backend ONNX indisponible (%s) ; repli paddlex.", exc)
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError as exc:
            raise OCRInitError(
                "Backend Paddle non disponible: installez paddleocr + "
                f"paddlepaddle (pip install paddleocr paddlepaddle). Erreur: {exc}"
            ) from exc
        started = time.perf_counter()
        try:
            self.logger.info(
                "Backend PaddleOCR: det=%s rec=%s lang=%s (mkldnn off)…",
                self._det_model_name,
                self._rec_model_name,
                self.lang,
            )
            # NOTE: enable_mkldnn=True échoue sur paddlepaddle 3.3.1 / PIR
            # (NotImplementedError: ConvertPirAttribute2RuntimeAttribute
            #  oneDNN). On désactive oneDNN et on mise sur l'entrée réduite.
            self._ocr = PaddleOCR(
                lang=self.lang,
                device="cpu",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name=self._det_model_name,
                text_recognition_model_name=self._rec_model_name,
                text_recognition_batch_size=16,
                cpu_threads=self.cpu_threads,
            )
        except Exception as exc:
            raise OCRInitError(
                f"Échec de l'initialisation PaddleOCR: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed = round(time.perf_counter() - started, 1)
        self.logger.info("PaddleOCR chargé et prêt en %.1f s.", elapsed)
        return self._ocr

    def warm_up(self) -> None:
        """Pré-charge les poids Paddle (premier appel)."""
        try:
            self._import_paddle()
            self._ready = self._ocr is not None
        except Exception as exc:
            self._ready = False
            self.logger.warning("Warm-up Paddle échoué: %s", exc)

    def close(self) -> None:
        """Libère le modèle Paddle (les fichiers restent en cache local)."""
        ocr = self._ocr
        try:
            if ocr is not None and hasattr(ocr, "close"):
                ocr.close()
        except Exception as exc:  # pragma: no cover - défensif
            self.logger.warning("Fermeture PaddleOCR échouée: %s", exc)
        self._ocr = None
        vlm = self._vlm_reader
        if vlm is not None:
            try:
                vlm.close()
            except Exception as exc:  # pragma: no cover - défensif
                self.logger.warning("Fermeture du lecteur VLM échouée: %s", exc)
            self._vlm_reader = None
        self._ready = False

    def __enter__(self) -> "PaddleOCREngine":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    @_as_ocr_error
    def _run_ocr(self, image: np.ndarray) -> list[OCRResultItem]:
        ocr = self._import_paddle()
        result = ocr.predict(image)
        if not result:
            return []
        return _items_from_page(result[0])

    @_as_ocr_error
    def _recognize_crop(self, crop: np.ndarray) -> list[OCRResultItem]:
        """Transcrit une petite zone (ROI) via PaddleOCR (cropts séparés)."""
        if crop.size == 0:
            return []
        ocr = self._import_paddle()
        result = ocr.predict(crop)
        if not result:
            return []
        return _items_from_page(result[0])

    # ------------------------------------------------------------------ #
    # HTR (TrOCR) pour les champs manuscrits — paresseux
    # ------------------------------------------------------------------ #
    def _htr_recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """Transcrit une zone manuscrite avec TrOCR (``(texte, confiance)``).

        Les modèles TrOCR ONNX vivent dans ``models/trocr/`` (via
        ``python -m scriptvault.download_trocr_models``). Chargés une seule
        fois (paresseux, verrouillé) ; session ONNX thread-safe. Si les
        modèles sont absents, retourne ``("", 0.0)`` — le pipeline conserve
        la lecture PP-OCR.
        """
        if crop is None or crop.size == 0:
            return "", 0.0
        if os.environ.get("SCRIPTVAULT_HTR", "0") not in ("1", "true", "yes", "on"):
            return "", 0.0  # TrOCR non finetuné : lecture inexploitable
        htr = self._htr
        if htr is None:
            with self._htr_lock:
                htr = self._htr
                if htr is None:
                    from .htr_engine import load_htr

                    root = os.path.dirname(os.path.abspath(__file__))
                    htr_dir = os.path.abspath(
                        os.path.join(root, "..", "..", "models", "trocr")
                    )
                    htr = load_htr(Path(htr_dir), threads=max(1, self.cpu_threads // 4))
                    self._htr = htr
        if htr is None:
            return "", 0.0
        try:
            from .image_processing import tight_ink_crop

            tight = tight_ink_crop(crop)
            if tight is None or tight.size == 0:
                return "", 0.0
            return htr.recognize(tight)
        except Exception as exc:
            self.logger.warning("Relecture HTR échouée: %s", exc)
            return "", 0.0

    def _vlm_handwritten_read(self, crop: np.ndarray, field_type: str) -> tuple[str, float]:
        """Routage VLM : lecture directe du crop manuscrit, repli TrOCR.

        Le VLM local reçoit la zone découpée + le type de champ (prompt
        contextuel : acronymes d'établissements, lexique de noms tunisiens).
        En cas d'échec ou de timeout interne, le lecteur bascule sur
        ``_htr_recognize`` (TrOCR) — la lecture PP-OCR composite est
        préservée si TrOCR est lui-même indisponible.
        """
        reader = self._vlm_reader
        if reader is None or not getattr(reader, "is_enabled", False):
            return self._htr_recognize(crop)
        try:
            result = reader.sync_read_handwritten_crop(crop, field_type)
        except Exception as exc:
            self.logger.warning("Lecture VLM en échec (%s) ; repli HTR.", exc)
            return self._htr_recognize(crop)
        text = str(result.get("text", "")).strip()
        if not text:
            return "", 0.0
        return text, max(0.0, min(1.0, float(result.get("confidence", 0.0))))

    def _vlm_band_grid_read(
        self,
        grid: np.ndarray,
        first_row: int,
        last_row: int,
    ) -> Optional[list[tuple[int, str, float]]]:
        """Routage VLM : lecture de la grille des bandes (repli TrOCR sinon).

        Retourne ``None`` si le lecteur VLM n'est pas disponible ou échoue —
        ``_transcribe_band_rows`` reprend alors la passe composite PP-OCR.
        """
        reader = self._vlm_reader
        if reader is None or not getattr(reader, "is_enabled", False):
            return None
        sync = getattr(reader, "sync_read_form_band_grid", None)
        if not callable(sync):
            return None
        try:
            return sync(grid, first_row, last_row)
        except Exception as exc:
            self.logger.warning("Grille VLM en échec (%s) ; repli PP-OCR.", exc)
            return None

    @_as_ocr_error
    def _recognize_crops(
        self, crops: list[np.ndarray]
    ) -> list[list[OCRResultItem]]:
        """Transcrit plusieurs zones en UNE passe réseau (détection unique).

        Les zones sont empilées dans une image composite (écart de 48 px) :
        PaddleOCR n'exécute sa détection qu'une seule fois au lieu d'une par
        zone — le pivot de vitesse des champs de formulaire. Les items sont
        redistribués à leur zone par position Y.
        """
        if not crops:
            return []
        if len(crops) == 1:
            ocr = self._import_paddle()
            results = ocr.predict(crops)
            return [_items_from_page(results[0]) if results else []]

        gap = 48
        width = max(crop.shape[1] for crop in crops)
        total_h = sum(crop.shape[0] for crop in crops) + gap * (len(crops) - 1)
        canvas = np.full((total_h, width, 3), 255, dtype=np.uint8)
        origins: list[int] = []
        y = 0
        for crop in crops:
            origins.append(y)
            canvas[y : y + crop.shape[0], 0 : crop.shape[1]] = crop
            y += crop.shape[0] + gap

        ocr = self._import_paddle()
        results = ocr.predict(canvas)
        if not results:
            return [[] for _ in crops]
        page = results[0]
        ranges = [
            (origin, origin + crop.shape[0]) for origin, crop in zip(origins, crops)
        ]
        output: list[list[OCRResultItem]] = [[] for _ in crops]
        for item in _items_from_page(page):
            box = item.get("box")
            if not box:
                continue
            cy = sum(pt[1] for pt in box) / 4.0
            index = min(range(len(ranges)), key=lambda i: abs(cy - sum(ranges[i]) / 2.0))
            y0, y1 = ranges[index]
            item["box"] = [
                [int(round(x)), int(round(y - y0))] for x, y in box
            ]
            output[index].append(item)
        return output

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #
    @_as_ocr_error
    def predict_array(
        self,
        image: np.ndarray,
        *,
        preprocess: bool = True,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
        zones: Optional[bool] = None,
    ) -> list[OCRResultItem]:
        """OCR Paddle sur un tableau numpy (une seule page).

        ``preprocess`` : la page est réduite à ``max_side_len`` (1400 par
        défaut) avant le réseau — le pivot de vitesse (13 s au lieu de
        60-90 s sur la feuille d'examen de référence). Paddle préfère
        l'image brute : la binarisation/CLAHE usuels sont sautés.

        ``zones`` : lecture du formulaire **zone par zone** (grilles de
        chiffres MNIST + lignes pointillées) au lieu de la passe pleine
        page. ``None`` = automatique (page pré-analysée), ``False`` = forcé.
        """
        if not self._ready:
            raise OCRInitError("Le moteur PaddleOCR n'est pas initialisé.")
        if rois:
            return self._page_result(image, 0, preprocess, rois, scan_barcode, zones)[
                "items"
            ]
        if zones is not False and has_form_structure(image):
            kwargs: dict[str, Any] = {}
            if self._handwritten_reader is not None:
                kwargs["handwritten_reader"] = self._handwritten_reader
                kwargs["handwritten_fields"] = self._handwritten_fields
                if self._band_grid_reader is not None:
                    kwargs["band_grid_reader"] = self._band_grid_reader
            else:
                kwargs["htr_recognize"] = self._htr_recognize
            return read_exam_form_zones(
                image,
                self._recognize_crop,
                recognize_crops=self._recognize_crops,
                **kwargs,
            )
        if preprocess:
            image, _ = self._preprocessor.resize_for_ocr(
                image, self._max_side_len
            )
        return self._run_ocr(image)

    def predict(
        self, image_path: PathLike, *, preprocess: bool = True
    ) -> list[OCRResultItem]:
        image = self._preprocessor.read_image(image_path)
        return self.predict_array(image, preprocess=preprocess)

    def predict_bytes(
        self, data: bytes | np.ndarray, *, preprocess: bool = True
    ) -> list[OCRResultItem]:
        image = self._preprocessor.read_image_bytes(data)
        return self.predict_array(image, preprocess=preprocess)

    # ------------------------------------------------------------------ #
    # Lecture par pages
    # ------------------------------------------------------------------ #
    def predict_pages(
        self,
        image_path: PathLike,
        *,
        preprocess: bool = True,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
        zones: Optional[bool] = False,
    ) -> list[PageResult]:
        pages = self._preprocessor.read_pages(image_path)
        return [
            self._page_result(page, index, preprocess, rois, scan_barcode, zones)
            for index, page in enumerate(pages, start=1)
        ]

    def predict_pages_bytes(
        self,
        data: bytes | np.ndarray,
        *,
        preprocess: bool = True,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
        zones: Optional[bool] = False,
    ) -> list[PageResult]:
        pages = self._preprocessor.read_pages_bytes(data)
        return [
            self._page_result(page, index, preprocess, rois, scan_barcode, zones)
            for index, page in enumerate(pages, start=1)
        ]

    @_as_ocr_error
    def _page_result(
        self,
        page: np.ndarray,
        index: int,
        preprocess: bool,
        rois: Optional[ROIProfile],
        scan_barcode: Optional[bool],
        zones: Optional[bool] = False,
    ) -> PageResult:
        started = time.perf_counter()
        barcodes: list[BarcodeResult] = []
        if self._barcode_scanner is not None and (
            scan_barcode is None or scan_barcode
        ):
            try:
                barcodes = self._barcode_scanner.scan(page)
            except Exception as exc:
                self.logger.warning("Scan code-barres ignoré: %s", exc)
            if barcodes:
                page = mask_barcodes(page, barcodes)
        image = page
        if preprocess:
            image, _ = self._preprocessor.resize_for_ocr(page, self._max_side_len)
        height, width = image.shape[:2]

        if rois:
            items: list[OCRResultItem] = []
            crops: list[tuple[str, np.ndarray, tuple[int, int]]] = []
            for label, fraction in rois.items():
                x0, y0, x1, y1 = frac_to_px(fraction, width, height)
                crop = image[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                crops.append((label, crop, (x0, y0)))
            if crops:
                batch = self._recognize_crops([crop for _, crop, _ in crops])
                for (label, crop, (x0, y0)), results in zip(crops, batch):
                    for entry in results or []:
                        box = entry.get("box") or [
                            [0, 0],
                            [crop.shape[1], 0],
                            [crop.shape[1], crop.shape[0]],
                            [0, crop.shape[0]],
                        ]
                        entry["box"] = [[x0 + p[0], y0 + p[1]] for p in box]
                        entry["label"] = label
                        items.append(entry)
        elif zones is not False and has_form_structure(page):
            kwargs: dict[str, Any] = {}
            if self._handwritten_reader is not None:
                kwargs["handwritten_reader"] = self._handwritten_reader
                kwargs["handwritten_fields"] = self._handwritten_fields
                if self._band_grid_reader is not None:
                    kwargs["band_grid_reader"] = self._band_grid_reader
            else:
                kwargs["htr_recognize"] = self._htr_recognize
            items = read_exam_form_zones(
                page,
                self._recognize_crop,
                recognize_crops=self._recognize_crops,
                **kwargs,
            )
            image = page
        else:
            items = self._run_ocr(image)

        height, width = image.shape[:2]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return make_page_result(index, width, height, items, elapsed_ms, barcodes, image)


# --------------------------------------------------------------------------- #
# Point d'entrée CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="OCR PaddleOCR local (CPU).")
    parser.add_argument("image", help="Chemin de l'image à analyser (PNG, JPG, TIF…).")
    parser.add_argument("--max-side", type=int, default=0, help="Reduction côté (0 = auto).")
    parser.add_argument("--roi", action="store_true", help="Mode zones d'intérêt (feuille d'examen).")
    parser.add_argument("--roi-json", default=None, help='Profil JSOn (ex. \'{"nom": [0.0,0.0,0.5,0.1]}\').')
    parser.add_argument("--zones", action="store_true", help="Lecture par zones (grilles + pointillés, jamais page entière).")
    parser.add_argument("--no-zones", action="store_true", help="Force la passe pleine page.")
    parser.add_argument("--no-barcode", action="store_true")
    args = parser.parse_args()

    rois: dict[str, tuple[float, float, float, float]] | None
    if args.roi_json is not None:
        try:
            raw = json.loads(args.roi_json)
            rois = {
                str(label): (
                    float(spec[0]),
                    float(spec[1]),
                    float(spec[2]),
                    float(spec[3]),
                )
                for label, spec in raw.items()
                if isinstance(spec, (list, tuple)) and len(spec) == 4
            }
        except json.JSONDecodeError as exc:
            print(f"--roi-json invalide: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        rois = None

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    from .core_ocr import DEFAULT_EXAM_ROIS

    rois = rois or (DEFAULT_EXAM_ROIS if args.roi else None)

    with PaddleOCREngine(max_side_len=args.max_side or None) as engine:
        zones: Optional[bool] = None
        if args.zones:
            zones = True
        elif args.no_zones:
            zones = False
        pages = engine.predict_pages(
            args.image,
            rois=rois,
            scan_barcode=not args.no_barcode,
            zones=zones,
        )
        payload = [
            {key: value for key, value in page.items() if key != "image"}
            for page in pages
        ]
        print(json.dumps({"backend": engine._backend_name, "pages": payload}))
        sys.exit(0)
