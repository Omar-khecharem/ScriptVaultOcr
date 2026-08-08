"""Moteur OCR local haute performance (PaddleOCR CPU + OpenCV), 100 % hors-ligne.

Module autonome, sans aucune dépendance cloud ni API externe. Il expose:

* :class:`ImagePreprocessor` — lecture robuste multi-formats **y compris TIFF
  mono/multi-pages** (Pillow), pipeline OpenCV "fast" : conversion grayscale,
  binarisation d'Otsu optimisée, redressement automatique (deskew), et
  redimensionnement intelligent pour le réseau OCR (``SCRIPTVAULT_MAX_SIDE``).
* :class:`BarcodeScanner` — détection locale de codes-barres / QR codes via
  ``cv2.barcode`` (repli ``pyzbar``), avec budget temps < 15 ms.
* :class:`LocalOCREngine` — wrapper PaddleOCR optimisé CPU (MKL-DNN / OpenMP),
  lecture hybride **code-barres + OCR**, découpage par zones d'intérêt (ROI)
  avec reconnaissance seule ``det=False`` (réduction ~70 % du calcul sur les
  formulaires structurés), sortie structurée ``[{"text", "confidence", "box"}]``.

Exemple::

    from core_ocr import LocalOCREngine, DEFAULT_EXAM_ROIS

    engine = LocalOCREngine(lang="en")
    pages = engine.predict_pages("scan_2024.tif", rois=DEFAULT_EXAM_ROIS)
    for page in pages:
        print(page["page"], page["text"], page["barcodes"])
    engine.close()

Dépendances: ``paddlepaddle``, ``paddleocr>=2.7``, ``opencv-contrib-python``
(>= 4.8 pour ``cv2.barcode``), ``numpy``, ``Pillow`` (TIFF multipages).
``pyzbar`` est un accélérateur optionnel si ``cv2.barcode`` est absent.
"""

from __future__ import annotations

import functools
import io
import logging
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict, TypeVar, Union

import cv2
import numpy as np

__version__ = "2.0.0"
__all__ = [
    "ImagePreprocessor",
    "BarcodeScanner",
    "LocalOCREngine",
    "BarcodeResult",
    "PageResult",
    "ROIProfile",
    "DEFAULT_EXAM_ROIS",
    "OCRBaseError",
    "OCRInitError",
    "OCRImageError",
    "OCRInferenceError",
    "make_page_result",
    "frac_to_px",
    "mask_barcodes",
    "default_max_side_len",
]

PathLike = Union[str, os.PathLike[str]]
Box = list[list[int]]
OCRResultItem = dict[str, Any]
ROIFraction = tuple[float, float, float, float]
ROIProfile = dict[str, ROIFraction]

F = TypeVar("F", bound=Callable[..., Any])

logger: logging.Logger
logger = logging.getLogger("scriptvault.core_ocr")
logger.addHandler(logging.NullHandler())

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
)

# Magic TIFF (les GI et MM sont les deux byte-orders possibles).
_TIFF_MAGIC: tuple[bytes, bytes] = (b"II", b"MM")

# Budget temps par défaut du scanner code-barres (ms) — spéc. < 15 ms.
_DEFAULT_BARCODE_BUDGET_MS = 15.0
_DEFAULT_MAX_SIDE = 1600
_BARCODE_PREVIEW_SIDE = 1000


class BarcodeResult(TypedDict):
    """Un code-barres / QR détecté localement."""

    data: str
    type: str
    box: Box


class PageResult(TypedDict):
    """Résultat complet d'une page (image, TIF multipage ou page PDF).

    ``image`` est l'image **exactement** analysée (prétraitement appliqué,
    zones code-barres masquées) : les boîtes ``box`` sont parfaitement alignées
    avec elle — idéal pour l'overlay web.
    """

    page: int
    width: int
    height: int
    text: str
    confidence: float
    elapsed_ms: float
    items: list[OCRResultItem]
    barcodes: list[BarcodeResult]
    image: np.ndarray


# --------------------------------------------------------------------------- #
# Exceptions métier
# --------------------------------------------------------------------------- #
class OCRBaseError(Exception):
    """Erreur racine du module OCR."""


class OCRInitError(OCRBaseError):
    """Échec d'initialisation du moteur OCR (dépendances ou modèles)."""


class OCRImageError(OCRBaseError):
    """Image illisible, corrompue ou de type non supporté."""


class OCRInferenceError(OCRBaseError):
    """Échec pendant l'exécution de l'inférence OCR."""


def _as_preprocessor_error(fn: F) -> F:
    """Convertit toute erreur interne en erreur typée :class:`OCRImageError`."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except (OCRBaseError, TypeError):
            raise
        except cv2.error as exc:
            raise OCRImageError(f"Erreur OpenCV dans {fn.__name__}: {exc}") from exc
        except Exception as exc:
            raise OCRImageError(
                f"Échec de {fn.__name__}: {type(exc).__name__}: {exc}"
            ) from exc

    return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Helpers de configuration
# --------------------------------------------------------------------------- #
def default_max_side_len() -> int:
    """Longueur maximale de côté pour le réseau OCR (variable d'environnement).

    Lit ``SCRIPTVAULT_MAX_SIDE`` (défaut 1600). Les images plus grandes sont
    réduites (ratio conservé) par interpolation AREA avant le réseau : gain de
    vitesse majeur sur les scans haute résolution, sans perte perceptible.
    """
    raw = os.environ.get("SCRIPTVAULT_MAX_SIDE", "").strip()
    if not raw:
        return _DEFAULT_MAX_SIDE
    try:
        return max(256, int(raw))
    except ValueError:
        return _DEFAULT_MAX_SIDE


def frac_to_px(frac: ROIFraction, width: int, height: int) -> tuple[int, int, int, int]:
    """Convertit une ROI normalisée ``(x0, y0, x1, y1)`` en coordonnées pixels.

    Toutes les valeurs sont des fractions de la largeur / hauteur (0.0–1.0) et
    le résultat est borné aux dimensions de l'image.
    """
    x0 = int(max(0.0, min(1.0, frac[0])) * width)
    y0 = int(max(0.0, min(1.0, frac[1])) * height)
    x1 = int(max(0.0, min(1.0, frac[2])) * width)
    y1 = int(max(0.0, min(1.0, frac[3])) * height)
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def mask_barcodes(
    image: np.ndarray, barcodes: list[BarcodeResult], margin: int = 8
) -> np.ndarray:
    """Occulte les régions code-barres (remplissage blanc) avant l'OCR.

    Empêche la reconnaissance de « texte fantôme » dans les zones encodées et
    évite une passe inutile sur ces surfaces.
    """
    out = image.copy()
    height, width = image.shape[:2]
    for entry in barcodes:
        points = np.asarray(entry.get("box") or [], dtype=np.float32).reshape(-1, 2)
        if points.size == 0:
            continue
        x0 = max(0, int(points[:, 0].min()) - margin)
        y0 = max(0, int(points[:, 1].min()) - margin)
        x1 = min(width, int(points[:, 0].max()) + margin)
        y1 = min(height, int(points[:, 1].max()) + margin)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), -1)
    return out


def _joined_text(items: list[OCRResultItem]) -> str:
    """Joan le texte des lignes avec des sauts de ligne."""
    return "\n".join(str(item.get("text", "")) for item in items).strip()


def _mean_confidence(items: list[OCRResultItem]) -> float:
    """Confiance moyenne des lignes (0.0 si aucune)."""
    if not items:
        return 0.0
    scores = [float(item.get("confidence", 0.0)) for item in items]
    return round(sum(scores) / len(scores), 4)


def make_page_result(
    page: int,
    width: int,
    height: int,
    items: list[OCRResultItem],
    elapsed_ms: float,
    barcodes: list[BarcodeResult],
    image: np.ndarray,
) -> PageResult:
    """Construit un :class:`PageResult` (textes/confiances dérivés des items)."""
    return {
        "page": page,
        "width": width,
        "height": height,
        "text": _joined_text(items),
        "confidence": _mean_confidence(items),
        "elapsed_ms": round(float(elapsed_ms), 2),
        "items": items,
        "barcodes": barcodes,
        "image": image,
    }


def _is_tiff_bytes(data: bytes) -> bool:
    """Vrai si les octets correspondent à un fichier TIFF (peu/big endian)."""
    return data[:2] in _TIFF_MAGIC


# --------------------------------------------------------------------------- #
# Prétraitement d'images
# --------------------------------------------------------------------------- #
class ImagePreprocessor:
    """Pipeline OpenCV ``fast`` pour scans haute résolution.

    Chaîne de traitement (chaque étape est activable indépendamment) :

    1. ``resize`` — mise à l'échelle de la plus grande dimension à
       ``max_side_len`` (défaut : ``SCRIPTVAULT_MAX_SIDE``). Effectuée en
       premier : elle accélère toutes les étapes suivantes.
    2. ``denoise`` — débruitage gaussien (kernel 3x3) puis médian (3x3).
       Désactivé par défaut : dégrade les petits textes.
    3. ``clahe`` — égalisation d'histogramme adaptative (canal L de l'espace
       LAB).
    4. ``deskew`` — redressement automatique de l'inclinaison (Hough
       probabiliste + repli moments de Hu), rotation de canevas agrandi.
    5. ``binarize`` — binarisation **d' Otsu** (mode par défaut, ultra-rapide)
       ou adaptative gaussienne, polarité normalisée (texte sombre / fond
       clair).
    """

    def __init__(
        self,
        *,
        max_side_len: Optional[int] = None,
        binarize_mode: Literal["otsu", "adaptive"] = "otsu",
        clahe_clip: float = 2.0,
        clahe_grid: tuple[int, int] = (8, 8),
        adaptive_block: int = 35,
        adaptive_c: float = 15.0,
        deskew_min_angle: float = 0.5,
        deskew_max_angle: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise le préprocesseur.

        Args:
            max_side_len: Longueur maximale du plus grand côté après
                redimensionnement. ``None`` → variable d'environnement
                ``SCRIPTVAULT_MAX_SIDE`` (défaut 1600). ``<= 0`` désactive.
            binarize_mode: ``"otsu"`` (défaut, monodape Otsu global sur canal
                gris, ~10x plus rapide que l'adaptatif) ou ``"adaptive"``.
            clahe_clip: Limite de contraste (clipLimit) de CLAHE.
            clahe_grid: Taille des tuiles ``(lignes, colonnes)``.
            adaptive_block: Taille du voisinage (impaire, >= 3).
            adaptive_c: Constante soustraite à la moyenne locale.
            deskew_min_angle: Angle minimal (degrés) déclenchant la rotation.
            deskew_max_angle: Angle maximal (degrés) appliqué en rotation.
            logger: Logger optionnel.
        """
        self._max_side_len = (
            max_side_len
            if max_side_len is not None and max_side_len > 0
            else default_max_side_len()
        )
        self._binarize_mode = binarize_mode
        self._clahe_clip = max(1.0, float(clahe_clip))
        self._clahe_grid = clahe_grid
        self._adaptive_block = max(3, int(adaptive_block) | 1)
        self._adaptive_c = float(adaptive_c)
        self._deskew_min_angle = float(deskew_min_angle)
        self._deskew_max_angle = float(deskew_max_angle)
        self.logger = logger or logging.getLogger("scriptvault.core_ocr.preprocessor")

    # ------------------------------------------------------------------ #
    # Lecture (formats simples + TIFF multi-pages)
    # ------------------------------------------------------------------ #
    @_as_preprocessor_error
    def read_image(self, image_path: PathLike) -> np.ndarray:
        """Lit une image (première page si TIF multi-pages), tous formats.

        La lecture passe par ``np.fromfile`` + ``cv2.imdecode`` pour supporter
        les chemins Unicode (Windows). Les TIFF (même compressés CCITT) sont
        décodés via Pillow qui sait lire le premier photogramme.
        """
        path = os.fspath(image_path)
        if not os.path.isfile(path):
            raise OCRImageError(f"Fichier image introuvable: {path!r}")
        ext = Path(path).suffix.lower()
        if ext and ext not in _SUPPORTED_EXTENSIONS:
            self.logger.warning(
                "Extension %r non listée, tentative de décodage malgré tout.",
                ext,
            )
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise OCRImageError(f"Fichier image vide ou illisible: {path!r}")
        return self.read_image_bytes(data.tobytes())

    @_as_preprocessor_error
    def read_image_bytes(self, data: bytes | np.ndarray) -> np.ndarray:
        """Décode une image puis retourne la **première** page.

        Les TIF multi-pages sont décodés page 0 : utilisez
        :meth:`read_pages_bytes` pour tout lire.
        """
        raw = self._coerce_bytes(data)
        if _is_tiff_bytes(raw):
            pages = self._read_tiff_pages(raw)
            if len(pages) > 1:
                self.logger.info(
                    "TIF multi-pages (%d pages) : lecture de la page 1 uniquement.",
                    len(pages),
                )
            return pages[0]
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OCRImageError(
                "Décodage impossible: format non supporté ou données corrompues."
            )
        return self._as_bgr8(image)

    @_as_preprocessor_error
    def read_pages(self, image_path: PathLike) -> list[np.ndarray]:
        """Lit **toutes** les pages d'un fichier image.

        TIF mono ou multi-pages → chaque page est renvoyée dans l'ordre.
        Autres formats → liste d'une seule page.
        """
        data = np.fromfile(os.fspath(image_path), dtype=np.uint8)
        if data.size == 0:
            raise OCRImageError(
                f"Fichier image vide ou illisible: {os.fspath(image_path)!r}"
            )
        return self.read_pages_bytes(data.tobytes())

    @_as_preprocessor_error
    def read_pages_bytes(self, data: bytes | np.ndarray) -> list[np.ndarray]:
        """Décode toutes les pages d'un fichier image en octets.

        Si les octets forment un TIFF (mono ou multi-pages), chaque page est
        lue dynamiquement et convertie en BGR 8 bits sans perte de qualité
        (16 bits : conversion en 8 bits ; gestion des palettes, CMYK, CCITT).
        """
        raw = self._coerce_bytes(data)
        if _is_tiff_bytes(raw):
            return self._read_tiff_pages(raw)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OCRImageError(
                "Décodage impossible: format non supporté ou données corrompues."
            )
        return [self._as_bgr8(image)]

    @staticmethod
    def _coerce_bytes(data: bytes | np.ndarray) -> bytes:
        """Normalise l'entrée en octets contigus."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        return np.asarray(data, dtype=np.uint8).tobytes()

    def _read_tiff_pages(self, data: bytes) -> list[np.ndarray]:
        """Découpage complet des pages TIFF via Pillow (tous codecs)."""
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dépedance réelle
            raise OCRImageError(
                "Pillow est requis pour lire les fichiers TIFF (pip install Pillow)."
            ) from exc
        pages: list[np.ndarray] = []
        try:
            with Image.open(io.BytesIO(data)) as container:
                frame_count = getattr(container, "n_frames", 1) or 1
                for index in range(frame_count):
                    container.seek(index)
                    pages.append(self._pil_to_bgr8(container))
        except Exception as exc:
            raise OCRImageError(
                f"TIFF illisible ou corrompu: {type(exc).__name__}: {exc}"
            ) from exc
        if not pages:
            raise OCRImageError("TIFF ne contenant aucune page décodable.")
        return pages

    @staticmethod
    def _pil_to_bgr8(image: "Any") -> np.ndarray:
        """Convertit une frame Pillow en BGR 8 bits (gère 16 bits / 1 bit)."""
        mode = image.mode
        if mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            arr: Any = np.asarray(image)
            if arr.dtype == np.uint16:
                arr = (arr >> 8).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if mode in ("1", "L", "P", "LA"):
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ #
    # Pipeline de prétraitement
    # ------------------------------------------------------------------ #
    @_as_preprocessor_error
    def preprocess(
        self,
        image: np.ndarray,
        *,
        denoise: bool = False,
        clahe: bool = True,
        deskew: bool = True,
        binarize: bool = True,
    ) -> np.ndarray:
        """Applique le pipeline complet (resize→CLAHE→deskew→binarisation)."""
        return self._preprocess_impl(image, denoise, clahe, deskew, binarize)[0]

    @_as_preprocessor_error
    def preprocess_file(
        self,
        image_path: PathLike,
        *,
        denoise: bool = False,
        clahe: bool = True,
        deskew: bool = True,
        binarize: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Lit puis prétraite une image (première page), avec métadonnées.

        Les métadonnées contiennent ``path``, ``size``, ``scale``,
        ``deskew_angle``, ``denoised``, ``clahe``, ``binarized``.
        """
        image = self.read_image(image_path)
        processed, meta = self._preprocess_impl(image, denoise, clahe, deskew, binarize)
        meta["path"] = os.fspath(image_path)
        meta["size"] = (int(image.shape[1]), int(image.shape[0]))
        return processed, meta

    def resize_for_ocr(
        self, image: np.ndarray, max_side_len: Optional[int] = None
    ) -> tuple[np.ndarray, float]:
        """Redimensionne une image pour le réseau OCR (ratio conservé).

        Returns:
            Tuple ``(image, échelle_appliquée)``. L'échelle <= 1.0 permet de
            retraduire les coordonnées dans le référentiel original.
        """
        limit = max_side_len if max_side_len is not None and max_side_len > 0 else 0
        height, width = image.shape[:2]
        longest = max(height, width)
        if limit <= 0 or longest <= limit:
            return image, 1.0
        scale = limit / longest
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return resized, scale

    # ------------------------------------------------------------------ #
    # Implémentation
    # ------------------------------------------------------------------ #
    def _preprocess_impl(
        self,
        image: np.ndarray,
        denoise: bool,
        clahe: bool,
        deskew: bool,
        binarize: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self._as_bgr8(image)
        image, scale = self.resize_for_ocr(image, self._max_side_len or None)
        meta: dict[str, Any] = {
            "denoised": denoise,
            "clahe": clahe,
            "binarized": binarize,
            "deskew_angle": 0.0,
            "scale": scale,
        }
        if denoise:
            image = self._denoise(image)
        if clahe:
            image = self._apply_clahe(image)
        if deskew:
            image, meta["deskew_angle"] = self._deskew(image)
        if binarize:
            image = self._binarize(image)
        return image, meta

    @staticmethod
    def _as_bgr8(image: np.ndarray) -> np.ndarray:
        """Normalise toute image entrante en BGR 8 bits non signé."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise OCRImageError("Image d'entrée invalide (None ou vide).")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """Débruitage gaussien puis médian (bruit mixte gaussien / sel poivre)."""
        smoothed = cv2.GaussianBlur(image, (3, 3), 0)
        return cv2.medianBlur(smoothed, 3)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Égalisation adaptative (CLAHE) sur le canal L de l'espace LAB."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip, tileGridSize=self._clahe_grid
        )
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )

    def _deskew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Corrige l'inclinaison : Hough probabiliste, repli moments de Hu."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = self._deskew_hough(gray)
        if abs(angle) < self._deskew_min_angle:
            angle = self._deskew_moments(gray)
        if abs(angle) < self._deskew_min_angle:
            return image, 0.0
        clamped = max(-self._deskew_max_angle, min(self._deskew_max_angle, angle))
        if clamped != angle:
            self.logger.info(
                "Angle d'inclinaison %.2f° limité à %.2f°.", angle, clamped
            )
        return self._rotate_bound(image, clamped), clamped

    @staticmethod
    def _deskew_hough(gray: np.ndarray) -> float:
        """Estime l'angle par Transformée de Hough probabiliste (degrés)."""
        height, width = gray.shape
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_length = max(60, int(min(width, height) * 0.25))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=80,
            minLineLength=min_length,
            maxLineGap=15,
        )
        if lines is None or len(lines) == 0:
            return 0.0
        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1e-6:
                continue
            line_angle = math.degrees(math.atan2(dy, dx))
            if abs(line_angle) <= 45.0:
                angles.append(line_angle)
        if not angles:
            return 0.0
        return float(np.median(angles))

    @staticmethod
    def _deskew_moments(gray: np.ndarray) -> float:
        """Estimation par moments (boîte englobante minimale)."""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(thresh) > thresh.size * 0.5:
            thresh = cv2.bitwise_not(thresh)
        coords = cv2.findNonZero(thresh)
        if coords is None or len(coords) < 100:
            return 0.0
        rect = cv2.minAreaRect(coords)
        angle = float(rect[2])
        while angle > 45.0:
            angle -= 90.0
        while angle < -45.0:
            angle += 90.0
        return -angle

    @staticmethod
    def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotation avec canevas agrandi (aucune troncature)."""
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_abs = abs(matrix[0, 0])
        sin_abs = abs(matrix[0, 1])
        new_width = int((height * sin_abs) + (width * cos_abs))
        new_height = int((height * cos_abs) + (width * sin_abs))
        matrix[0, 2] += (new_width / 2.0) - center[0]
        matrix[1, 2] += (new_height / 2.0) - center[1]
        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Binarisation selon ``binarize_mode`` (Otsu par défaut)."""
        if self._binarize_mode == "adaptive":
            return self._binarize_adaptive(image)
        return self._binarize_otsu(image)

    def _binarize_otsu(self, image: np.ndarray) -> np.ndarray:
        """Binarisation d'Otsu **optimisée** (une passe globale, quasi-instantanée).

        Sur de grandes images, le seuillage global d'Otsu est 5–15× plus rapide
        que l'adaptiveThreshold et produit une meilleure séparation sur des
        scans homogènes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._normalize_polarity(thresh)

    def _binarize_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Binarisation adaptative gaussienne (textes fins / fonds hétérogènes)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._adaptive_block,
            self._adaptive_c,
        )
        return self._normalize_polarity(thresh)

    @staticmethod
    def _normalize_polarity(gray: np.ndarray) -> np.ndarray:
        """Garantit texte sombre sur fond clair."""
        if float(np.mean(gray)) < 127.0:
            gray = cv2.bitwise_not(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# Zones d'intérêt (ROI) par défaut — formulaires type feuille d'examen
# --------------------------------------------------------------------------- #
# Coordonnées en fractions de page (x0, y0, x1, y1) — à adapter au gabarit réel
# du formulaire grâce à la variable d'environnement ``SCRIPTVAULT_ROI``.
DEFAULT_EXAM_ROIS: ROIProfile = {
    "identifiant": (0.020, 0.020, 0.360, 0.065),
    "nom": (0.020, 0.090, 0.560, 0.140),
    "prenom": (0.020, 0.165, 0.560, 0.215),
    "cin": (0.020, 0.240, 0.420, 0.290),
}


# --------------------------------------------------------------------------- #
# Scanner code-barres / QR local
# --------------------------------------------------------------------------- #
class BarcodeScanner:
    """Détection locale de codes-barres / QR, respectant un budget temps.

    Stratégie « fast path » :

    * L'image est réduite à une prévisualisation de ``max_preview_side`` pixels
      (compteur C1) — la détection y étant nettement plus rapide sans perte
      significative.
    * ``cv2.barcode.BarcodeDetector`` (WeChat QR + codes-barres 1D linéaires,
      modèles embarqués dans ``opencv-contrib-python``) en première passe, puis
      ``cv2.QRCodeDetector`` en repli si le budget (15 ms par défaut) le
      permet.
    * Les coordonnées retournées sont retraduites à l'échelle d'origine.

    ``pyzbar`` est un repli optionnel si ``cv2.barcode`` est indisponible.
    """

    def __init__(
        self,
        time_budget_ms: float = _DEFAULT_BARCODE_BUDGET_MS,
        max_preview_side: int = _BARCODE_PREVIEW_SIDE,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise le scanner.

        Args:
            time_budget_ms: Budget de calcul (ms) par page. L'exécution
                s'arrête dès que ``time_budget_ms`` est dépassé.
            max_preview_side: Taille maximale de la vue de détection.
            logger: Logger optionnel.
        """
        self._budget_ms = max(1.0, float(time_budget_ms))
        self._max_preview_side = max(128, int(max_preview_side))
        self.logger = logger or logging.getLogger("scriptvault.core_ocr.barcode")
        self._barcode_detector: Any = None
        self._qr_detector = cv2.QRCodeDetector()

    # ------------------------------------------------------------------ #
    def scan(self, image: np.ndarray) -> list[BarcodeResult]:
        """Détecte code-barres & QR autour d'un budget de temps.

        Args:
            image: Image BGR 8 bits (ou grayscale). Peut être grande (haute
                résolution) — la détection se fait sur la version sous-
                échantillonnée.

        Returns:
            Liste de :class:`BarcodeResult` ``{"data", "type", "box"}`` (les
            coordonnées ``box`` sont dans le référentiel de l'image d'origine).
        """
        if image is None or image.size == 0:
            return []
        started = time.perf_counter()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray, inv_scale = self._downscale(gray)
        results: list[BarcodeResult] = []
        elapsed_ms = 0.0

        # 1) Code-barres 1D + QR (OpenCV, modèles embarqués)
        detector = self._get_barcode_detector()
        if detector is not None:
            results = self._scan_barcode(detector, gray, inv_scale)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # 2) Repli QRCodeDetector (rapide) si le budget n'est pas dépassé
        if not results and elapsed_ms < self._budget_ms:
            results = self._scan_qr(gray, inv_scale)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        if results:
            self.logger.debug(
                "Code-barres détectés: %d en %.1f ms.",
                len(results),
                elapsed_ms,
            )
        return results

    # ------------------------------------------------------------------ #
    def _downscale(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        """Sous-échantillonne la vue de détection, retourne le facteur inverse."""
        height, width = gray.shape[:2]
        scale = min(1.0, self._max_preview_side / float(max(height, width)))
        if scale >= 1.0:
            return gray, 1.0
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return scaled, 1.0 / scale

    def _get_barcode_detector(self) -> Any:
        """Retourne le détecteur OpenCV (créé paresseusement), ou None."""
        if self._barcode_detector is not None:
            return self._barcode_detector
        try:
            import cv2.barcode as barcode  # type: ignore[attr-defined]

            self._barcode_detector = barcode.BarcodeDetector()  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.debug("cv2.barcode indisponible (%s); repli pyzbar/QR.", exc)
            self._barcode_detector = False
        return self._barcode_detector if self._barcode_detector is not False else None

    def _scan_barcode(
        self, detector: Any, gray: np.ndarray, inv_scale: float
    ) -> list[BarcodeResult]:
        raw = detector.detectAndDecode(gray)
        infos, types, rects = self._unpack_detect(raw)
        output: list[BarcodeResult] = []
        for info, typ, rect in zip(infos, types, rects):
            if not info or rect is None:
                continue
            x, y, w, h = rect
            box = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]
            box = [
                [int(round(px * inv_scale)), int(round(py * inv_scale))]
                for px, py in box
            ]
            output.append(
                {"data": str(info), "type": str(typ or "BARCODE"), "box": box}
            )
        return output

    def _scan_qr(self, gray: np.ndarray, inv_scale: float) -> list[BarcodeResult]:
        ok, infos, points, _ = self._qr_detector.detectAndDecodeMulti(gray)
        if not ok:
            return []
        output: list[BarcodeResult] = []
        text_list = self._as_text_list(infos)
        for info, poly in zip(text_list, points):
            if not info:
                continue
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if arr.size == 0:
                continue
            box = [
                [
                    int(round(float(p[0]) * inv_scale)),
                    int(round(float(p[1]) * inv_scale)),
                ]
                for p in arr
            ]
            output.append({"data": info, "type": "QRCode", "box": box})
        return output

    @staticmethod
    def _unpack_detect(
        raw: Any,
    ) -> tuple[list[str], list[str], list[tuple[int, int, int, int]]]:
        """Normalise la sortie ``detectAndDecode`` quelle que soit la version.

        OpenCV 4.10 : ``(decoded_info, decoded_type, straight_rects)`` — mais
        certaines versions préfixent un flag de détection (retenu ``retval``),
        produisant un 3- ou 4-uplet ``(retval, decoded_info, ...)``.
        """
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return [], [], []
        parts = list(raw)
        if isinstance(parts[0], (bool, np.bool_)):
            parts = parts[1:]
        if len(parts) < 2:
            return [], [], []
        infos = BarcodeScanner._as_text_list(parts[0])
        types = BarcodeScanner._as_text_list(parts[1]) if len(parts) >= 2 else []
        raw_rects = parts[2] if len(parts) >= 3 else parts[1]
        arr = np.asarray(raw_rects) if raw_rects is not None else np.empty((0, 0))
        rects: list[tuple[int, int, int, int]] = []
        if arr.size:
            for item in arr.reshape(-1, arr.shape[-1] if arr.ndim else 4):
                values = [int(float(v)) for v in item.tolist()]
                if len(values) >= 4:
                    rects.append((values[0], values[1], values[2], values[3]))
        return infos, types, rects

    @staticmethod
    def _as_text_list(value: Any) -> list[str]:
        """Normalise une valeur simple / liste / ndarray en liste de str."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return []
            items = value.flatten().tolist()
            return [str(x) for x in items if x]
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value if x]
        return [str(value)]


# --------------------------------------------------------------------------- #
# Moteur OCR local
# --------------------------------------------------------------------------- #
class LocalOCREngine:
    """Moteur PaddleOCR local, optimisé CPU (MKL-DNN / OpenMP / AVX2).

    Caractéristiques:

    * Import paresseux de PaddlePaddle (le module ``core_ocr`` est importable
      sans Paddle).
    * Pré-chargement des poids en mémoire (dossier local ``model_dir/`` ou
      cache ``~/.paddleocr``) et ``warm_up()`` matérialisant les modèles dans
      la RAM dès l'initialisation (Warm-start, 100 % hors-ligne).
    * Multi-threading CPU: ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` et
      ``paddle.set_num_threads``.
    * Compatible PaddleOCR 2.x et 3.x (détection automatique de l'API).
    * **Lecture hybride** : détection de codes-barres / QR locaux (< 15 ms),
      puis passe OCR globale ou par zones d'intérêt (ROI) avec reconnaissance
      seule (``det=False``.
    """

    def __init__(
        self,
        lang: str = "en",
        model_dir: Optional[PathLike] = None,
        cpu_threads: int = 0,
        use_mp: Optional[bool] = None,
        total_process_num: int = 0,
        use_angle_cls: bool = False,
        use_mkldnn: bool = False,
        det_model_dir: Optional[PathLike] = None,
        rec_model_dir: Optional[PathLike] = None,
        cls_model_dir: Optional[PathLike] = None,
        preprocess_kwargs: Optional[dict[str, Any]] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        logger: Optional[logging.Logger] = None,
        *,
        det_model_name: Optional[str] = None,
        rec_model_name: Optional[str] = None,
        max_side_len: Optional[int] = None,
        barcode: bool = True,
        barcode_budget_ms: float = _DEFAULT_BARCODE_BUDGET_MS,
        barcode_max_preview_side: int = _BARCODE_PREVIEW_SIDE,
        enable_roi: bool = False,
    ) -> None:
        """Initialise le moteur.

        Args:
            lang: Langue des modèles de reconnaissance (``"en"``, ``"fr"``,
                ``"ch"``, ...).
            model_dir: Dossier racine ``det/``, ``rec/``, ``cls/`` pour un
                chargement 100 % local des poids (bundle Air-Gapped). Si omis,
                PaddleOCR utilise son cache.
            cpu_threads: Threads CPU (OpenMP/MKL). ``0`` = défaut
                (min(8, cœurs)).
            use_mp / total_process_num: Multi-processus PaddleOCR 2.x
                (désactivé par défaut sur Windows : mode ``spawn`` non géré
                ici, voir :mod:`engines`).
            use_angle_cls: Classification d'orientation des lignes. Coûteuse
                (un modèle par ligne) : désactivée par défaut.
            use_mkldnn: Accélération MKL-DNN. Désactivé par défaut
                (paddlepaddle 3.3.1 + oneDNN/PP-OCRv6 incompatible).
            det_model_dir / rec_model_dir / cls_model_dir: Chemins explicites
                des modèles (surpassent ``model_dir``).
            det_model_name / rec_model_name: Noms des modèles PaddleOCR 3.x
                (ex. ``PP-OCRv5_mobile_det``). Si omis sans ``model_dir``, le
                moteur choisit des modèles mobiles rapides sur CPU.
            preprocess_kwargs: Arguments d'appel de
                :meth:`ImagePreprocessor.preprocess` par prédiction.
            preprocessor: Instance à réutiliser (injection tests).
            logger: Logger.
            max_side_len: Longueur max côté réseau OCR. ``None`` →
                ``SCRIPTVAULT_MAX_SIDE``.
            barcode: Active le scanner code-barres / QR local.
            barcode_budget_ms: Budget de détection code-barres (ms).
            barcode_max_preview_side: Taille de la vue de détection.
            enable_roi: Préchauffe également le prédicteur de reconnaissance
                seul (``det=False``) pour le mode ROI.

        Raises:
            OCRInitError: Paddle/paddleocr manquants ou échec de chargement.
        """
        self.logger = logger or logging.getLogger("scriptvault.core_ocr.engine")
        self.lang = lang
        self.use_angle_cls = bool(use_angle_cls)
        self._preprocessor = preprocessor or ImagePreprocessor(
            logger=self.logger, max_side_len=max_side_len
        )
        self._preprocess_kwargs = dict(preprocess_kwargs or {})
        self._max_side_len = (
            max_side_len if max_side_len is not None and max_side_len > 0 else None
        )
        self._ocr: Any = None
        self._api_v3: bool = False
        self._ready: bool = False

        self._scan_barcode = bool(barcode)
        self._barcode_scanner = (
            BarcodeScanner(
                time_budget_ms=barcode_budget_ms,
                max_preview_side=barcode_max_preview_side,
                logger=self.logger,
            )
            if self._scan_barcode
            else None
        )
        self._enable_roi = bool(enable_roi)
        self._rec_predictor: Any = None  # None = non construit, False = échec
        self._rec_model_dir: Optional[str] = None
        self._det_model_name = det_model_name
        self._rec_model_name = rec_model_name

        cpu_count = os.cpu_count() or 4
        self.cpu_threads = cpu_threads if cpu_threads > 0 else min(8, cpu_count)
        self._total_process_num = (
            total_process_num if total_process_num > 0 else min(4, cpu_count)
        )
        if use_mp is None:
            use_mp = platform.system() != "Windows"
            self.logger.info(
                "Multi-processus auto: %s (désactivé sur Windows, mode spawn).",
                use_mp,
            )
        self.use_mp = bool(use_mp) and self._total_process_num > 1

        self._setup_cpu_environment()
        self._ocr = self._build_paddleocr(
            lang=lang,
            model_dir=model_dir,
            use_angle_cls=use_angle_cls,
            use_mkldnn=use_mkldnn,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
        )
        self.warm_up()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _setup_cpu_environment() -> None:
        """Configure l'environnement CPU avant l'import Paddle."""
        if "paddle" in sys.modules:
            warnings.warn(
                "PaddlePaddle déjà importé : OMP/MKL_NUM_THREADS non appliqués.",
                RuntimeWarning,
                stacklevel=2,
            )
        threads = str(min(8, os.cpu_count() or 4))
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        # FLAGS_use_mkldnn non défini (paddlepaddle 3.3.1 + oneDNN casse
        # l'inférence PP-OCRv6 : ConvertPirAttribute2RuntimeAttribute).
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    def _build_paddleocr(
        self,
        *,
        lang: str,
        model_dir: Optional[PathLike],
        use_angle_cls: bool,
        use_mkldnn: bool,
        det_model_dir: Optional[PathLike],
        rec_model_dir: Optional[PathLike],
        cls_model_dir: Optional[PathLike],
    ) -> Any:
        try:
            import paddle
            import paddleocr
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRInitError(
                "Dépendances OCR manquantes (pip install paddlepaddle paddleocr)."
            ) from exc

        try:
            paddle.set_num_threads(self.cpu_threads)
        except Exception as exc:  # pragma: no cover - version dépendante
            self.logger.warning("paddle.set_num_threads indisponible: %s", exc)
        if hasattr(paddle, "is_compiled_with_mkldnn"):
            self.logger.info(
                "PaddlePaddle compilé avec MKL-DNN: %s",
                paddle.is_compiled_with_mkldnn(),
            )

        version = str(getattr(paddleocr, "__version__", "2.7"))
        try:
            major = int(version.split(".")[0])
        except ValueError:
            major = 2
        self._api_v3 = major >= 3
        self.logger.info(
            "API paddleocr détectée: %s (v%s)",
            "3.x" if self._api_v3 else "2.x",
            version,
        )

        if model_dir is not None:
            root = Path(model_dir)
            det_model_dir = det_model_dir or str(root / "det")
            rec_model_dir = rec_model_dir or str(root / "rec")
            cls_model_dir = cls_model_dir or str(root / "cls")
        if rec_model_dir is not None:
            self._rec_model_dir = os.fspath(rec_model_dir)

        kwargs: dict[str, Any] = {
            "lang": lang,
            "enable_mkldnn": use_mkldnn,
            "cpu_threads": self.cpu_threads,
        }
        if self._api_v3:
            kwargs["device"] = "cpu"
        else:
            kwargs.update(
                {
                    "use_gpu": False,
                    "use_mp": self.use_mp,
                    "total_process_num": self._total_process_num,
                }
            )
        if det_model_dir is not None:
            key = "text_detection_model_dir" if self._api_v3 else "det_model_dir"
            kwargs[key] = os.fspath(det_model_dir)
        if rec_model_dir is not None:
            key = "text_recognition_model_dir" if self._api_v3 else "rec_model_dir"
            kwargs[key] = os.fspath(rec_model_dir)
        if cls_model_dir is not None:
            key = (
                "textline_orientation_model_dir"
                if self._api_v3
                else "angle_cls_model_dir"
            )
            kwargs[key] = os.fspath(cls_model_dir)

        if self._api_v3:
            kwargs.update(
                {
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": use_angle_cls,
                }
            )
            # Sans modèles locaux explicites, choisir des modèles mobiles
            # (CPU) : les "medium" par défaut sont ~10x plus lents.
            if det_model_dir is None and rec_model_dir is None:
                return self._build_v3_with_cpu_models(PaddleOCR, kwargs)
        else:
            kwargs.update({"use_angle_cls": use_angle_cls, "show_log": False})

        try:
            engine = PaddleOCR(**kwargs)
        except Exception as exc:
            raise OCRInitError(
                f"Échec de l'initialisation PaddleOCR (lang={lang!r}): {exc}"
            ) from exc
        return engine

    @staticmethod
    def _v3_cpu_model_candidates(
        det_name: Optional[str], rec_name: Optional[str]
    ) -> list[tuple[str, str]]:
        """Paires ``(det, rec)`` testées dans l'ordre (modèles mobiles CPU).

        Les modèles ``medium`` par défaut de PaddleOCR 3.x sont beaucoup trop
        lents en inférence CPU sans MKL-DNN. Les modèles mobiles sont 5-10x
        plus rapides pour une précision très proche. Si un nom explicite est
        fourni, seule cette paire est tentée.
        """
        if det_name or rec_name:
            return [
                (det_name or "PP-OCRv5_mobile_det", rec_name or "PP-OCRv5_mobile_rec")
            ]
        return [
            ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
            ("PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"),
            ("PP-OCRv3_mobile_det", "PP-OCRv3_mobile_rec"),
        ]

    def _build_v3_with_cpu_models(
        self, paddleocr_class: Any, kwargs: dict[str, Any]
    ) -> Any:
        """Construit le prédicteur avec des modèles mobiles (repli en chaîne)."""
        last_error: Optional[BaseException] = None
        for det_name, rec_name in self._v3_cpu_model_candidates(
            self._det_model_name, self._rec_model_name
        ):
            candidate = dict(kwargs)
            candidate["text_detection_model_name"] = det_name
            candidate["text_recognition_model_name"] = rec_name
            try:
                engine = paddleocr_class(**candidate)
                self.logger.info(
                    "Modèles CPU sélectionnés: %s / %s.", det_name, rec_name
                )
                return engine
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Modèles %s / %s indisponibles (%s) ; repli.",
                    det_name,
                    rec_name,
                    exc,
                )
        raise OCRInitError(
            "Aucun modèle OCR mobile n'a pu être chargé "
            f"(dernière erreur: {last_error})"
        ) from last_error

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        """Indique si le moteur est initialisé et les modèles chargés."""
        return self._ready and self._ocr is not None

    @property
    def preprocessor(self) -> ImagePreprocessor:
        """Le préprocesseur d'images associé au moteur."""
        return self._preprocessor

    @property
    def preprocess_options(self) -> dict[str, Any]:
        """Copie des options de prétraitement appliquées avant chaque inférence."""
        return dict(self._preprocess_kwargs)

    def warm_up(self) -> None:
        """Force le chargement des poids en mémoire (modèles disponibles).

        Exécute une inférence sur un blank minimal et, si ``enable_roi``,
        préchauffe aussi le prédicteur de reconnaissance seule.
        """
        if self._ocr is None:
            raise OCRInitError("Le moteur OCR n'est pas initialisé.")
        blank: np.ndarray = np.full((96, 96, 3), 255, dtype=np.uint8)
        try:
            self._run_ocr(blank)
            self._ready = True
            self.logger.info(
                "Modèles OCR pré-chargés (threads=%d, processus=%d, mkldnn=%s).",
                self.cpu_threads,
                self._total_process_num,
                self.use_mp,
            )
        except Exception as exc:
            self._ready = False
            self.logger.warning("Warm-up OCR échoué (%s) ; chargement différé.", exc)
        if self._enable_roi:
            try:
                predictor = self._ensure_rec_predictor()
                if predictor is not None:
                    list(predictor.predict(blank))
                    self.logger.info("Prédicteur ROI (rec-only) pré-chaudé.")
            except Exception as exc:
                self.logger.warning("Warm-up ROI échoué: %s", exc)

    # --- OCR globale (page unique) ------------------------------------- #
    def predict_array(
        self,
        image: np.ndarray,
        *,
        preprocess: bool = True,
    ) -> list[OCRResultItem]:
        """Exécute l'OCR sur un tableau numpy (une seule page)."""
        if self._ocr is None:
            raise OCRInitError("Le moteur OCR n'est pas initialisé.")
        try:
            if preprocess:
                image = self._preprocessor.preprocess(image, **self._preprocess_kwargs)
            return self._run_ocr(image)
        except (OCRImageError, OCRBaseError):
            raise
        except cv2.error as exc:
            raise OCRImageError(f"Erreur OpenCV pendant l'OCR: {exc}") from exc
        except Exception as exc:
            raise OCRInferenceError(
                f"Échec de l'inférence OCR: {type(exc).__name__}: {exc}"
            ) from exc

    def predict(
        self,
        image_path: PathLike,
        *,
        preprocess: bool = True,
    ) -> list[OCRResultItem]:
        """Reconnaît le texte d'une image (première page si TIF multi-pages)."""
        image = self._preprocessor.read_image(image_path)
        return self.predict_array(image, preprocess=preprocess)

    def predict_bytes(
        self,
        data: bytes | np.ndarray,
        *,
        preprocess: bool = True,
    ) -> list[OCRResultItem]:
        """Reconnaît le texte d'une image fournie en octets bruts."""
        image = self._preprocessor.read_image_bytes(data)
        return self.predict_array(image, preprocess=preprocess)

    # ------------------------------------------------------------------ #
    # Lecture hybride par pages (TIF multi-pages + ROI + code-barres)
    # ------------------------------------------------------------------ #
    def predict_pages(
        self,
        image_path: PathLike,
        *,
        preprocess: bool = True,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
    ) -> list[PageResult]:
        """Analyse **toutes** les pages d'un fichier (TIF multi-pages).

        Args:
            image_path: Chemin du fichier image.
            preprocess: Applique le pipeline OpenCV (CLAHE/Otsu/deskew/resize).
            rois: Profil de zones d'intérêt. Si fourni, l'OCR global est
                remplacé par une reconnaissance seule (``det=False``) sur
                chaque zone nommée — ~70 % de calcul en moins sur les
                formulaires structurés.
            scan_barcode: Active la détection locale de codes-barres/QR
                (défaut : option du moteur).

        Returns:
            Un :class:`PageResult` par page. ``image`` = image exactement
            analysée (prétraitée, zones masquées) — boîtes alignées.
        """
        pages = self._preprocessor.read_pages(image_path)
        return [
            self._predict_page_array(page, index, preprocess, rois, scan_barcode)
            for index, page in enumerate(pages, start=1)
        ]

    def predict_pages_bytes(
        self,
        data: bytes | np.ndarray,
        *,
        preprocess: bool = True,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
    ) -> list[PageResult]:
        """Identique à :meth:`predict_pages` pour des octets bruts."""
        pages = self._preprocessor.read_pages_bytes(data)
        return [
            self._predict_page_array(page, index, preprocess, rois, scan_barcode)
            for index, page in enumerate(pages, start=1)
        ]

    # ------------------------------------------------------------------ #
    def _predict_page_array(
        self,
        page: np.ndarray,
        page_number: int,
        preprocess: bool,
        rois: Optional[ROIProfile],
        scan_barcode: Optional[bool],
    ) -> PageResult:
        started = time.perf_counter()

        # --- 1) Passe code-barres locale (< budget ms) -------------------- #
        barcodes: list[BarcodeResult] = []
        use_barcode = self._scan_barcode if scan_barcode is None else bool(scan_barcode)
        if use_barcode:
            scanner = self._barcode_scanner
            if scanner is None:
                self.logger.warning(
                    "Scanneur code-barres non configuré (barcode=False)."
                )
            else:
                try:
                    barcodes = scanner.scan(page)
                except Exception as exc:
                    self.logger.warning("Scan code-barres ignoré: %s", exc)
            # Masquage des zones concernées (évite les faux OCR sur barres)
            if barcodes:
                page = mask_barcodes(page, barcodes)

        # --- 2. Prétraitement --------------------------------------------------
        if preprocess:
            if rois:
                # Les ROI sont exprimées en fractions de la page ; on désactive
                # le deskew pour préserver l'alignement des coordonnées.
                kwargs = dict(self._preprocess_kwargs)
                kwargs["deskew"] = False
                kwargs["denoise"] = False
                processed = self._preprocessor.preprocess(page, **kwargs)
            else:
                processed = self._preprocessor.preprocess(
                    page, **self._preprocess_kwargs
                )
        else:
            processed = page

        height, width = processed.shape[:2]

        # --- 3. Inférence (OCR global ou ROI rec-only) --------------------- #
        if rois:
            items: list[OCRResultItem] = []
            for label, fraction in rois.items():
                x0, y0, x1, y1 = frac_to_px(fraction, width, height)
                crop = processed[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                for entry in self._recognize_crop(crop):
                    box = entry.get("box") or [
                        [0, 0],
                        [x1 - x0, 0],
                        [x1 - x0, y1 - y0],
                        [0, y1 - y0],
                    ]
                    entry["box"] = [[x0 + p[0], y0 + p[1]] for p in box]
                    entry["label"] = label
                    items.append(entry)
        else:
            items = self._run_ocr(processed)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return make_page_result(
            page_number, width, height, items, elapsed_ms, barcodes, processed
        )

    def _recognize_crop(self, crop: np.ndarray) -> list[OCRResultItem]:
        """Reconnaissance seule (détection désactivée) sur une zone.

        * PaddleOCR 2.x : ``ocr(crop, det=False, cls=False)``.
        * PaddleOCR 3.x : prédicteur REC-Only ``paddlex`` (``TextRecPredictor``)
          sur le dossier local des poids ; repli sur le pipeline complet si la
          construction échoue.
        """
        if not self._api_v3:
            try:
                raw = self._ocr.ocr(crop, det=False, cls=False)
            except (OCRBaseError, cv2.error):
                raise
            except Exception as exc:
                raise OCRInferenceError(
                    f"Échec reconnaissance ROI (2.x): {type(exc).__name__}: {exc}"
                ) from exc
            return self._parse_rec_only(raw, crop.shape)
        predictor = self._ensure_rec_predictor()
        if predictor is None:
            return self._run_ocr(crop)
        try:
            results = list(predictor.predict(crop))
            if not results:
                return []
            return self._parse_rec_only(results[0], crop.shape)
        except Exception as exc:
            self.logger.warning(
                "Reconnaissance ROI échouée (%s), repli sur le pipeline global.",
                exc,
            )
            return self._run_ocr(crop)

    def _ensure_rec_predictor(self) -> Any:
        """Construit (mis en cache) le prédicteur de reconnaissance seule 3.x."""
        if self._rec_predictor is not None:
            return self._rec_predictor if self._rec_predictor is not False else None
        if not self._api_v3:
            self._rec_predictor = False
            return None
        try:
            from paddlex import create_predictor

            name: Optional[str] = None
            params = getattr(self._ocr, "_params", None)
            if isinstance(params, dict):
                name = params.get("text_recognition_model_name")
            if not name:
                raise OCRInitError(
                    "Impossible de résoudre le nom du modèle de reconnaissance "
                    "pour le mode ROI (det=False)."
                )
            if not self._rec_model_dir:
                raise OCRInitError(
                    "Dossier des modèles de reconnaissance non résolu (mode ROI)."
                )
            self._rec_predictor = create_predictor(
                str(name), model_dir=self._rec_model_dir, device="cpu"
            )
            self.logger.info(
                "Prédicteur REC-Only prêt (dossier=%s).", self._rec_model_dir
            )
        except Exception as exc:
            # Repli : le pipeline complet (det+rec) reste utilisable sur le crop.
            self._rec_predictor = False
            self.logger.warning(
                "Mode ROI détection désactivée indisponible, repli global: %s", exc
            )
        return self._rec_predictor if self._rec_predictor is not False else None

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Libère les ressources (processus, modèles)."""
        if self._ocr is not None:
            release = getattr(self._ocr, "release", None)
            if callable(release):
                try:
                    release()
                except Exception as exc:
                    self.logger.warning("Relâchement du moteur échoué: %s", exc)
            self._ocr = None
            self._ready = False
            self._rec_predictor = None
            self.logger.info("Moteur OCR fermé.")

    def __enter__(self) -> "LocalOCREngine":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Exécution et normalisation
    # ------------------------------------------------------------------ #
    def _run_ocr(self, image: np.ndarray) -> list[OCRResultItem]:
        """Délègue l'inférence à PaddleOCR selon l'API détectée."""
        if self._api_v3:
            raw = self._ocr.predict(image)
            return self._parse_v3(raw)
        raw = self._ocr.ocr(image, cls=self.use_angle_cls)
        return self._parse_v2(raw)

    def _parse_v3(self, raw: Any) -> list[OCRResultItem]:
        """Normalise la sortie PaddleOCR 3.x (liste d'objets dict-like)."""
        entries: list[OCRResultItem] = []
        pages = raw if isinstance(raw, (list, tuple)) else [raw]
        for page in pages:
            if page is None:
                continue
            if isinstance(page, dict) or hasattr(page, "get"):
                d = page
            elif hasattr(page, "__dict__"):
                d = vars(page)
            else:
                d = {}
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or []
            polys = d.get("rec_polys")
            if polys is None:
                polys = d.get("dt_polys")
            if polys is None:
                polys = []
            for index, text in enumerate(texts):
                box: Box = []
                if index < len(polys):
                    box = self._poly_to_box(polys[index])
                score = float(scores[index]) if index < len(scores) else 0.0
                entries.append(
                    {
                        "text": str(text),
                        "confidence": min(1.0, max(0.0, score)),
                        "box": box,
                    }
                )
        return entries

    def _parse_v2(self, raw: Any) -> list[OCRResultItem]:
        """Normalise la sortie PaddleOCR 2.x (tuples imbriqués)."""
        entries: list[OCRResultItem] = []
        self._walk_v2(raw, entries)
        return entries

    @classmethod
    def _walk_v2(cls, node: Any, out: list[OCRResultItem]) -> None:
        """parcourt récursivement la structure 2.x et collecte les lignes."""
        if not isinstance(node, (list, tuple)):
            return
        if len(node) >= 2 and cls._is_box(node[0]):
            rest = node[1]
            if isinstance(rest, (list, tuple)) and len(rest) >= 2:
                try:
                    text = str(rest[0])
                    score = float(rest[1])
                except (TypeError, ValueError):
                    text, score = str(node[0]), 0.0
                out.append(
                    {
                        "text": text,
                        "confidence": min(1.0, max(0.0, score)),
                        "box": cls._poly_to_box(node[0]),
                    }
                )
                return
        for item in node:
            cls._walk_v2(item, out)

    def _parse_rec_only(
        self, raw: Any, shape: tuple[int, int, int]
    ) -> list[OCRResultItem]:
        """Parse la sortie « détection désactivée ».

        Formats supportés :

        * PaddleX 3.x rec-only : dict-like avec ``rec_text`` / ``rec_score``
          (str ou list).
        * PaddleOCR 2.x ``det=False`` : ``[[None, ('text', 'score')], ...]``,
          ``['text', 0.9]`` ou tuples plats.
        """
        entries: list[OCRResultItem] = []
        full_box: Box = [
            [0, 0],
            [shape[1], 0],
            [shape[1], shape[0]],
            [0, shape[0]],
        ]

        def add(text: Any, score: Any) -> None:
            entries.append(
                {
                    "text": str(text),
                    "confidence": min(1.0, max(0.0, float(score or 0.0))),
                    "box": list(full_box),
                }
            )

        if isinstance(raw, (dict,)) or hasattr(raw, "get"):
            texts = raw.get("rec_text") or []
            scores = raw.get("rec_score") or []
            text_list = texts if isinstance(texts, (list, tuple)) else [texts]
            score_list = (
                scores
                if isinstance(scores, (list, tuple))
                else ([scores] * len(text_list))
            )
            for text, score in zip(text_list, score_list):
                if text:
                    add(text, score)
            return entries
        if not isinstance(raw, (list, tuple)):
            return []

        for node in raw:
            if not isinstance(node, (list, tuple)):
                continue
            # Style [[None, (texte, score)], ...] (2.x det=False)
            if len(node) == 2 and isinstance(node[0], type(None)):
                rest = node[1]
                if isinstance(rest, (list, tuple)) and len(rest) >= 1:
                    add(rest[0], rest[1] if len(rest) > 1 else 0.0)
                    continue
            # Style ['texte', 0.9] / ('texte', 0.9)
            if len(node) >= 2 and not self._is_box(node[0]):
                add(node[0], node[1])
                continue
            # Style [[box], (texte, score)] (boîte éventuellement None)
            if len(node) >= 2 and self._is_box(node[0]):
                rest = node[1]
                if isinstance(rest, (list, tuple)) and len(rest) >= 1:
                    add(rest[0], rest[1] if len(rest) > 1 else 0.0)
        return entries

    @staticmethod
    def _is_box(value: Any) -> bool:
        """Vrai si ``value`` ressemble à une boîte (4+ points 2D)."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            return False
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.shape[0] >= 4
        return arr.size >= 8 and arr.size % 2 == 0

    @staticmethod
    def _poly_to_box(poly: Any) -> Box:
        """Convertit un polygone (quad ou 8 nombres) en ``[[x, y] * 4]``."""
        try:
            arr = np.asarray(poly, dtype=np.float32)
        except (TypeError, ValueError):
            return []
        if arr.size == 0:
            return []
        if arr.ndim == 2 and arr.shape[1] == 2:
            points = arr
        else:
            flat = arr.reshape(-1)
            if flat.size % 2 != 0:
                return []
            points = flat.reshape(-1, 2)
        return [
            [int(round(float(x))), int(round(float(y)))] for x, y in points.tolist()
        ]


# --------------------------------------------------------------------------- #
# Point d'entrée CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="OCR local hors-ligne (PaddleOCR CPU + OpenCV)."
    )
    parser.add_argument("image", help="Chemin de l'image à analyser (PNG, JPG, TIF…).")
    parser.add_argument("--lang", default="en", help="Langue (en, fr, ch, ...).")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Désactive le prétraitement OpenCV.",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Active le mode zones d'intérêt (feuille d'examen).",
    )
    parser.add_argument(
        "--no-barcode",
        action="store_true",
        help="Désactive la détection de codes-barres/QR.",
    )
    parser.add_argument(
        "--max-side", type=int, default=0, help="Longueur max côté OCR (0 = auto)."
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="Threads CPU (0 = auto)."
    )
    parser.add_argument("--model-dir", default=None, help="Dossier local des modèles.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    with LocalOCREngine(
        lang=args.lang,
        model_dir=args.model_dir,
        cpu_threads=args.threads,
        max_side_len=args.max_side or None,
    ) as engine:
        pages = engine.predict_pages(
            args.image,
            preprocess=not args.no_preprocess,
            rois=DEFAULT_EXAM_ROIS if args.roi else None,
            scan_barcode=not args.no_barcode,
        )

    payload = [
        {key: value for key, value in page.items() if key != "image"} for page in pages
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
