"""Moteur HTR local haute performance (TrOCR ONNX + OpenCV), 100 % hors-ligne.

Module autonome, sans aucune dépendance cloud ni API externe. Il expose :

* :class:`ImagePreprocessor` — lecture robuste multi-formats **y compris TIFF
  mono/multi-pages** (Pillow), pipeline OpenCV : conversion grayscale,
  binarisation d'Otsu, redressement automatique (deskew) et redimensionnement
  intelligent (``SCRIPTVAULT_MAX_SIDE``).
* :class:`BarcodeScanner` — détection locale de codes-barres / QR codes via
  ``cv2.barcode`` (repli ``cv2.QRCodeDetector``), avec budget temps < 15 ms.
* :class:`LocalOCREngine` — transcription manuscrite par TrOCR-small
  (ONNX Runtime quantifié int8) : détection de lignes OpenCV, reconnaissance
  par zone d'intérêt (ROI), sortie structurée
  ``[{"text", "confidence", "box"}]``.

Le moteur est thread-safe : les sessions ONNX supportent les appels
simultanés et sont chargées une seule fois au démarrage (``warm_up``).

Exemple::

    from core_ocr import LocalOCREngine, DEFAULT_EXAM_ROIS

    engine = LocalOCREngine()
    pages = engine.predict_pages("scan_2024.tif", rois=DEFAULT_EXAM_ROIS)
    for page in pages:
        print(page["page"], page["text"], page["barcodes"])
    engine.close()

Dépendances: ``onnxruntime``, ``opencv-contrib-python`` (>= 4.8 pour
``cv2.barcode``), ``numpy``, ``Pillow`` (TIFF multipages). Les poids TrOCR
quantifiés sont dans ``models/trocr/`` (``download_trocr_models.py``).
"""

from __future__ import annotations

import functools
import io
import logging
import math
import os
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import cv2
import numpy as np

if TYPE_CHECKING:
    from .htr_engine import TrOcrEngine

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
        except ImportError as exc:  # pragma: no cover - dépendance réelle
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
        smoothed = cv2.GaussianBlur(image, (1, 1), 0)
        return cv2.medianBlur(smoothed, 3)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Égalisation adaptative (CLAHE) sur le canal L de l'espace LAB."""
        return self._apply_clahe_strength(image, clip=self._clahe_clip)

    def _apply_clahe_strength(self, image: np.ndarray, *, clip: float) -> np.ndarray:
        """CLAHE avec force configurable (``clip`` élevé = contraste local plus
        marqué, adapté aux zones manuscrites pâles)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=max(1.0, float(clip)), tileGridSize=(8, 8))
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
        """Estimation par Hough probabiliste (lignes longues, angle dominant)."""
        _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(np.mean(edges)) > 127.0:
            edges = cv2.bitwise_not(edges)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360.0,
            threshold=180,
            minLineLength=max(32, gray.shape[1] // 12),
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

    def _binarize_adaptive(
        self, image: np.ndarray, *, block: Optional[int] = None, c: Optional[float] = None
    ) -> np.ndarray:
        """Binarisation adaptative gaussienne (textes fins / fonds hétérogènes).

        ``block``/``c`` surchargent les valeurs configurées (utile pour les
        zones manuscrites aux traits fins).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block_size = block if block and block > 0 else self._adaptive_block
        c_value = c if c is not None else self._adaptive_c
        block_size = max(3, int(block_size) | 1)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            float(c_value),
        )
        return self._normalize_polarity(thresh)

    @staticmethod
    def _normalize_polarity(gray: np.ndarray) -> np.ndarray:
        """Garantit texte sombre sur fond clair."""
        if float(np.mean(gray)) < 127.0:
            gray = cv2.bitwise_not(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ #
    # Zones manuscrites : pipeline adaptatif (CLAHE + ombres + adaptative)
    # ------------------------------------------------------------------ #
    @_as_preprocessor_error
    def preprocess_handwritten(
        self,
        image: np.ndarray,
        *,
        shadow_suppress: bool = True,
        contrast: bool = True,
        binarize: bool = True,
        structural: bool = False,
    ) -> np.ndarray:
        """Prétraite une **zone manuscrite** — écriture pauvre en contraste,
        fonds tachés/ombragés des copies scannées.

        Le pipeline, propre aux zones à la main, combine :

        1. ``shadow_suppress`` — estimation du fond par un flou très collant
           puis soustraction : la teinte continue du papier (ombre/taches)
           disparaît, l'encre sombre est préservée ;
        2. ``contrast`` — CLAHE renforcé (clipLimit 3.0) sur le canal L de
           l'espace LAB ;
        3. ``binarize`` — seuillage **adaptatif gaussien** par blocs
           (19×19, C=8) résistant aux pâleurs de la plume, polarité
           normalisée (texte sombre sur fond clair) ;
        4. ``structural`` — renforcement des barres des lettres (ouverture/
           fermeture verticale légère) pour aider la segmentation de
           l'écriture cursive avant OCR.

        Entrées : BGR ou gris (automatiquement normalisé). Sortie : BGR.
        """
        image = self._as_bgr8(image)
        if shadow_suppress:
            image = self._suppress_shadow(image)
        if contrast:
            image = self._apply_clahe_strength(image, clip=3.0)
        if binarize:
            image = self._binarize_adaptive(image, block=19, c=8.0)
        if structural:
            image = self._emphasize_strokes(image)
        return self._final_contour(image)

    @staticmethod
    def _suppress_shadow(image: np.ndarray) -> np.ndarray:
        """Retire le fond papier (ombres, taches, relief du scan).

        Soustrait un flou très fort (sigma 25) : seule la variation locale
        rapide (l'encre) ressort, la teinte continue du fond disparaît.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=25.0)
        difference = cv2.subtract(blurred, image)
        return cv2.add(difference, 128)

    @staticmethod
    def _emphasize_strokes(image: np.ndarray) -> np.ndarray:
        """Renforce les barres de l'écriture (morphologie verticale) avant OCR.

        Fermeture verticale (1×3) reliant les points de l'encre dans le sens
        de l'écriture ; l'excès est borné pour ne pas coller les lettres.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        merged = cv2.addWeighted(gray, 0.85, closed, 0.15, 0.0)
        return cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _final_contour(image: np.ndarray) -> np.ndarray:
        """Normalisation finale : BGR 8 bits, polarité texte-sombre."""
        image = np.clip(image, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# Zones d'intérêt (ROI) par défaut — formulaires type feuille d'examen
# --------------------------------------------------------------------------- #
# Coordonnées en fractions de page (x0, y0, x1, y1) — à adapter au gabarit réel
# du formulaire (mode CLI ``--roi``).
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
            self.logger.debug("cv2.barcode indisponible (%s); repli QR.", exc)
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
# Moteur OCR local (HTR TrOCR — ONNX, quantifié int8)
# --------------------------------------------------------------------------- #
def _dark_ratio(image: np.ndarray) -> float:
    """Proportion de pixels sombres (encre) : ``gray < 128`` dans [0, 1]."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray < 128).mean())


class LocalOCREngine:
    """Moteur HTR local : TrOCR-small-handwritten en ONNX Runtime (CPU).

    Caractéristiques :

    * Sessions ONNX **quantifiées int8** (encodeur + décodeur fusionnés),
      chargées une seule fois au démarrage (``warm_up``) — thread-safe.
    * Détection de lignes manuscrites par projection OpenCV (sans réseau).
    * Zones d'intérêt (ROI) : transcription d'une passe par zone ; les zones
      vides (aucune encre) sont ignorées — aucun coût de calcul inutile.
    * Sortie normalisée ``[{"text", "confidence", "box"}]``.
    """

    def __init__(
        self,
        lang: str = "en",
        model_dir: Optional[PathLike] = None,
        cpu_threads: int = 0,
        preprocess_kwargs: Optional[dict[str, Any]] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        logger: Optional[logging.Logger] = None,
        *,
        max_side_len: Optional[int] = None,
        barcode: bool = True,
        barcode_budget_ms: float = _DEFAULT_BARCODE_BUDGET_MS,
        barcode_max_preview_side: int = _BARCODE_PREVIEW_SIDE,
        min_ink_ratio: float = 0.0015,
    ) -> None:
        """Initialise le moteur HTR.

        Args:
            lang: Langue (informatif — TrOCR est bilingue/zéro-shot).
            model_dir: Dossier des poids TrOCR (``encoder_model_quantized.onnx``
                + ``decoder_model_merged_quantized.onnx`` + ``tokenizer.json``).
                Si omis : ``<racine du projet>/models/trocr``.
            cpu_threads: Threads CPU ONNX. ``0`` = moins(8, cœurs).
            preprocess_kwargs: Options de :meth:`ImagePreprocessor.preprocess`.
            preprocessor: Instance à réutiliser (injection tests).
            logger: Logger.
            max_side_len: Longueur max côté après prétraitement.
            barcode: Active le scanner local de codes-barres.
            barcode_budget_ms: Budget du scan code-barres par page (ms).
            barcode_max_preview_side: Taille de la vue de détection.
            min_ink_ratio: Ratio minimal d'encre (dans [0,1]) sous lequel une
                zone est considérée vide — évite les inférences inutiles et le
                bruit (règles imprimées, ombres).

        Raises:
            OCRInitError: modèles TrOCR ONNX absents (voir
                ``scriptvault.download_trocr_models``).
        """
        self.logger = logger or logging.getLogger("scriptvault.core_ocr.engine")
        self.lang = lang
        self._preprocessor = preprocessor or ImagePreprocessor(
            logger=self.logger, max_side_len=max_side_len
        )
        self._preprocess_kwargs = dict(preprocess_kwargs or {})
        self._max_side_len = (
            max_side_len if max_side_len is not None and max_side_len > 0 else None
        )
        self._min_ink_ratio = max(0.0, float(min_ink_ratio))
        self._ready = False

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

        cpu_count = os.cpu_count() or 4
        self.cpu_threads = cpu_threads if cpu_threads > 0 else min(8, cpu_count)

        from .htr_engine import HandwrittenLineDetector, TrOcrEngine

        htr_dir = self._resolve_model_dir(model_dir)
        self._htr: TrOcrEngine | None = TrOcrEngine(htr_dir, threads=self.cpu_threads)
        self._detector = HandwrittenLineDetector(max_lines=32)
        self._backend_name = "htr"
        self.logger.info("Backend HTR (TrOCR ONNX) initialisé : %s.", htr_dir)
        self.warm_up()

    @staticmethod
    def _resolve_model_dir(model_dir: Optional[PathLike]) -> Path:
        """Localise le dossier des modèles TrOCR.

        Cherche successivement ``model_dir`` lui-même, ``model_dir/trocr``,
        puis la racine du projet ``models/trocr`` (sortie du script de
        téléchargement).
        """
        candidates: list[Path] = []
        if model_dir is not None:
            root = Path(model_dir)
            candidates = [root / "trocr", root]
        candidates.append(Path(__file__).resolve().parents[2] / "models" / "trocr")
        for candidate in candidates:
            if (
                candidate / "encoder_model_quantized.onnx"
            ).exists():
                return candidate
        raise OCRInitError(
            "Moteur HTR : modèles TrOCR ONNX absents. "
            "Lancez : python -m scriptvault.download_trocr_models"
        )

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        """Indique si le moteur est initialisé (modèles chargés)."""
        return self._ready

    @property
    def preprocessor(self) -> ImagePreprocessor:
        """Le préprocesseur d'images associé au moteur."""
        return self._preprocessor

    def warm_up(self) -> None:
        """Préchauffe les sessions ONNX (poids en RAM, graph JIT compilé).

        Exécute une transcription sur une image de test synthétique : la
        première requête utilisateur n'encaisse ni chargement de fichiers ni
        compilation du graphe.
        """
        blank: np.ndarray = np.zeros((96, 96, 3), dtype=np.uint8)
        blank[:] = 255
        cv2.rectangle(blank, (10, 40), (86, 60), (10, 10, 10), -1)
        htr = self._htr
        try:
            assert htr is not None
            htr.recognize(blank)
            self._ready = True
            self.logger.info(
                "Sessions TrOCR pré-chargées (threads=%d).",
                self.cpu_threads,
            )
        except Exception as exc:
            self._ready = False
            self.logger.warning("Warm-up HTR échoué (%s).", exc)

    # --- OCR globale (page unique) ------------------------------------ #
    def predict_array(
        self,
        image: np.ndarray,
        *,
        preprocess: bool = True,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
        zones: Optional[bool] = None,
    ) -> list[OCRResultItem]:
        """Exécute l'HTR sur un tableau numpy (une seule page).

        Args:
            image: Image BGR/GRAY 8 bits.
            preprocess: Applique le pipeline OpenCV (CLAHE/Otsu/deskew).
            rois: Profil de zones d'intérêt. Chaque zone nommée est transcrite
                individuellement (une inférence par zone, les zones vides sont
                ignorées) et les items résultants portent ``label`` +
                ``box`` (repère page).
            scan_barcode: Détection de codes-barres/QR avant l'OCR (défaut:
                réglage du moteur).
            zones: Lecture du formulaire **zone par zone** (OpenCV + chiffres
                MNIST via :mod:`scriptvault.image_processing`) au lieu de la
                passe ``_run_ocr`` pleine page. ``None`` = auto, ``False`` =
                forcé pleine page.
        """
        if not self._ready:
            raise OCRInitError("Le moteur OCR n'est pas initialisé.")
        try:
            if rois:
                result = self._predict_page_array(
                    image, 0, preprocess, rois, scan_barcode, zones
                )
                return result["items"]
            if zones is not False:
                from .image_processing import has_form_structure, read_exam_form_zones

                if has_form_structure(image):
                    return read_exam_form_zones(image, self._recognize_crop)
            if preprocess:
                image = self._preprocessor.preprocess(image, **self._preprocess_kwargs)
            return self._run_ocr(image)
        except (OCRImageError, OCRBaseError):
            raise
        except cv2.error as exc:
            raise OCRImageError(f"Erreur OpenCV pendant l'OCR: {exc}") from exc
        except OCRBaseError as exc:
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
        zones: Optional[bool] = None,
    ) -> list[PageResult]:
        """Analyse **toutes** les pages d'un fichier (TIF multi-pages).

        Args:
            image_path: Chemin du fichier image.
            preprocess: Applique le pipeline OpenCV (CLAHE/Otsu/deskew/resize).
            rois: Profil de zones d'intérêt. Si fourni, chaque zone nommée est
                transcrite individuellement (une inférence par zone) — les
                zones sans encre sont ignorées.
            scan_barcode: Active la détection locale de codes-barres/QR.
            zones: Lecture zonal du formulaire (voir :meth:`predict_array`).

        Returns:
            Un :class:`PageResult` par page. ``image`` = image exactement
            analysée (prétraitée, zones code-barres masquées).
        """
        pages = self._preprocessor.read_pages(image_path)
        return [
            self._predict_page_array(page, index, preprocess, rois, scan_barcode, zones)
            for index, page in enumerate(pages, start=1)
        ]

    def predict_pages_bytes(
        self,
        data: bytes | np.ndarray,
        *,
        preprocess: bool = True,
        rois: Optional[ROIProfile] = None,
        scan_barcode: Optional[bool] = None,
        zones: Optional[bool] = None,
    ) -> list[PageResult]:
        """Identique à :meth:`predict_pages` pour des octets bruts."""
        pages = self._preprocessor.read_pages_bytes(data)
        return [
            self._predict_page_array(page, index, preprocess, rois, scan_barcode, zones)
            for index, page in enumerate(pages, start=1)
        ]

    def _predict_page_array(
        self,
        page: np.ndarray,
        page_number: int,
        preprocess: bool,
        rois: Optional[ROIProfile],
        scan_barcode: Optional[bool],
        zones: Optional[bool] = None,
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
            if scanner is not None:
                try:
                    barcodes = scanner.scan(page)
                except Exception as exc:
                    self.logger.warning("Scan code-barres ignoré: %s", exc)
            # Masquage des zones concernées (évite les faux OCR sur barres)
            if barcodes:
                page = mask_barcodes(page, barcodes)

        # --- 2) Prétraitement --------------------------------------------- #
        if preprocess:
            if rois:
                # Robot résistent au deskew : les coordonnées ROI restent alignées.
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

        # --- 3) Inférence (OCR par ROI ou par lignes) --------------------- #
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
                        [crop.shape[1], 0],
                        [crop.shape[1], crop.shape[0]],
                        [0, crop.shape[0]],
                    ]
                    entry["box"] = [[x0 + p[0], y0 + p[1]] for p in box]
                    entry["label"] = label
                    items.append(entry)
        else:
            if zones is not False:
                from .image_processing import has_form_structure, read_exam_form_zones

                if has_form_structure(page):
                    items = read_exam_form_zones(page, self._recognize_crop)
                    processed = page
                    height, width = page.shape[:2]
                else:
                    items = self._run_ocr(processed)
            else:
                items = self._run_ocr(processed)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return make_page_result(
            page_number, width, height, items, elapsed_ms, barcodes, processed
        )

    # ------------------------------------------------------------------ #
    # Reconnaissance
    # ------------------------------------------------------------------ #
    def _run_ocr(self, image: np.ndarray) -> list[OCRResultItem]:
        """Transcrit la page : lignes détectées, triées en lecture naturelle."""
        # Prune : une page entièrement blanche (ou sans encre) ne coûte rien.
        if _dark_ratio(image) < self._min_ink_ratio:
            return []
        lines = self._detector.detect_lines(image)
        entries: list[OCRResultItem] = []
        for line in sorted(lines, key=_line_sort_key):
            crop = image[line.y0 : line.y1, line.x0 : line.x1]
            text, confidence = self._transcribe(crop)
            if not text:
                continue
            entries.append({"text": text, "confidence": confidence, "box": line.box})
        return entries

    def _recognize_crop(self, crop: np.ndarray) -> list[OCRResultItem]:
        """Transcrit une zone (ROI) : une transcription directe par champ."""
        text, confidence = self._transcribe(crop)
        if not text:
            return []
        return [
            {
                "text": text,
                "confidence": confidence,
                "box": [
                    [0, 0],
                    [crop.shape[1], 0],
                    [crop.shape[1], crop.shape[0]],
                    [0, crop.shape[0]],
                ],
            }
        ]

    def _transcribe(self, crop: np.ndarray) -> tuple[str, float]:
        """Transcrit une crop si elle contient de l'encre (sinon vide)."""
        if crop.size == 0 or _dark_ratio(crop) < self._min_ink_ratio:
            return "", 0.0
        htr = self._htr
        assert htr is not None
        text, confidence = htr.recognize(crop)
        return text.strip(), confidence

    # ------------------------------------------------------------------ #
    # Cycle de vie
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Libère les sessions ONNX et les ressources."""
        htr = self._htr
        if htr is not None:
            try:
                htr.close()
            except Exception as exc:  # pragma: no cover - défensif
                self.logger.warning("Fermeture du backend HTR échouée: %s", exc)
            self._htr = None
        self._ready = False

    def __enter__(self) -> "LocalOCREngine":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _line_sort_key(line: Any) -> tuple[int, int]:
    """Ordre de lecture naturel : haut→bas puis gauche→droite."""
    return (line.y0, line.x0)


# --------------------------------------------------------------------------- #
# Point d'entrée CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="HTR local hors-ligne (TrOCR ONNX CPU + OpenCV)."
    )
    parser.add_argument("image", help="Chemin de l'image à analyser (PNG, JPG, TIF…).")
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
        "--roi-json",
        default=None,
        help=(
            "Profil des zones d'intérêt en JSON "
            '(ex. \'{"nom": [0.02, 0.09, 0.55, 0.14], "cin": [0.02, 0.23, 0.40, 0.28]}\'). '
            "Les clés correspondent aux champs du gabarit (form_analyzer.FORM_FIELDS)."
        ),
    )
    parser.add_argument(
        "--no-barcode",
        action="store_true",
        help="Désactive la détection de codes-barres/QR.",
    )
    parser.add_argument(
        "--zones",
        action="store_true",
        help="Lecture par zones du formulaire (grilles + pointillés, jamais page entière).",
    )
    parser.add_argument(
        "--no-zones",
        action="store_true",
        help="Force la passe pleine page (désactive le mode zones auto).",
    )
    parser.add_argument(
        "--max-side", type=int, default=0, help="Longueur max côté OCR (0 = auto)."
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="Threads CPU (0 = auto)."
    )
    parser.add_argument("--model-dir", default=None, help="Dossier local des modèles.")
    args = parser.parse_args()

    rois: dict[str, tuple[float, float, float, float]] | None
    if args.roi_json is not None:
        try:
            raw = json.loads(args.roi_json)
        except json.JSONDecodeError as exc:
            print(f"--roi-json invalide: {exc}", file=sys.stderr)
            sys.exit(2)
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
    else:
        rois = DEFAULT_EXAM_ROIS if args.roi else None

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    with LocalOCREngine(
        model_dir=args.model_dir,
        cpu_threads=args.threads,
        max_side_len=args.max_side or None,
    ) as engine:
        pages = engine.predict_pages(
            args.image,
            preprocess=not args.no_preprocess,
            rois=(rois if rois else None),
            scan_barcode=not args.no_barcode,
            zones=(True if args.zones else (False if args.no_zones else None)),
        )
        payload = [
            {key: value for key, value in page.items() if key != "image"}
            for page in pages
        ]
        print(json.dumps({"backend": engine._backend_name, "pages": payload}))
        sys.exit(0)
