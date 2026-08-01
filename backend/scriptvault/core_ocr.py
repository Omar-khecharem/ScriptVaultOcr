"""Moteur OCR local haute performance (PaddleOCR CPU + OpenCV).

Module autonome, sans aucune dépendance cloud ou API externe. Il expose deux
classes principales:

* :class:`ImagePreprocessor` : lecture robuste multi-formats (PNG, JPG, TIFF,
  WebP) et pipeline OpenCV avancé (débruitage gaussien/médian, CLAHE,
  redressement automatique de l'inclinaison par Hough / moments de Hu,
  binarisation adaptative).
* :class:`LocalOCREngine` : wrapper PaddleOCR optimisé CPU (MKL-DNN / threads
  OpenMP), poids pré-chargés en mémoire locale, sortie structurée
  ``[{"text", "confidence", "box"}]``.

Exemple d'utilisation::

    from core_ocr import LocalOCREngine

    engine = LocalOCREngine(lang="en")
    results = engine.predict("scan_001.png")
    for item in results:
        print(item["text"], item["confidence"])
    engine.close()

Dépendances: ``paddlepaddle``, ``paddleocr>=2.7``, ``opencv-python``, ``numpy``.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import cv2
import numpy as np

__version__ = "1.0.0"
__all__ = [
    "ImagePreprocessor",
    "LocalOCREngine",
    "OCRBaseError",
    "OCRInitError",
    "OCRImageError",
    "OCRInferenceError",
]

PathLike = Union[str, os.PathLike[str]]
Box = list[list[int]]
OCRResultItem = dict[str, Any]

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger("core_ocr")
logger.addHandler(logging.NullHandler())

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
)


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
# Prétraitement d'images
# --------------------------------------------------------------------------- #
class ImagePreprocessor:
    """Pipeline OpenCV de prétraitement pour documents scannés.

    Chaîne de traitement (chaque étape est activable indépendamment) :

    1. ``denoise`` : débruitage gaussien (kernel 3x3) puis médian (kernel 3x3).
       Désactivé par défaut: dégrade les petits textes et casse la
       binarisation adaptative (texte vide en sortie OCR).
    2. ``clahe`` : égalisation d'histogramme adaptative (CLAHE) sur le canal
       L de l'espace LAB.
    3. ``deskew`` : détection de l'inclinaison du texte par Transformée de
       Hough probabiliste (méthode principale) avec repli sur les moments de
       Hu (boîte englobante minimale via ``cv2.minAreaRect``), puis rotation
       avec agrandissement du canevas.
    4. ``binarize`` : binarisation adaptative gaussienne avec normalisation
       de polarité (texte sombre sur fond clair, format le plus compatible
       avec les modèles de reconnaissance).
    """

    def __init__(
        self,
        *,
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
            clahe_clip: Limite de contraste (clipLimit) de CLAHE.
            clahe_grid: Taille des tuiles ``(lignes, colonnes)`` de CLAHE.
            adaptive_block: Taille du voisinage (impair, >= 3) pour
                ``cv2.adaptiveThreshold``.
            adaptive_c: Constante soustraite à la moyenne locale lors de la
                binarisation adaptative.
            deskew_min_angle: Angle minimal (degrés) déclenchant une rotation.
            deskew_max_angle: Angle maximal (degrés) appliqué en rotation.
            logger: Logger optionnel pour les messages de diagnostic.
        """
        self._clahe_clip = max(1.0, float(clahe_clip))
        self._clahe_grid = clahe_grid
        self._adaptive_block = max(3, int(adaptive_block) | 1)
        self._adaptive_c = float(adaptive_c)
        self._deskew_min_angle = float(deskew_min_angle)
        self._deskew_max_angle = float(deskew_max_angle)
        self.logger = logger or logging.getLogger("core_ocr.preprocessor")

    # ------------------------------------------------------------------ #
    # Lecture
    # ------------------------------------------------------------------ #
    @_as_preprocessor_error
    def read_image(self, image_path: PathLike) -> np.ndarray:
        """Lit une image depuis le disque, tous formats confondus.

        La lecture passe par ``np.fromfile`` + ``cv2.imdecode`` afin de
        supporter les chemins Unicode (Windows) et les formats PNG, JPG,
        JPEG, TIFF, WebP, BMP.

        Args:
            image_path: Chemin du fichier image.

        Returns:
            Image BGR 8 bits, forme ``(H, W, 3)``.

        Raises:
            OCRImageError: Fichier introuvable, illisible ou invalide.
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
        return self.read_image_bytes(data)

    @_as_preprocessor_error
    def read_image_bytes(self, data: bytes | np.ndarray) -> np.ndarray:
        """Décode une image depuis ses octets bruts.

        Args:
            data: Octets bruts du fichier image (ou tableau uint8 brut).

        Returns:
            Image BGR 8 bits, forme ``(H, W, 3)``.

        Raises:
            OCRImageError: Les données ne correspondent à aucune image
                décodable (format non supporté ou fichier corrompu).
        """
        raw = (
            np.frombuffer(data, dtype=np.uint8)
            if isinstance(data, (bytes, bytearray, memoryview))
            else np.asarray(data, dtype=np.uint8)
        )
        if raw.size == 0:
            raise OCRImageError("Données image vides.")
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise OCRImageError(
                "Décodage impossible: format non supporté ou données corrompues."
            )
        return self._as_bgr8(image)

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
        """Applique le pipeline complet de prétraitement.

        Args:
            image: Image d'entrée (BGR, éventuellement 8 bits).
            denoise: Active le débruitage gaussien + médian. Désactivé par
                défaut (dégrade la binarisation sur les petits textes).
            clahe: Active l'égalisation d'histogramme adaptative (CLAHE).
            deskew: Active la correction automatique d'inclinaison.
            binarize: Active la binarisation adaptative (texte sombre sur
                fond clair). À désactiver pour les photos où le modèle de
                reconnaissance donne de meilleurs résultats sur l'image
                brute.

        Returns:
            Image BGR 8 bits prétraitée, forme ``(H, W, 3)``.

        Raises:
            OCRImageError: Entrée invalide ou erreur OpenCV.
        """
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
        """Lit puis prétraite une image, avec métadonnées de diagnostic.

        Args:
            image_path: Chemin du fichier image.
            denoise: Active le débruitage. Désactivé par défaut (voir
                :meth:`preprocess`).
            clahe: Active le CLAHE.
            deskew: Active la correction d'inclinaison.
            binarize: Active la binarisation adaptative.

        Returns:
            Tuple ``(image_prétraitée, métadonnées)``. Les métadonnées
            contiennent les clés ``path``, ``size``, ``deskew_angle``,
            ``denoised``, ``clahe``, ``binarized``.
        """
        image = self.read_image(image_path)
        processed, meta = self._preprocess_impl(image, denoise, clahe, deskew, binarize)
        meta["path"] = os.fspath(image_path)
        meta["size"] = (int(image.shape[1]), int(image.shape[0]))
        return processed, meta

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
        meta: dict[str, Any] = {
            "denoised": denoise,
            "clahe": clahe,
            "binarized": binarize,
            "deskew_angle": 0.0,
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
            image = (image // 256).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """Débruitage gaussien puis médian (bruit gaussien + poivre et sel)."""
        smoothed = cv2.GaussianBlur(image, (3, 3), 0)
        return cv2.medianBlur(smoothed, 3)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Égalisation d'histogramme adaptative (CLAHE) sur le canal L."""
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
        """Corrige l'inclinaison du document.

        La méthode principale utilise la Transformée de Hough probabiliste
        (médiane des angles des lignes quasi horizontales). En cas d'échec,
        repli sur les moments de Hu via la boîte englobante minimale
        (``cv2.minAreaRect``).

        Returns:
            Tuple ``(image_redressée, angle_appliqué_en_degrés)``.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = self._deskew_hough(gray)
        if abs(angle) < self._deskew_min_angle:
            angle = self._deskew_moments(gray)
        if abs(angle) < self._deskew_min_angle:
            return image, 0.0
        clamped = max(-self._deskew_max_angle, min(self._deskew_max_angle, angle))
        if clamped != angle:
            self.logger.info(
                "Angle d'inclinaison %.2f° limité à %.2f°.",
                angle,
                clamped,
            )
        return self._rotate_bound(image, clamped), clamped

    @staticmethod
    def _deskew_hough(gray: np.ndarray) -> float:
        """Estime l'angle d'inclinaison par Transformée de Hough.

        Returns:
            Angle de correction (degrés), 0.0 si aucune inclinaison
            significative. Convention: angle positif = rotation anti-horaire.
        """
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
        """Estime l'angle par moments de Hu (boîte englobante minimale).

        Returns:
            Angle de correction (degrés), 0.0 si indétectable.
        """
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(thresh) > thresh.size * 0.5:
            thresh = cv2.bitwise_not(thresh)
        coords = cv2.findNonZero(thresh)
        if coords is None or len(coords) < 100:
            return 0.0
        rect = cv2.minAreaRect(coords)
        angle = float(rect[2])
        # La convention d'angle de minAreaRect varie selon les versions
        # d'OpenCV ([-90, 0) sur les anciennes, [0, 90] sur les récentes):
        # normalisation périodique (période 90°) dans [-45, 45].
        while angle > 45.0:
            angle -= 90.0
        while angle < -45.0:
            angle += 90.0
        return -angle

    @staticmethod
    def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotation de ``angle`` degrés (anti-horaire si positif) avec canevas
        agrandi pour ne tronquer aucun contenu."""
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
        """Binarisation adaptative gaussienne, polarité normalisée.

        Le résultat garantit un texte sombre sur fond clair, quelle que
        soit la polarité originale du document.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._adaptive_block,
            self._adaptive_c,
        )
        if float(np.mean(thresh)) < 127.0:
            thresh = cv2.bitwise_not(thresh)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# Moteur OCR local
# --------------------------------------------------------------------------- #
class LocalOCREngine:
    """Moteur PaddleOCR local, optimisé pour CPU (MKL-DNN / OpenMP / AVX2).

    Caractéristiques:

    * Import paresseux de PaddlePaddle (le module ``core_ocr`` peut être
      importé sans que Paddle soit installé).
    * Pré-chargement des poids en mémoire locale (dossier de modèles local ou
      cache ``~/.paddleocr``) et ``warm_up()`` forçant le chargement complet
      des modèles dans la RAM dès l'initialisation.
    * Multi-threading CPU: ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` et
      ``paddle.set_num_threads`` (``enable_mkldnn`` désactivé par défaut:
      bug oneDNN sur paddlepaddle 3.3.1, voir :meth:`__init__`).
    * Multi-processus: ``use_mp=True`` + ``total_process_num`` (PaddleOCR 2.x
      uniquement — l'API 3.x ne l'expose plus; activé par défaut sur
      Linux/macOS, désactivé sur Windows car le mode ``spawn`` exige que le
      script appelant soit protégé par ``if __name__ == "__main__"``).
    * Compatible PaddleOCR 2.x et 3.x (détection automatique de l'API).
    """

    def __init__(
        self,
        lang: str = "en",
        model_dir: Optional[PathLike] = None,
        cpu_threads: int = 0,
        use_mp: Optional[bool] = None,
        total_process_num: int = 0,
        use_angle_cls: bool = True,
        use_mkldnn: bool = False,
        det_model_dir: Optional[PathLike] = None,
        rec_model_dir: Optional[PathLike] = None,
        cls_model_dir: Optional[PathLike] = None,
        preprocess_kwargs: Optional[dict[str, Any]] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise le moteur OCR.

        Args:
            lang: Langue des modèles de reconnaissance (``"en"``, ``"fr"``,
                ``"ch"``, ...).
            model_dir: Dossier racine contenant les sous-dossiers ``det``,
                ``rec`` et ``cls`` pour un chargement 100 % local des poids.
                Si omis, PaddleOCR utilise son cache ``~/.paddleocr``.
            cpu_threads: Nombre de threads CPU (OpenMP/MKL). ``0`` = défaut
                (min(8, cœurs)).
            use_mp: Multi-processus PaddleOCR. ``None`` = auto (activé sauf
                sur Windows).
            total_process_num: Nombre de processus pour le mode multi-
                processus. ``0`` = défaut (min(4, cœurs)).
            use_angle_cls: Active la classification d'orientation des lignes
                de texte.
            use_mkldnn: Active l'accélération MKL-DNN (AVX2 sur x86-64).
                Désactivé par défaut: paddlepaddle 3.3.1 + oneDNN échoue sur
                les modèles PP-OCRv6 (``ConvertPirAttribute2RuntimeAttribute
                not support [pir::ArrayAttribute<pir::DoubleAttribute>]``).
            det_model_dir: Chemin du modèle de détection (surpasse
                ``model_dir``).
            rec_model_dir: Chemin du modèle de reconnaissance (surpasse
                ``model_dir``).
            cls_model_dir: Chemin du modèle de classification d'angle
                (surpasse ``model_dir``).
            preprocess_kwargs: Arguments passés à
                :meth:`ImagePreprocessor.preprocess` lors de chaque
                prédiction (ex. ``{"binarize": False}``).
            preprocessor: Instance :class:`ImagePreprocessor` à réutiliser
                (injectable pour les tests).
            logger: Logger optionnel.

        Raises:
            OCRInitError: PaddlePaddle/paddleocr non installés, ou échec de
                chargement des modèles.
        """
        self.logger = logger or logging.getLogger("core_ocr.engine")
        self.lang = lang
        self.use_angle_cls = bool(use_angle_cls)
        self._preprocessor = preprocessor or ImagePreprocessor(logger=self.logger)
        self._preprocess_kwargs = dict(preprocess_kwargs or {})
        self._ocr: Any = None
        self._api_v3: bool = False
        self._ready: bool = False

        cpu_count = os.cpu_count() or 4
        self.cpu_threads = cpu_threads if cpu_threads > 0 else min(8, cpu_count)
        self._total_process_num = (
            total_process_num if total_process_num > 0 else min(4, cpu_count)
        )
        if use_mp is None:
            use_mp = platform.system() != "Windows"
            self.logger.info(
                "Multi-processus auto: %s (désactivé sur Windows, mode spawn "
                "non protégé).",
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
        """Configure les variables d'environnement CPU avant l'import Paddle.

        Doit être appelé avant tout ``import paddle`` afin que les threads
        OpenMP/MKL soient appliqués aux blocs BLAS du framework.
        """
        if "paddle" in sys.modules:
            warnings.warn(
                "PaddlePaddle est déjà importé dans ce processus: les "
                "variables OMP/MKL_NUM_THREADS ne seront pas appliquées.",
                RuntimeWarning,
                stacklevel=2,
            )
        threads = str(min(8, os.cpu_count() or 4))
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        # FLAGS_use_mkldnn volontairement NON défini: sur paddlepaddle 3.3.1 il
        # force oneDNN sur le chemin PIR et casse l'inférence des modèles
        # PP-OCRv6 (NotImplementedError ConvertPirAttribute2RuntimeAttribute).
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
                "Dépendances OCR manquantes. Installez-les avec: "
                "pip install paddlepaddle paddleocr"
            ) from exc

        try:
            paddle.set_num_threads(self.cpu_threads)
        except Exception as exc:  # pragma: no cover - API dépendante de version
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
        else:
            kwargs.update({"use_angle_cls": use_angle_cls, "show_log": False})

        try:
            engine = PaddleOCR(**kwargs)
        except Exception as exc:
            raise OCRInitError(
                f"Échec de l'initialisation PaddleOCR (lang={lang!r}): {exc}"
            ) from exc
        return engine

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
        """Copie des options de prétraitement appliquées avant chaque
        inférence. Utile pour reproduire l'image prétraitée (alignement des
        boîtes OCR avec l'image affichée)."""
        return dict(self._preprocess_kwargs)

    def warm_up(self) -> None:
        """Force le chargement complet des poids des modèles en mémoire.

        Exécute une inférence sur une image vierge minimale (96x96) afin de
        matérialiser les modèles en RAM et d'amortir le premier appel réel.
        Les erreurs de warm-up sont journalisées mais ne bloquent pas le
        moteur.
        """
        if self._ocr is None:
            raise OCRInitError("Le moteur OCR n'est pas initialisé.")
        blank: np.ndarray = np.full((96, 96, 3), 255, dtype=np.uint8)
        try:
            self._run_ocr(blank)
            self._ready = True
            self.logger.info(
                "Modèles OCR pré-chargés en mémoire (threads=%d, "
                "processus=%d, mkldnn=%s).",
                self.cpu_threads,
                self._total_process_num,
                self.use_mp,
            )
        except Exception as exc:
            self._ready = False
            self.logger.warning(
                "Warm-up du moteur OCR échoué (%s); les modèles seront "
                "chargés au premier appel.",
                exc,
            )

    def predict_array(
        self,
        image: np.ndarray,
        *,
        preprocess: bool = True,
    ) -> list[dict[str, Any]]:
        """Exécute l'OCR sur un tableau numpy (BGR ou BGR prétraité).

        Args:
            image: Image BGR 8 bits, forme ``(H, W, 3)``.
            preprocess: Applique le pipeline de prétraitement avant OCR.

        Returns:
            Liste des lignes détectées, chaque élément au format
            ``{"text": str, "confidence": float, "box": [[x, y], [x, y],
            [x, y], [x, y]]}``.

        Raises:
            OCRImageError: Image invalide.
            OCRInferenceError: Échec de l'inférence.
        """
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
    ) -> list[dict[str, Any]]:
        """Reconnaît le texte d'une image sur le disque.

        Args:
            image_path: Chemin du fichier image (PNG, JPG, TIFF, WebP, ...).
            preprocess: Applique le pipeline de prétraitement
                (CLAHE, deskew, binarisation; le débruitage est désactivé par
                défaut) avant l'OCR.

        Returns:
            Liste structurée::

                [
                    {"text": "Bonjour", "confidence": 0.99,
                     "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]},
                    ...
                ]

        Raises:
            OCRImageError: Image illisible ou invalide.
            OCRInferenceError: Échec de l'inférence.
            OCRInitError: Moteur non initialisé.
        """
        image = self._preprocessor.read_image(image_path)
        return self.predict_array(image, preprocess=preprocess)

    def predict_bytes(
        self,
        data: bytes | np.ndarray,
        *,
        preprocess: bool = True,
    ) -> list[dict[str, Any]]:
        """Reconnaît le texte d'une image fournie en octets bruts.

        Args:
            data: Octets bruts d'un fichier image.
            preprocess: Applique le pipeline de prétraitement.

        Returns:
            Même format structuré que :meth:`predict`.

        Raises:
            OCRImageError: Données non décodables.
            OCRInferenceError: Échec de l'inférence.
        """
        image = self._preprocessor.read_image_bytes(data)
        return self.predict_array(image, preprocess=preprocess)

    def close(self) -> None:
        """Libère les ressources (processus et modèles) du moteur."""
        if self._ocr is not None:
            release = getattr(self._ocr, "release", None)
            if callable(release):
                try:
                    release()
                except Exception as exc:
                    self.logger.warning("Relâchement du moteur échoué: %s", exc)
            self._ocr = None
            self._ready = False
            self.logger.info("Moteur OCR fermé.")

    def __enter__(self) -> "LocalOCREngine":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Exécution et normalisation
    # ------------------------------------------------------------------ #
    def _run_ocr(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Délègue l'inférence à PaddleOCR selon l'API détectée."""
        if self._api_v3:
            raw = self._ocr.predict(image)
            return self._parse_v3(raw)
        raw = self._ocr.ocr(image, cls=self.use_angle_cls)
        return self._parse_v2(raw)

    def _parse_v3(self, raw: Any) -> list[dict[str, Any]]:
        """Normalise la sortie PaddleOCR 3.x (liste d'objets dict-like)."""
        entries: list[dict[str, Any]] = []
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

    def _parse_v2(self, raw: Any) -> list[dict[str, Any]]:
        """Normalise la sortie PaddleOCR 2.x (tuples imbriqués)."""
        entries: list[dict[str, Any]] = []
        self._walk_v2(raw, entries)
        return entries

    @classmethod
    def _walk_v2(cls, node: Any, out: list[dict[str, Any]]) -> None:
        """Parcourt récursivement la structure 2.x et collecte les lignes."""
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
    parser.add_argument("image", help="Chemin de l'image à analyser.")
    parser.add_argument("--lang", default="en", help="Langue (en, fr, ch, ...).")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Désactive le prétraitement OpenCV.",
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="Threads CPU (0 = auto)."
    )
    parser.add_argument("--model-dir", default=None, help="Dossier local des modèles.")
    parser.add_argument(
        "--mp",
        action="store_true",
        help="Force le mode multi-processus (Linux/macOS uniquement).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    with LocalOCREngine(
        lang=args.lang,
        model_dir=args.model_dir,
        cpu_threads=args.threads,
        use_mp=args.mp or None,
    ) as engine:
        results = engine.predict(args.image, preprocess=not args.no_preprocess)

    print(json.dumps(results, ensure_ascii=False, indent=2))
