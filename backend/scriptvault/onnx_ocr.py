"""PP-OCRv5 mobile en pur ONNX Runtime (CPU) — remplace paddlex sans oneDNN.

PaddlePaddle 3.x (PIR, sans oneDNN — le paramètre fait crash avec
``ConvertPirAttribute2RuntimeAttribute``) coûte ~0,5 s **par boîte de texte**
sur CPU. Les exports ONNX officiels des modèles PP-OCRv5 (générés avec
paddle2onnx, publiés sur Hugging Face) s'exécutent sous onnxruntime :
noyaux optimisés multi-threads ~30-60 ms/boîte — un facteur ~10×.

Les pré/post-traitements sont réutilisés depuis ``paddlex`` (pur numpy/cv2,
aucune dépendance paddle) : ``DetResizeForTest``, ``NormalizeImage``,
``DBPostProcess``, ``OCRReisizeNormImg``, ``CTCLabelDecode``.

La classe :class:`OnnxPaddleOCR` expose une interface compatible avec
l'objet ``PaddleOCR`` de paddlex : ``predict(image | [images])`` retourne
une liste de dictionnaires ``{"rec_texts", "rec_scores", "dt_polys"}`` —
consommable telle quelle par :mod:`scriptvault.paddle_engine`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np

from .core_ocr import OCRInitError

logger = logging.getLogger("scriptvault.onnx_ocr")

#: Nom des modèles ONNX attendus dans ``models/paddle_onnx/``.
_DET_ONNX = "PP-OCRv5_mobile_det.onnx"
_REC_ONNX = "PP-OCRv5_mobile_rec.onnx"

#: Limite max du côté long pour la détection (config officielle PP-OCRv5).
_DET_LIMIT_SIDE = 960

#: Hauteur de ligne de reconnaissance (config officielle).
_REC_H = 48
_REC_MAX_W = 3200

#: Taille max d'un lot de lignes (mémoire borne).
_REC_BATCH = 32


def _default_model_dir() -> str:
    """``models/paddle_onnx`` sous la racine du projet."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "models", "paddle_onnx"))


def _load_character_dict() -> Optional[list[str]]:
    """Dictionnaire de caractères de PP-OCRv5 mobile rec.

    Source : ``config.json`` du modèle paddlex local (champ
    ``PostProcess.character_dict``).
    """
    candidates = [
        os.path.expanduser(
            r"~\.paddlex\official_models\PP-OCRv5_mobile_rec\config.json"
        ),
        os.path.join(_default_model_dir(), "ppocrv5_dict.txt"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith(".json"):
                import json

                data = json.load(open(path, encoding="utf-8"))
                chars = data.get("PostProcess", {}).get("character_dict")
                if isinstance(chars, list) and len(chars) > 100:
                    return list(chars)
            else:
                text = open(path, encoding="utf-8").read()
                chars = [chr(int(c)) for c in text.split() if c.strip()]
                if len(chars) > 100:
                    return chars
        except Exception as exc:  # pragma: no cover - défensif
            logger.warning("Dictionnaire %r illisible: %s", path, exc)
    return None


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Ordonne les 4 sommets d'un polygone dans le sens horaire (TL, TR, BR, BL)."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _rotate_crop_image(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Coupe la boîte détectée et la redresse à l'horizontale (PaddleOCR)."""
    points = _order_points_clockwise(np.asarray(box, dtype=np.float32).reshape(-1, 2))
    src = np.array(
        [
            [points[0][0], points[0][1]],
            [points[1][0], points[1][1]],
            [points[2][0], points[2][1]],
            [points[3][0], points[3][1]],
        ],
        dtype=np.float32,
    )
    width = int(max(
        np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[2] - src[3])
    ))
    height = int(max(
        np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2])
    ))
    if width < 2 or height < 2:
        return np.zeros((_REC_H, _REC_H, 3), dtype=np.uint8)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img, matrix, (width, height), borderValue=(0, 0, 0)
    )


class OnnxPaddleOCR:
    """Pipette PP-OCRv5 (det + rec) sous onnxruntime — interface paddlex.

    Args:
        model_dir: Dossier contenant ``PP-OCRv5_mobile_det.onnx`` et
            ``PP-OCRv5_mobile_rec.onnx`` (défaut ``models/paddle_onnx/``).
        cpu_threads: Threads d'inférence onnxruntime (``0`` = auto).
        logger: Logger optionnel.

    Raises:
        OCRInitError: Modèles ONNX ou dictionnaire introuvables.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        cpu_threads: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("scriptvault.onnx_ocr")
        model_dir = model_dir or _default_model_dir()
        det_path = os.path.join(model_dir, _DET_ONNX)
        rec_path = os.path.join(model_dir, _REC_ONNX)
        missing = [p for p in (det_path, rec_path) if not os.path.exists(p)]
        if missing:
            raise OCRInitError(
                "Modèles ONNX absents: "
                + ", ".join(os.path.basename(p) for p in missing)
                + f" (dossier {model_dir}). "
                "Téléchargez les exports officiels PP-OCRv5 dans models/paddle_onnx/."
            )
        chars = _load_character_dict()
        if chars is None:
            raise OCRInitError(
                "Dictionnaire de caractères PP-OCRv5 introuvable "
                "(config.json paddlex requis)."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OCRInitError(
                "onnxruntime absent: pip install onnxruntime"
            ) from exc

        threads = cpu_threads if cpu_threads and cpu_threads > 0 else (
            os.cpu_count() or 4
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._det = ort.InferenceSession(
            det_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._rec = ort.InferenceSession(
            rec_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._det_name = self._det.get_inputs()[0].name
        self._rec_name = self._rec.get_inputs()[0].name

        from paddlex.inference.models.text_detection.processors import (
            DBPostProcess,
            DetResizeForTest,
            NormalizeImage,
        )

        self._resize = DetResizeForTest(
            limit_side_len=_DET_LIMIT_SIDE, limit_type="resize_long"
        )
        self._normalize = NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order="hwc",
            scale="1./255.",
        )
        self._db = DBPostProcess(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1000,
            unclip_ratio=1.5,
        )

        from paddlex.inference.models.text_recognition.processors import (
            CTCLabelDecode,
            OCRReisizeNormImg,
        )

        self._rec_prep = OCRReisizeNormImg(
            rec_image_shape=[3, _REC_H, _REC_MAX_W]
        )
        self._decode = CTCLabelDecode(character_list=chars, use_space_char=True)
        self._backend_name = "ppocrv5-onnx"

        self._warm_up()
        self.logger.info(
            "OCR ONNX PP-OCRv5 prêt (%d threads, modèles %s).",
            threads,
            model_dir,
        )

    # ------------------------------------------------------------------ #
    def _warm_up(self) -> None:
        """Première exécution : matérialise les graphes et élimine le JIT."""
        dummy = np.full((100, 200, 3), 255, dtype=np.uint8)
        try:
            self.predict(dummy)
        except Exception as exc:  # pragma: no cover - défensif
            self.logger.warning("Warm-up ONNX ignoré: %s", exc)

    def close(self) -> None:
        """Libère les sessions (aucun état persistant)."""
        self._det = None  # type: ignore[assignment]
        self._rec = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Détection
    # ------------------------------------------------------------------ #
    def _detect(self, image: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """Boîtes de texte (quadrilatères, repère image) + scores DB."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized, shapes = self._resize([image])
        rimg, shape = resized[0], shapes[0]
        norm = self._normalize([rimg])[0]
        tensor = np.transpose(norm, (2, 0, 1))[None].astype(np.float32)
        prob = self._det.run(None, {self._det_name: tensor})[0]
        boxes, scores = self._db([prob], [shape])
        polys = []
        for box in boxes[0]:
            pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            polys.append(pts)
        return polys, scores[0]

    # ------------------------------------------------------------------ #
    # Reconnaissance (batch unique par image)
    # ------------------------------------------------------------------ #
    def _recognize(self, crops: list[np.ndarray]) -> tuple[list[str], list[float]]:
        if not crops:
            return [], []
        max_wh = max((crop.shape[1] / max(1, crop.shape[0]) for crop in crops))
        tensors: list[np.ndarray] = []
        for crop in crops:
            if crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h == 0 or w == 0:
                continue
            tensors.append(self._rec_prep.resize_norm_img(crop, max_wh))
        if not tensors:
            return [], []
        texts: list[str] = []
        scores: list[float] = []
        for start in range(0, len(tensors), _REC_BATCH):
            batch = tensors[start : start + _REC_BATCH]
            out = self._rec.run(None, {self._rec_name: np.stack(batch)})[0]
            t, s = self._decode([out])
            texts.extend(t)
            scores.extend(s)
        return texts, scores

    # ------------------------------------------------------------------ #
    # API compatible paddlex
    # ------------------------------------------------------------------ #
    def predict(self, input: Any) -> list[dict[str, Any]]:
        """Analyse une image ou une liste d'images.

        Retourne une liste de dictionnaires au format paddlex
        (``rec_texts`` / ``rec_scores`` / ``dt_polys``).
        """
        images = input if isinstance(input, (list, tuple)) else [input]
        pages: list[dict[str, Any]] = []
        for image in images:
            if image is None or image.size == 0:
                pages.append({"rec_texts": [], "rec_scores": [], "dt_polys": []})
                continue
            polys, det_scores = self._detect(image)
            crops = [_rotate_crop_image(image, poly) for poly in polys]
            texts, rec_scores = self._recognize(crops)
            pages.append(
                {
                    "rec_texts": texts,
                    "rec_scores": rec_scores,
                    "dt_polys": [poly.tolist() for poly in polys],
                    "det_scores": det_scores,
                }
            )
        return pages
