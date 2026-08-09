"""Moteur HTR (Handwritten Text Recognition) 100 % local, optimisé CPU.

Remplace PaddleOCR par un pipeline ultra-léger dédié au manuscrit :

1. **Détection de lignes** : segmentation OpenCV (Otsu + projections +
   composantes connexes), sans réseau -- quelques ms/page.
2. **Reconnaissance** : TrOCR-small-handwritten en ONNX Runtime (encodeur +
   décodeur quantifiés int8), chargé une seule fois au démarrage, greedy
   decoding avec cache de clés/valeurs. Modèles dans ``models/trocr/`` :
   aucun appel réseau à l'exécution.

Sorties conformes au contrat OCR du projet
(``[{"text", "confidence", "box": [[x,y]x4]}]``).

Le tokenizer Unigram (SentencePiece) est réimplémenté en Python pur :
``transformers``/``torch`` ne sont jamais nécessaires.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from .core_ocr import OCRInferenceError, OCRInitError

logger = logging.getLogger("scriptvault.htr")

BOS_ID = 0
EOS_ID = 2
SPACE_MARK = "\u2581"  # métaphore SentencePiece de l'espace
IMAGE_SIZE = 384
MODEL_LAYERS = 6
KV_SCOPES = (
    "decoder.key",
    "decoder.value",
    "encoder.key",
    "encoder.value",
)


# --------------------------------------------------------------------------- #
# Tokenizer Unigram (SentencePiece) -- implementation locale sans dependance
# --------------------------------------------------------------------------- #
class TrOcrTokenizer:
    """Tokenizer Unigram : vocabulaire + scores, Viterbi, prefixe ▁ (Metaspace).

    Tokens speciaux : ``<s>=0``, ``<pad>=1``, ``</s>=2``, ``<unk>=3``.
    """

    def __init__(self, tokenizer_path: Path) -> None:
        try:
            raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise OCRInitError(f"fichier tokenizer JSON invalide: {tokenizer_path}") from exc
        model = raw.get("model", {})
        vocab = model.get("vocab")
        if not isinstance(vocab, (dict, list)) or not vocab:
            raise OCRInitError(f"tokenizer sans vocabulaire: {tokenizer_path}")
        self.tokens: dict[str, int] = {}
        self.scores: dict[str, float] = {}
        if isinstance(vocab, dict):
            for tok, score in vocab.items():
                self.tokens[str(tok)] = len(self.tokens)
                self.scores[str(tok)] = float(score)
        else:
            for index, entry in enumerate(vocab):
                tok, score = entry
                self.tokens[str(tok)] = index
                self.scores[str(tok)] = float(score)
        self.inverse: dict[int, str] = {v: k for k, v in self.tokens.items()}
        self.bos_id = self.tokens.get("<s>", 0)
        self.eos_id = self.tokens.get("</s>", 2)
        self.unk_id = self.tokens.get("<unk>", 3)

    def _viterbi(self, piece: str) -> list[int]:
        """Segmente une chaîne (sans espace) en tokens via Viterbi."""
        n = len(piece)
        if n == 0:
            return []
        best_score = [0.0] + [float("-inf")] * n
        best_split = [0] * (n + 1)
        for i in range(1, n + 1):
            best_j = 0
            best_s = float("-inf")
            for j in range(i - 1, -1, -1):
                score = self.scores.get(piece[j:i])
                if score is None:
                    continue
                acc = best_score[j] + score
                if acc > best_s:
                    best_s = acc
                    best_j = j
            best_score[i] = best_s
            best_split[i] = best_j if best_s != float("-inf") else i
        ids: list[int] = []
        end = n
        while end > 0:
            start = best_split[end]
            if start == end or best_score[end] == float("-inf"):
                ids.append(self.unk_id)
                end -= 1
            else:
                ids.append(self.tokens.get(piece[start:end], self.unk_id))
                end = start
        return list(reversed(ids))

    def encode(self, text: str) -> list[int]:
        """Metaspace + Viterbi (les sauts de ligne découpent les morceaux)."""
        text = text.replace(" ", SPACE_MARK)
        if text and not text.startswith(SPACE_MARK):
            text = SPACE_MARK + text
        ids: list[int] = []
        for chunk in text.split("\n"):
            ids.extend(self._viterbi(chunk))
        return ids

    def encode_with_special(self, text: str) -> list[int]:
        return [self.bos_id, *self.encode(text), self.eos_id]

    def decode(self, ids: Iterable[int]) -> str:
        chars: list[str] = []
        for i in ids:
            tok = self.inverse.get(int(i))
            if tok is None or tok in {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}:
                continue
            chars.append(tok)
        return "".join(chars).replace(SPACE_MARK, " ").strip()


# --------------------------------------------------------------------------- #
# Prenraitement d'une crop manuscrite pour TrOCR
# --------------------------------------------------------------------------- #
def preprocess_trocr(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Redimensionne (ratio conservé, padding blanc) puis normalise en (-0.5, 0.5).

    Retourne un tenseur float32 ``[1, 3, size, size]`` prêt pour l'encodeur.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if max(h, w) != size:
        scale = size / max(h, w)
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    h2, w2 = image.shape[:2]
    canvas: np.ndarray = np.full((size, size, 3), 255, dtype=np.uint8)
    y0 = (size - h2) // 2
    x0 = (size - w2) // 2
    canvas[y0 : y0 + h2, x0 : x0 + w2] = image
    tensor = canvas.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor)


# --------------------------------------------------------------------------- #
# Detection de lignes de texte manuscrit (OpenCV pur)
# --------------------------------------------------------------------------- #
class TextLine:
    """Boîte d'une ligne détectée (origine page)."""

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @property
    def box(self) -> list[list[float]]:
        return [
            [float(self.x0), float(self.y0)],
            [float(self.x1), float(self.y0)],
            [float(self.x1), float(self.y1)],
            [float(self.x0), float(self.y1)],
        ]


class HandwrittenLineDetector:
    """Segmente une page en lignes manuscrites (OpenCV, aucun réseau)."""

    def __init__(self, min_line_height: int = 8, max_lines: int = 64) -> None:
        self.min_line_height = min_line_height
        self.max_lines = max_lines

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        if gray.mean() > 160:
            gray = cv2.bitwise_not(gray)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bin_img

    def detect_lines(self, image: np.ndarray) -> list[TextLine]:
        """Détecte les bandes horizontales contenant de l'encre sombre."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        bin_img = self._binarize(gray)
        ksize = max(3, min(21, w // 60))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize * 3, ksize))
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        row_sums = closed.sum(axis=1) / 255.0
        lines: list[TextLine] = []
        in_line = False
        start = 0
        for y in range(h):
            active = row_sums[y] > w * 0.015
            if active and not in_line:
                in_line, start = True, y
            elif (not active or y == h - 1) and in_line:
                in_line = False
                end = y if not active else y + 1
                if end - start >= self.min_line_height:
                    x0, x1 = _line_margins(gray, start, end)
                    lines.append(TextLine(x0, start, x1, end))
        return lines[: self.max_lines]


def _line_margins(gray: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    """Bornes horizontales d'une ligne (incluant une marge latérale)."""
    band = gray[y0:y1, :]
    bin_band = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    col_sums = bin_band.sum(axis=0) / 255.0
    active = col_sums > band.shape[0] * 0.002
    idx = np.where(active)[0]
    if idx.size == 0:
        return 0, band.shape[1]
    pad = max(4, band.shape[1] // 200)
    return max(0, int(idx[0]) - pad), min(band.shape[1], int(idx[-1]) + pad)


# --------------------------------------------------------------------------- #
# Inference TrOCR (ONNX Runtime)
# --------------------------------------------------------------------------- #
class TrOcrEngine:
    """Reconnaissance d'une ligne manuscrite avec TrOCR-small (ONNX).

    Sessions chargées une seule fois à ``__init__`` ; greedy decoding avec
    cache de clés/valeurs. ``recognize`` est thread-safe : ONNX Runtime
    accepte les appels simultanés sur la même session.
    """

    def __init__(
        self,
        model_dir: Path,
        threads: int = 0,
        max_new_tokens: int = 64,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise OCRInitError("onnxruntime manquant: pip install onnxruntime") from exc

        encoder_path = model_dir / "encoder_model_quantized.onnx"
        decoder_path = model_dir / "decoder_model_merged_quantized.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        missing = [p for p in (encoder_path, decoder_path, tokenizer_path) if not p.exists()]
        if missing:
            names = ", ".join(str(p) for p in missing)
            raise OCRInitError(
                "Modeles ONNX absents : " + names
                + ". Lancez : python -m scriptvault.download_trocr_models"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._encoder = ort.InferenceSession(
            str(encoder_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._decoder = ort.InferenceSession(
            str(decoder_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = TrOcrTokenizer(tokenizer_path)
        self.max_new_tokens = max_new_tokens
        self._eos = self._tokenizer.eos_id
        self._bos = self._tokenizer.bos_id
        logger.info(
            "HTR charge : %s (%d tokens, ONNX int8).",
            model_dir,
            len(self._tokenizer.tokens),
        )

    def encode(self, image: np.ndarray) -> np.ndarray:
        tensor = preprocess_trocr(image)
        return self._encoder.run(None, {"pixel_values": tensor})[0]

    def recognize(
        self, image: np.ndarray, max_new_tokens: int | None = None
    ) -> tuple[str, float]:
        """Reconnaît une ligne : ``(texte, confiance)``.

        La confiance est la moyenne des vraisemblances des tokens générés.
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        hidden = self.encode(image)
        seq = np.array([[self._bos]], dtype=np.int64)
        past: dict[str, np.ndarray] = {}
        token_ids: list[int] = []
        log_probs: list[float] = []
        for step in range(max_tokens):
            # Le mode cache du décodeur fusionné ignore encoder_hidden_states :
            # la première passe doit donc se faire SANS cache (use_cache_branch=False),
            # les passes suivantes peuvent réutiliser past_key_values.
            use_cache = step > 0
            feeds: dict[str, Any] = {
                "input_ids": seq,
                "encoder_hidden_states": hidden,
                "use_cache_branch": np.ones((1,), dtype=np.bool_) if use_cache else np.zeros((1,), dtype=np.bool_),
            }
            for layer in range(MODEL_LAYERS):
                for scope in KV_SCOPES:
                    name = f"past_key_values.{layer}.{scope}"
                    feeds[name] = past.get(
                        name, np.zeros((1, 8, 0, 32), dtype=np.float32)
                    )
            try:
                outputs = self._decoder.run(None, feeds)
            except Exception as exc:  # noqa: BLE001
                raise OCRInferenceError(f"inférence TrOCR : {exc}") from exc
            logits = outputs[0][:, -1, :]  # [1, vocab]
            z = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(z)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_id = int(np.argmax(logits))
            token_ids.append(next_id)
            log_probs.append(float(np.log(probs[0, next_id] + 1e-9)))
            if next_id == self._eos:
                break
            out_names = [o.name for o in self._decoder.get_outputs()]
            by_name = {name_: outputs[offset] for offset, name_ in enumerate(out_names)}
            past = {}
            for layer in range(MODEL_LAYERS):
                for scope in KV_SCOPES:
                    key = f"present.{layer}.{scope}"
                    if key in by_name:
                        past[f"past_key_values.{layer}.{scope}"] = by_name[key].copy()
            seq = np.array([[next_id]], dtype=np.int64)
        text = self._tokenizer.decode(token_ids)
        score = (
            float(math.exp(sum(log_probs) / len(log_probs))) if log_probs else 0.0
        )
        return text, max(0.0, min(1.0, score))

    def close(self) -> None:  # pragma: no cover -- liberation explicite
        del self._encoder
        del self._decoder


def load_htr(model_dir: Path, threads: int = 4) -> Optional[TrOcrEngine]:
    """Instancie le moteur si les fichiers ONNX existent, sinon ``None``."""
    try:
        return TrOcrEngine(model_dir, threads=threads)
    except OCRInitError as exc:
        logger.warning("HTR desactive : %s", exc)
        return None
