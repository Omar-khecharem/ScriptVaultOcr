"""Analyse structurelle des feuilles d'examen — lecture **zone par zone**.

Ce module implémente le pipeline OCR « jamais de page entière » pour les
formulaires type concours. Au lieu de détecter puis transcrire le texte de
toute la page en une passe coûteuse (13-20 s sur CPU), il :

1. **Détecte les grilles de chiffres** (cases de CIN, Série, Identifiant)
   par analyse de contours OpenCV — sans réseau neuronal.
2. **Classe chaque chiffre** via un classifieur ONNX MNIST local
   (``models/digits/mnist-8.onnx``, 26 Ko) ; si le modèle est absent, repli
   sur la reconnaissance PaddleOCR case par case.
3. **Repère les lignes pointillées** (guides d'écriture) par composantes
   connexes — chaque ligne pointe un libellé + sa zone de valeur.
4. **Coupe le libellé** (gauche de la ligne) et le transcrit, reconnaît le
   champ par alias ; **coupe la zone de valeur** (droite de la ligne) après
   masquage des pointillés, la transcrit.
5. Assemble des items OCR étiquetés (``label`` = clé de champ) directement
   consommables par ``form_analyzer`` (passe 0 des zones d'intérêt).

Coût constaté sur la feuille de référence : détection OpenCV ~0,3-0,6 s +
   transcription d'une dizaine de petites zones (0,1-0,5 s chacune) ≈ 2-4 s,
   contre ~13-19 s en passe pleine page.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import numpy as np

__all__ = [
    "DigitCell",
    "DigitGrid",
    "FieldBand",
    "DigitClassifier",
    "extract_digit_grids",
    "detect_dotted_bands",
    "crop_field_value",
    "crop_dotted_mask",
    "match_field_label",
    "has_form_structure",
    "read_exam_form_zones",
    "tight_ink_crop",
]

logger = logging.getLogger("scriptvault.image_processing")

#: Taille de travail commune aux détecteurs OpenCV (économie CPU).
_WORK_SIDE = 1400

#: Un reconnaisseur de secours ``(crop) -> (texte, confiance)`` (TrOCR).
HandwrittenRecognizer = Callable[[np.ndarray], tuple[str, float]]

#: Un lecteur de champ manuscrit **conscient du champ**
#: ``(crop, field_type) -> (texte, confiance)`` (VLM local).
HandwrittenReader = Callable[[np.ndarray, str], tuple[str, float]]

#: Un lecteur de grille de bandes ``(grille, première, dernière) -> ...``
#: (VLM local) : lit toutes les lignes numérotées de la grille en un appel
#: et retourne ``[(index_absolu, texte, confiance), ...]`` — ``None`` si
#: l'appel a échoué (le pipeline retombe sur le chemin TrOCR).
BandGridReader = Callable[
    [np.ndarray, int, int], Optional[list[tuple[int, str, float]]]
]

#: Plage de tailles (px, repère page) des cases de chiffres du gabarit.
_CELL_MIN = 60
_CELL_MAX = 260

#: Nombre de cases caractéristique d'un champ (grille chiffrée).
_GRID_KEY_BY_COUNT: dict[int, str] = {
    3: "serie",
    6: "identifiant",
    8: "cin",
}

#: Écart vertical toléré entre une grille de chiffres et sa bande de
#: pointillés (px) : la valeur de cette bande vient des cases, pas de l'OCR.
_GRID_BAND_TOL_PX = 48

#: Nombre minimal de cases sur une même rangée pour former une grille.
_MIN_GRID_CELLS = 2

#: Marges verticales de la zone libellé/valeur autour de la ligne pointillée.
#: Le libellé et les lettres manuscrites s'étendent bien au-dessus de la
#: ligne guide : montant = ~2 % de la hauteur page, descendant = ~1,2 %.
_UP_FRAC = 0.020
_DOWN_FRAC = 0.012
_MIN_PAD_Y = 20
_VALUE_PAD_X = 10

#: Deux lignes pointillées plus proches que ce seuil (px) sont considérées
#: comme la même rangée (dédoublement de détection) — fusion pour éviter
#: de transcrire deux fois la même ligne.
_MERGE_GAP_PX = 24

#: Droite maximale de la zone de valeur (fraction de la largeur).
_VALUE_MAX_X = 0.95

#: Largeur maximale de la zone de libellé (fraction de la largeur).
_LABEL_MAX_X = 0.62


def _line_pad_y(height: int) -> tuple[int, int]:
    """Marges verticales (haut, bas) autour d'une ligne pointillée (px page)."""
    return (
        max(_MIN_PAD_Y, int(round(height * _UP_FRAC))),
        max(_MIN_PAD_Y, int(round(height * _DOWN_FRAC))),
    )


@dataclass(frozen=True)
class DigitCell:
    """Une cellule (case) d'une grille de chiffres (coordonnées pixels page)."""

    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class DigitGrid:
    """Une rangée de cases chiffrées (CIN : 8, Identifiant : 6, Série : 3)."""

    cells: tuple[DigitCell, ...]
    label: str = ""

    @property
    def y_center(self) -> int:
        if not self.cells:
            return 0
        return int(round(sum(c.y + c.h / 2 for c in self.cells) / len(self.cells)))

    @property
    def x0(self) -> int:
        return min(c.x for c in self.cells) if self.cells else 0

    @property
    def x1(self) -> int:
        return max(c.x + c.w for c in self.cells) if self.cells else 0


@dataclass(frozen=True)
class FieldBand:
    """Une rangée de formulaire : sa ligne pointillée (coordonnées page)."""

    y0: int
    y1: int
    dots_x0: int  # encre la plus à gauche de la ligne pointillée
    dots_x1: int  # encre la plus à droite de la ligne pointillée
    y_center: int = 0

    def __post_init__(self) -> None:
        if self.y_center == 0:
            object.__setattr__(self, "y_center", (self.y0 + self.y1) // 2)


# --------------------------------------------------------------------------- #
# Grilles de chiffres
# --------------------------------------------------------------------------- #
def extract_digit_grids(image: np.ndarray) -> list[DigitGrid]:
    """Détecte les grilles de cases chiffrées (contours OpenCV, sans réseau).

    Critères d'une case : contour ~carré (aspect 0,8-1,25), taille dans
    ``[CELL_MIN, CELL_MAX]`` px, remplissage > 0,55. Les cases sont groupées
    en rangées (centres verticaux à ±1/3 de la hauteur de case) ; une rangée
    doit compter au moins ``_CELL_MIN_GRID`` cases. Le champ est déduit du
    nombre de cases (8 → CIN, 6 → Identifiant, 3 → Série), sinon vide.

    Returns:
        Grilles triées de haut en bas (cellules triées de gauche à droite).
    """
    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells: list[DigitCell] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < _CELL_MIN or h < _CELL_MIN or w > _CELL_MAX or h > _CELL_MAX:
            continue
        aspect = w / h if h else 0.0
        if not 0.8 <= aspect <= 1.25:
            continue
        filled = cv2.contourArea(c)
        if w * h <= 0 or (filled / (w * h)) < 0.55:
            continue
        cells.append(DigitCell(x, y, w, h))
    if not cells:
        return []

    cells.sort(key=lambda c: c.y)
    cell_h = max((c.h for c in cells), default=1)
    rows: list[list[DigitCell]] = []
    current: list[DigitCell] = []
    last_center: Optional[int] = None
    for cell in cells:
        center = cell.y + cell.h // 2
        if last_center is not None and abs(center - last_center) > cell_h // 3:
            if len(current) >= _MIN_GRID_CELLS:
                rows.append(current)
            current = []
        current.append(cell)
        last_center = center
    if len(current) >= _MIN_GRID_CELLS:
        rows.append(current)

    grids: list[DigitGrid] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c.x)
        key = _GRID_KEY_BY_COUNT.get(len(row_sorted), "")
        grids.append(DigitGrid(cells=tuple(row_sorted), label=key))
    grids.sort(key=lambda g: g.y_center)
    return grids


# --------------------------------------------------------------------------- #
# Lignes pointillées (bandes de formulaire)
# --------------------------------------------------------------------------- #
def detect_dotted_bands(
    image: np.ndarray, *, max_side: int = _WORK_SIDE
) -> list[FieldBand]:
    """Détecte les lignes de pointillés guides d'écriture d'un formulaire.

    Une ligne pointillée forme, sur sa rangée, une large bande d'encre (chaque
    point couvre 50-70 % de la largeur, les trous éventuels sont de petits
    interstices) : on projette la densité d'encre par rangée sur une image de
    travail (plus grand côté ≤ ``max_side``) et on conserve les suites de
    rangées contiguës présentant ≥ 10 % de couverture par rangée, étendues en
    x sur ≥ 30 % de la largeur.

    Returns:
        Bandes triées du haut vers le bas (coordonnées repère **page**).
    """
    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return []
    scale = max_side / max(height, width)
    work = image
    if scale < 1.0:
        work = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
    work_h, work_w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Densité d'encre par rangée.
    ink = (binary > 0).astype(np.float32)
    density = ink.sum(axis=1) / work_w
    candidates = np.where(density >= 0.10)[0]
    if candidates.size == 0:
        return []

    # Suite de rangées contiguës (gap ≤ 3 px).
    runs: list[list[int]] = []
    current: list[int] = [int(candidates[0])]
    for y in candidates[1:]:
        y_int = int(y)
        if y_int - current[-1] > 3:
            runs.append(current)
            current = []
        current.append(y_int)
    if current:
        runs.append(current)

    bands: list[FieldBand] = []
    for run in runs:
        if len(run) < 2:
            continue
        y0, y1 = run[0], run[-1]
        slice_ink = ink[y0 : y1 + 1, :].sum(axis=0)
        cols = np.where(slice_ink > 0)[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols.min()), int(cols.max())
        if x1 - x0 < 0.30 * work_w:
            continue
        bands.append(
            FieldBand(
                y0=int(round(y0 * (height / work_h))),
                y1=int(round(y1 * (height / work_h))),
                dots_x0=int(round(x0 * (width / work_w))),
                dots_x1=int(round(x1 * (width / work_w))),
                y_center=int(round(((y0 + y1) / 2) * (height / work_h))),
            )
        )
    bands.sort(key=lambda b: b.y_center)
    return _merge_overlapping_bands(bands, height)


def _merge_overlapping_bands(
    bands: list[FieldBand], height: int
) -> list[FieldBand]:
    """Fusionne les bandes dont la détection s'est dédoublée (même rangée).

    Deux lignes pointillées distantes de moins de ``_MERGE_GAP_PX`` px
    relèvent de la même rangée de formulaire : les fusionner évite de
    transcrire deux fois la même ligne (doublons dans les items OCR).
    """
    if not bands:
        return []
    merged: list[FieldBand] = []
    for band in bands:
        if merged and band.y0 - merged[-1].y1 <= _MERGE_GAP_PX:
            prev = merged[-1]
            merged[-1] = FieldBand(
                y0=min(prev.y0, band.y0),
                y1=max(prev.y1, band.y1),
                dots_x0=min(prev.dots_x0, band.dots_x0),
                dots_x1=max(prev.dots_x1, band.dots_x1),
                y_center=(min(prev.y0, band.y0) + max(prev.y1, band.y1)) // 2,
            )
        else:
            merged.append(band)
    return merged


# --------------------------------------------------------------------------- #
# Zone de valeur
# --------------------------------------------------------------------------- #
def crop_dotted_mask(crop: np.ndarray, *, work_scale: int = 700) -> np.ndarray:
    """Masque des pointillés d'une zone (0/255, même taille que ``crop``).

    Réduit la zone à ``work_scale`` px au côté, isole les taches de
    pointillés (3-26 px de haut, ≤ 90 px de large), renvoie un masque
    pleine taille (dilatation de 2 px).
    """
    if crop is None or crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = work_scale / max(h, w)
    small = gray
    if scale < 1.0:
        small = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask_small = np.zeros_like(binary)
    for idx in range(1, n):
        x0, y0, w_, h_, _area = stats[idx]
        if h_ < 3 or h_ > 26 or w_ < 3 or w_ > 90:
            continue
        cv2.rectangle(mask_small, (x0, y0), (x0 + int(w_), y0 + int(h_)), 255, -1)
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def crop_field_value(image: np.ndarray, band: FieldBand) -> np.ndarray:
    """Découpe la zone de valeur d'une bande (pointillés effacés).

    La zone s'étend de ``dots_x0 - 10 px`` (la valeur manuscrite peut
    chevaucher les pointillés) jusqu'à 95 % de la largeur, sur la hauteur
    de la bande +- marge. Les pointillés du band sont blancs (cachés).
    """
    if image is None or image.size == 0 or band is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    pad_up, pad_down = _line_pad_y(height)
    x0 = max(0, band.dots_x0 - _VALUE_PAD_X)
    x1 = min(width, int(width * _VALUE_MAX_X))
    y0 = max(0, band.y0 - pad_up)
    y1 = min(height, band.y1 + pad_down)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    crop = image[y0:y1, x0:x1].copy()
    mask = crop_dotted_mask(crop, work_scale=700)
    if mask.shape[:2] == crop.shape[:2]:
        if crop.ndim == 3:
            crop[mask.astype(bool)] = (255, 255, 255)
        else:
            crop[mask.astype(bool)] = 255
    return crop


def tight_ink_crop(crop: np.ndarray, *, pad: int = 12) -> np.ndarray:
    """Recadre une zone sur son encre (bordures blanches retirées).

    Les moteurs de transcription (TrOCR notamment) hallucinent sur les
    zones larges quasi vides ; cette découpe ne garde que l'écriture
    (pixels sombres) avec une marge de ``pad`` px.
    """
    if crop is None or crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ys, xs = np.where(gray < 200)
    if ys.size == 0:
        return crop
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(crop.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(crop.shape[1], int(xs.max()) + pad + 1)
    return crop[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# Libellés → clés de champ
# --------------------------------------------------------------------------- #
def match_field_label(text: str) -> Optional[str]:
    """Reconnaît le libellé d'un champ depuis son texte OCR.

    S'appuie sur le dictionnaire d'alias de ``form_analyzer`` (``nom``,
    ``prénom``, ``date et lieu de naissance``…). Le meilleur candidat est
    l'alias le plus long contenu (normalisé) — ex. ``date de naissance``
    gagne sur ``date`` (date_concours).
    """
    from .form_analyzer import FORM_FIELDS, _normalize_keyword

    raw = str(text or "").strip()
    if not raw:
        return None
    norm = _normalize_keyword(raw)
    best: tuple[int, str, str] = (0, "", "")
    for spec in FORM_FIELDS:
        for alias in spec.aliases:
            na = _normalize_keyword(alias)
            if na and na in norm and len(na) > best[0]:
                best = (len(na), spec.key, na)
    return best[1] or None


# --------------------------------------------------------------------------- #
# Class nombre (MNIST ONNX local, repli rec par case)
# --------------------------------------------------------------------------- #
class DigitClassifier:
    """Classe un chiffre manuscrit via MNIST en ONNX (CPU).

    Le modèle ``mnist-8.onnx`` est recherché dans ``models/digits/`` (racine
    du projet) ou via la variable d'environnement ``SCRIPTVAULT_DIGITS_MODEL``.
    S'il est absent, ``available`` vaut False : les appels ``classify``
    basculent sur la transcription du chiffre par le callback rec du moteur.
    """

    def __init__(
        self,
        model_path: Optional[os.PathLike[str] | str] = None,
        *,
        recognizer: Optional[Callable[[np.ndarray], list[dict[str, Any]]]] = None,
    ) -> None:
        self.recognizer = recognizer
        self._session: Any = None
        self._input_name = ""
        path = model_path or os.environ.get("SCRIPTVAULT_DIGITS_MODEL")
        if path is None:
            root = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(root, "..", "..", "models", "digits", "mnist-8.onnx")
        path = os.path.abspath(path)
        if os.path.exists(path):
            try:
                import onnxruntime as ort

                self._session = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                logging.getLogger("scriptvault.digits").info(
                    "Classifieur MNIST initialisé (%s).", path
                )
            except Exception as exc:
                logging.getLogger("scriptvault.digits").warning(
                    "Classifieur MNIST indisponible (%s).", exc
                )
                self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    @staticmethod
    def _cell_to_tensor(cell: np.ndarray) -> Optional[np.ndarray]:
        """Convertit une case en tenseur MNIST 1x1x28x28 (encre sans cadre)."""
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = binary.shape
        erode = max(2, min(h, w) // 14)
        frame = np.zeros_like(binary)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), 255, erode * 2)
        frame_ink = cv2.bitwise_and(binary, frame)
        content = cv2.bitwise_and(binary, cv2.bitwise_not(frame_ink))
        coords = cv2.findNonZero(content)
        if coords is None or len(coords) < 15:
            return None
        bx, by, bw, bh = cv2.boundingRect(coords)
        pad = 3
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        digit = content[y0:y1, x0:x1]
        if digit.size == 0:
            return None
        side = max(digit.shape)
        canvas = np.zeros((side, side), np.uint8)
        ox, oy = (side - digit.shape[1]) // 2, (side - digit.shape[0]) // 2
        canvas[oy : oy + digit.shape[0], ox : ox + digit.shape[1]] = digit
        resized = cv2.resize(canvas, (20, 20), interpolation=cv2.INTER_AREA)
        out = np.zeros((28, 28), np.float32)
        out[4:24, 4:24] = resized.astype(np.float32) / 255.0
        return out

    def classify(self, cell: np.ndarray) -> tuple[str, float] | None:
        """Retourne ``(digit, confiance)`` ou ``None`` (case vide / repli vide)."""
        session = self._session
        if session is None:
            return self._classify_fallback(cell)
        tensor = self._cell_to_tensor(cell)
        if tensor is None:
            return None
        out = session.run(None, {self._input_name: tensor[None, None, :, :]})[0]
        probs = np.exp(out[0] - np.max(out[0]))
        probs /= probs.sum()
        digit = int(np.argmax(probs))
        return str(digit), float(probs.max())

    def _classify_fallback(self, cell: np.ndarray) -> tuple[str, float] | None:
        """Repli : transcription de la case via le callback rec du moteur."""
        if self.recognizer is None:
            return None
        try:
            items = self.recognizer(cell)
        except Exception:
            return None
        for item in items or []:
            text = str(item.get("text", "")).strip()
            digits = "".join(ch for ch in text if ch.isdigit())
            if not digits:
                continue
            return digits[0], float(item.get("confidence", 0.0))
        return None


# --------------------------------------------------------------------------- #
# Détection de structure (mode auto)
# --------------------------------------------------------------------------- #
def has_form_structure(image: np.ndarray) -> bool:
    """True si la page ressemble à un formulaire (grilles ou lignes guides)."""
    try:
        grids = extract_digit_grids(image)
        bands = detect_dotted_bands(image)
        return bool(grids) or len(bands) >= 3
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Orchestrateur : lecture complète par zones
# --------------------------------------------------------------------------- #
def read_exam_form_zones(
    image: np.ndarray,
    recognize_crop: Callable[[np.ndarray], list[dict[str, Any]]],
    *,
    recognize_crops: Optional[
        Callable[[list[np.ndarray]], list[list[dict[str, Any]]]]
    ] = None,
    digit_classifier: Optional[DigitClassifier] = None,
    max_side: int = _WORK_SIDE,
    htr_recognize: Optional[HandwrittenRecognizer] = None,
    handwritten_reader: Optional[HandwrittenReader] = None,
    handwritten_fields: tuple[str, ...] = ("nom", "prenom"),
    band_grid_reader: Optional[BandGridReader] = None,
) -> list[dict[str, Any]]:
    """Lit une feuille d'examen zone par zone (jamais la page entière).

    Args:
        image: Page (BGR). Les coordonnées retournées sont alignées dessus.
        recognize_crop: transcrit une petite zone → ``[{"text", "confidence"}]``.
        recognize_crops: variante **batch** (une seule passe pour plusieurs
            zones, ex. Paddle) — utilisée dans l'ordre si fournie.
        digit_classifier: classifieur MNIST (sinon auto + repli rec).
        max_side: plus grand côté de l'image de travail OpenCV.
        htr_recognize: transcrit une zone manuscrite → ``(texte, confiance)``
            (moteur TrOCR). Utilisé pour les champs manuscrits (nom/prénom)
            où PP-OCR est faible.
        handwritten_reader: lecteur de champ manuscrit **conscient du champ**
            ``(crop, field_type) -> (texte, confiance)`` (ex. VLM local) —
            prioritaire sur ``htr_recognize`` : chaque champ reçoit son type
            (``nom``, ``prenom``, ``etablissement``) pour un prompt
            contextuel. Son repli interne (TrOCR/PP-OCR) préserve la chaîne.
        handwritten_fields: champs rédigés à la main relus par le lecteur
            manuscrit (``nom``/``prenom`` par défaut ; ``etablissement`` en
            plus quand le lecteur VLM est actif).
        band_grid_reader: lecteur de grille de bandes
            ``(grille, première, dernière) -> [(index, texte, confiance)]``
            (VLM local). Quand il est fourni, les bandes sont lues en un
            appel VLM (grille numérotée) au lieu de la passe composite OCR —
            lecture « comme Gemini » : chaque ligne « libellé : valeur » est
            transcrite correctement même quand TrOCR hallucine. ``None`` en
            retour = repli sur le chemin composite TrOCR.

    Returns:
        Items type ``[{"text", "confidence", "box", "label"}]`` — les clés
        ``label`` alimentent la passe 0 de ``form_analyzer``.
    """
    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return []

    items: list[dict[str, Any]] = []
    classifier = digit_classifier or DigitClassifier(recognizer=recognize_crop)

    # --- 1) Grilles de chiffres : CIN / Identifiant / Série --------------- #
    grids = extract_digit_grids(image)
    for grid in grids:
        if not grid.label:
            continue
        digits: list[str] = []
        confs: list[float] = []
        boxes: list[list[list[int]]] = []
        for cell in grid.cells:
            result = classifier.classify(
                image[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
            )
            if result is None:
                continue
            digit, conf = result
            digits.append(digit)
            confs.append(conf)
            boxes.append(
                [
                    [cell.x, cell.y],
                    [cell.x + cell.w, cell.y],
                    [cell.x + cell.w, cell.y + cell.h],
                    [cell.x, cell.y + cell.h],
                ]
            )
        if not digits:
            continue
        items.append(
            {
                "text": "".join(digits),
                "confidence": round(sum(confs) / len(confs), 4),
                "box": _union_box(boxes),
                "label": grid.label,
            }
        )

    # --- 2) Bandes pointillées : libellé + valeur ------------------------- #
    # Une bande = une rangée de formulaire. On transcrit la **ligne entière**
    # (sans masquage) ; le découpage « libellé : valeur » est assuré par
    # form_analyzer. Toutes les rangées sont empilées dans **une seule image
    # composite** : le moteur OCR n'exécute qu'UNE passe de détection au lieu
    # d'une par bande (pivot de vitesse 20 s → 2-4 s sur CPU).
    bands = detect_dotted_bands(image, max_side=max_side)
    bands = [
        band
        for band in bands
        if not any(_band_overlaps_grid(band, grid) for grid in grids)
    ]

    crop_pairs: list[tuple[FieldBand, np.ndarray]] = []
    for band in bands:
        crop = _band_row_crop(image, band)
        if crop.size == 0:
            continue
        crop_pairs.append((band, crop))

    items.extend(
        _transcribe_band_rows(
            image,
            crop_pairs,
            recognize_crop,
            recognize_crops,
            htr_recognize,
            handwritten_reader,
            handwritten_fields,
            band_grid_reader,
        )
    )

    return items


#: Écart vertical (px, repère page) entre deux rangées empilées du composite.
_COMPOSITE_GAP_Y = 48

#: Largeur max (px) d'une rangée envoyée au réseau (pivot de vitesse).
_COMPOSITE_MAX_W = 1400


def _transcribe_band_rows(
    image: np.ndarray,
    pairs: list[tuple[FieldBand, np.ndarray]],
    recognize_crop: Callable[[np.ndarray], list[dict[str, Any]]],
    recognize_crops: Optional[
        Callable[[list[np.ndarray]], list[list[dict[str, Any]]]]
    ],
    htr_recognize: Optional[HandwrittenRecognizer] = None,
    handwritten_reader: Optional[HandwrittenReader] = None,
    handwritten_fields: tuple[str, ...] = ("nom", "prenom"),
    band_grid_reader: Optional[BandGridReader] = None,
) -> list[dict[str, Any]]:
    """Transcrit les rangées en une seule passe réseau (image composite).

    Chaque rangée est réduite à ``_COMPOSITE_MAX_W`` puis empilée avec un
    écart ; le réseau OCR (détection + reconnaissance) ne traite qu'UNE
    image. Les items sont ensuite redistribués à leur bande par position Y
    et les coordonnées sont remises à l'échelle du repère page.

    Les champs manuscrits (Nom/Prénom) sont **retranscrits par le moteur
    HTR** (TrOCR) si ``htr_recognize`` est fourni, ou par le lecteur
    ``handwritten_reader`` (VLM local, prioritaire) : PP-OCR lit mal les
    lettres manuscrites.

    Repli : si la passe composite échoue (ex. modèle HTR saturé), les rangées
    sont transcrites individuellement (comportement historique).

    **Lecture grille VLM** (``band_grid_reader``) : quand le moteur OCR local
    lit mal les bandes (TrOCR hallucine sur ce scanner), la grille numérotée
    est lue en un appel VLM (comme le ferait Gemini) : chaque ligne « libellé
    : valeur » est lue telle quelle. Si le lecteur échoue (``None``), le
    chemin composite TrOCR historique reprend.
    """
    if not pairs:
        return []
    height, width = image.shape[:2]
    scale = min(1.0, _COMPOSITE_MAX_W / max(1, width))

    rows: list[np.ndarray] = []
    metas: list[tuple[FieldBand, int, float]] = []
    for band, crop in pairs:
        top = max(0, band.y0 - _line_pad_y(height)[0])
        if scale < 1.0:
            crop = cv2.resize(
                crop,
                (max(1, int(round(crop.shape[1] * scale))),
                 max(1, int(round(crop.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        rows.append(crop)
        metas.append((band, top, scale))

    composite = _stack_composite(rows)

    def _map_back(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        yc = 0
        ranges: list[tuple[FieldBand, int, float, int, int]] = []
        for (band, top, s), row in zip(metas, rows):
            ranges.append((band, top, s, yc, yc + row.shape[0]))
            yc += row.shape[0] + _COMPOSITE_GAP_Y
        for entry in items:
            box = entry.get("box")
            if box:
                cy = sum(pt[1] for pt in box) / 4.0
                band, top, s, yc0, _ = min(
                    ranges, key=lambda r: abs(cy - (r[3] + r[4]) / 2.0)
                )
                entry["box"] = [
                    [int(round(x / s)), int(round((y - yc0) / s + top))]
                    for x, y in box
                ]
            else:
                entry["box"] = _value_box(ranges[0][0], width, height)
            out.append(entry)
        return out

    def _assign_labels(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_labels: dict[str, float] = {}
        result: list[dict[str, Any]] = []
        for entry in items:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            if ":" in text or "：" in text:
                key = match_field_label(text)
            else:
                key = None
            if key and key == "cin":
                key = None  # le CIN vient de la grille de cases (MNIST)
            if key:
                conf = float(entry.get("confidence", 0.0))
                if seen_labels.get(key, 0.0) >= conf:
                    continue  # même champ déjà lu avec une meilleure confiance
                seen_labels[key] = conf
                entry["label"] = key
            result.append(entry)
        return result

    try:
        if band_grid_reader is not None:
            mapped = _read_bands_via_grid(band_grid_reader, rows, metas, width, height)
            if mapped is not None:
                mapped = _assign_labels(mapped)
                mapped = _re_read_incomplete_bands(
                    image,
                    pairs,
                    rows,
                    metas,
                    mapped,
                    band_grid_reader,
                    recognize_crops,
                    recognize_crop,
                    handwritten_reader,
                    handwritten_fields,
                )
                return mapped
            logger.warning("Lecture grille VLM indisponible ; repli composite TrOCR.")
        if recognize_crops is not None:
            results = recognize_crops([composite])
            mapped = _map_back(results[0] if results else [])
        else:
            mapped = _map_back(recognize_crop(composite))
        mapped = _assign_labels(mapped)
        if handwritten_reader is not None:
            mapped = _re_recognize_handwritten(
                image, pairs, mapped, handwritten_reader, handwritten_fields
            )
        elif htr_recognize is not None:

            def reader(crop: np.ndarray, _field: str) -> tuple[str, float]:
                return htr_recognize(crop)

            mapped = _re_recognize_handwritten(
                image, pairs, mapped, reader, handwritten_fields
            )
        return mapped
    except Exception:
        logger.warning("Passe composite échouée ; repli rangée par rangée.", exc_info=True)

    seen_labels: dict[str, float] = {}
    result: list[dict[str, Any]] = []
    for (band, crop), (_, _, _) in zip(pairs, metas):
        crop_items = recognize_crop(crop)
        for entry in crop_items or []:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            box = entry.get("box")
            if box:
                entry["box"] = [
                    [int(round(x + 0)), int(round(y + max(0, band.y0 - _line_pad_y(height)[0])))]
                    for x, y in box
                ]
            else:
                entry["box"] = _value_box(band, width, height)
            if ":" in text or "：" in text:
                key = match_field_label(text)
            else:
                key = None
            if key and key == "cin":
                key = None
            if key:
                conf = float(entry.get("confidence", 0.0))
                if seen_labels.get(key, 0.0) >= conf:
                    continue
                seen_labels[key] = conf
                entry["label"] = key
            result.append(entry)
    return result


def _stack_composite(rows: list[np.ndarray]) -> np.ndarray:
    """Empile des rangées (même largeur) en une seule image avec écarts."""
    width = max(row.shape[1] for row in rows)
    total_h = sum(row.shape[0] for row in rows) + _COMPOSITE_GAP_Y * (len(rows) - 1)
    canvas = np.full((total_h, width, 3), 255, dtype=np.uint8)
    y = 0
    for row in rows:
        canvas[y : y + row.shape[0], 0 : row.shape[1]] = row
        y += row.shape[0] + _COMPOSITE_GAP_Y
    return canvas


# --------------------------------------------------------------------------- #
# Grille numérotée pour la lecture VLM des bandes
# --------------------------------------------------------------------------- #
#: Hauteur cible (px) d'une rangée dans la grille VLM.
_GRID_ROW_H = 72
#: Écart (px) entre deux rangées de la grille VLM.
_GRID_GAP = 12
#: Largeur (px) de la marge de gauche accueillant le numéro de rangée.
_GRID_MARGIN_W = 120
#: Largeur maximale (px) de la partie « rangée » d'une grille VLM.
_GRID_MAX_W = 1280
#: Nombre maximal de rangées par grille (l'image reste lisible pour le VLM).
_GRID_MAX_ROWS = 16
#: Couleur des numéros de rangée (gris foncé, lisible mais discret).
_GRID_NUM_COLOR = (70, 70, 70)


def _build_band_grid(
    rows: list[np.ndarray],
    row_offset: int,
    *,
    max_rows: int = _GRID_MAX_ROWS,
    row_height: int = _GRID_ROW_H,
) -> list[tuple[np.ndarray, int, int]]:
    """Découpe des rangées en grilles numérotées pour la lecture VLM.

    Chaque grille contient au plus ``max_rows`` rangées (hauteur uniforme
    ``row_height``), numérotées en absolu (``1..N``) dans la marge gauche.
    Retourne ``[(grille, première, dernière), ...]`` — le lecteur VLM reçoit
    les bornes absolues et retourne les lignes lues avec leur index.

    ``row_height`` permet une relecture ciblée d'une seule rangée plus
    grande (image plus nette pour le VLM) que la grille multi-lignes.
    """
    grids: list[tuple[np.ndarray, int, int]] = []
    for start in range(0, len(rows), max_rows):
        chunk = rows[start : start + max_rows]
        resized: list[np.ndarray] = []
        max_w = _GRID_MAX_W - _GRID_MARGIN_W
        for row in chunk:
            scale = row_height / max(1, row.shape[0])
            width = max(1, int(round(row.shape[1] * scale)))
            if width > max_w:
                width = max_w
                scale = width / max(1, row.shape[1])
            height = max(1, int(round(row.shape[0] * scale)))
            resized.append(cv2.resize(row, (width, height), interpolation=cv2.INTER_AREA))
        grid_w = _GRID_MARGIN_W + max(r.shape[1] for r in resized)
        grid_h = sum(r.shape[0] for r in resized) + _GRID_GAP * (len(resized) - 1)
        grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
        y = 0
        for idx, item in enumerate(resized):
            number = start + idx + 1
            cv2.putText(
                grid,
                str(number),
                (10, y + item.shape[0] // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                _GRID_NUM_COLOR,
                2,
            )
            grid[
                y : y + item.shape[0], _GRID_MARGIN_W : _GRID_MARGIN_W + item.shape[1]
            ] = item
            y += item.shape[0] + _GRID_GAP
        grids.append((grid, start + 1, start + len(resized)))
    return grids


def _read_bands_via_grid(
    band_grid_reader: BandGridReader,
    rows: list[np.ndarray],
    metas: list[tuple[FieldBand, int, float]],
    width: int,
    height: int,
) -> Optional[list[dict[str, Any]]]:
    """Lit toutes les bandes via le lecteur de grille VLM (ou ``None``).

    Retourne un item par bande lue : ``text`` = « libellé : valeur » tel que
    le VLM l'a lu (l'étiquetage et le nettoyage de la valeur interviennent en
    aval, dans ``form_analyzer``), ``box`` = rangée entière dans le repère
    page, ``confidence`` du VLM. ``None`` si le lecteur a échoué — le pipeline
    retombe alors sur la passe composite TrOCR historique.
    """
    grids = _build_band_grid(rows, 0)
    found: list[tuple[int, str, float]] = []
    for grid, first, last in grids:
        result = band_grid_reader(grid, first, last)
        if result is None:
            return None
        found.extend(result)
    by_index = {index: (text, conf) for index, text, conf in found}
    items: list[dict[str, Any]] = []
    for i, (band, top, scale) in enumerate(metas):
        text, conf = by_index.get(i + 1, ("", 0.0))
        text = str(text or "").strip()
        if not text:
            continue
        bottom = top + int(round(rows[i].shape[0] / max(1e-6, scale)))
        items.append(
            {
                "text": text,
                "confidence": conf,
                "box": [[0, top], [width, top], [width, bottom], [0, bottom]],
            }
        )
    return items


#: Rangées dont la valeur ne vient PAS de la lecture VLM de la rangée :
#: la valeur est portée par une grille de cases (chiffres) ou absente par
#: conception (zones de signature / anonymat). Pas de relecture inutile.
_NO_REREAD_KEYS: frozenset[str] = frozenset(
    {"cin", "serie", "identifiant", "zone_signature", "anonyme"}
)

#: Champs rédigés à la main (l'OCR PP-OCR y est faible → HTR TrOCR).
#: ``etablissement`` s'y ajoute quand le lecteur VLM est actif (le sigle
#: manuscrit exige le prompt contextuel des acronymes).
_HTR_FIELDS: tuple[str, ...] = ("nom", "prenom")


def _row_text_complete(text: str) -> bool:
    """Vrai si la rangée a une valeur exploitable (« libellé : valeur »).

    Un texte sans valeur (chaîne vide, « libellé : » seul) est considéré
    incomplet : la relecture ciblée tentera de le récupérer.
    """
    text = str(text or "").strip()
    if not text:
        return False
    if ":" not in text and "：" not in text:
        return True
    tail = re.split(r"[:：]", text)[-1].strip()
    return any(ch.isalnum() for ch in tail)


#: Budget maximal (s) d'une relecture VLM mono-ligne : quand le modèle est
#: en rechargement à froid, un appel peut durer ~3 min (timeout de grille) ;
#: la relecture abandonne au budget et laisse le repli TrOCR prendre le relais.
_REREAD_BUDGET_S = 30.0


def _read_single_row(
    band_grid_reader: BandGridReader,
    row: np.ndarray,
    budget_s: float = _REREAD_BUDGET_S,
) -> Optional[tuple[str, float]]:
    """Relit une rangée en grille VLM mono-ligne, bornée par ``budget_s``."""
    result: dict[str, object] = {}

    def _run() -> None:
        try:
            grids = _build_band_grid([row], 0, row_height=_GRID_ROW_H * 2)
            single = band_grid_reader(grids[0][0], 1, 1) if grids else None
            if single:
                result["out"] = (
                    str(single[0][1] or "").strip(),
                    float(single[0][2] or 0.0),
                )
        except Exception as exc:  # pragma: no cover - défensif
            result["err"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=budget_s)
    out = result.get("out")
    if isinstance(out, tuple):
        return out
    return None


def _row_has_value(text: str) -> bool:
    """Vrai si le texte porte une valeur après le séparateur (« libellé : valeur »).

    Plus strict que :func:`_row_text_complete` : un texte sans séparateur
    (ex. une relecture qui n'a capté que le libellé imprimé) n'est pas une
    valeur exploitable.
    """
    text = str(text or "").strip()
    if ":" not in text and "：" not in text:
        return False
    tail = re.split(r"[:：]", text)[-1].strip()
    return any(ch.isalnum() for ch in tail)


def _re_read_incomplete_bands(
    image: np.ndarray,
    pairs: list[tuple[FieldBand, np.ndarray]],
    rows: list[np.ndarray],
    metas: list[tuple[FieldBand, int, float]],
    items: list[dict[str, Any]],
    band_grid_reader: BandGridReader,
    recognize_crops: Optional[
        Callable[[list[np.ndarray]], list[list[dict[str, Any]]]]
    ],
    recognize_crop: Callable[[np.ndarray], list[dict[str, Any]]],
    handwritten_reader: Optional[HandwrittenReader] = None,
    handwritten_fields: tuple[str, ...] = _HTR_FIELDS,
) -> list[dict[str, Any]]:
    """Relecture ciblée des rangées que la grille VLM a manquées ou rendues vides.

    Le VLM peut sauter une ligne (ou renvoyer « libellé : » sans valeur),
    ce qui laissait un champ « non lu » en aval. Chaque rangée incomplète
    est relue en cascade, du plus fiable au plus économique :

    1. **Grille VLM mono-ligne** : une seule rangée agrandie (image plus
       nette) relue par le VLM ;
    2. **Passe réseau locale** : PP-OCR sur la rangée (excellent sur
       l'imprimé, lit au moins le libellé) ;
    3. **Lecteur manuscrit dédié** : pour les champs ``nom``/``prenom``/
       ``etablissement``, la zone de valeur est relue avec le prompt
       contextuel du champ (VLM local, repli TrOCR).

    Les rangées sans valeur par conception (``_NO_REREAD_KEYS``) et celles
    déjà complètes ne coûtent aucun appel supplémentaire.
    """
    height, width = image.shape[:2]
    by_index: dict[int, dict[str, Any]] = {}
    for entry in items:
        box = entry.get("box")
        if not box:
            continue
        top = box[0][1]
        for idx, (_band, meta_top, _scale) in enumerate(metas):
            if abs(top - meta_top) < 8:
                by_index[idx] = entry
                break

    def _row_box(idx: int) -> list[list[int]]:
        _band, top, scale = metas[idx]
        bottom = top + int(round(rows[idx].shape[0] / max(1e-6, scale)))
        return [[0, top], [width, top], [width, bottom], [0, bottom]]

    repaired: list[dict[str, Any]] = []
    for idx, (band, _crop) in enumerate(pairs):
        item = by_index.get(idx)
        text = str(item.get("text", "") if item else "").strip()
        if _row_text_complete(text):
            if item is not None:
                repaired.append(item)
            continue
        key = item.get("label") if item else None
        if key is None and text:
            key = match_field_label(text)
        if key in _NO_REREAD_KEYS:
            continue
        conf = float(item.get("confidence", 0.0)) if item else 0.0
        label = key
        new_text = ""
        handwritten_ok = False

        # 1 — Relecture VLM mono-ligne (rangée agrandie, budget borné).
        single = _read_single_row(band_grid_reader, rows[idx])
        if single:
            new_text, re_conf = single
            conf = re_conf
            label = label or match_field_label(new_text)

        # 2 — Passe réseau locale sur la rangée (libellé imprimé).
        if not _row_has_value(new_text):
            try:
                local: list[list[dict[str, Any]]] = (
                    recognize_crops([rows[idx]]) if recognize_crops is not None else []
                )
                if not local:
                    local = [recognize_crop(rows[idx]) or []]
                parts = [
                    str(entry.get("text", "")).strip()
                    for entry in local[0]
                    if str(entry.get("text", "")).strip()
                ]
                if parts:
                    new_text = " ".join(parts)
                    label = label or match_field_label(new_text)
                    conf = max(
                        (float(entry.get("confidence", 0.0)) or 0.0) for entry in local[0]
                    )
            except Exception:
                logger.warning(
                    "Relecture locale bande %d échouée.", idx, exc_info=True
                )

        # 3 — Lecteur manuscrit dédié sur la zone de valeur.
        if (
            not _row_has_value(new_text)
            and handwritten_reader is not None
            and label in handwritten_fields
        ):
            try:
                value_crop = crop_field_value(image, band)
                if value_crop.size > 0:
                    htr_text, htr_conf = handwritten_reader(value_crop, label)
                    htr_text = str(htr_text or "").strip()
                    if len(htr_text) >= 2:
                        new_text = htr_text
                        conf = float(htr_conf or 0.0)
                        handwritten_ok = True
            except Exception:
                logger.warning(
                    "Relecture manuscrite bande %d échouée.", idx, exc_info=True
                )

        # Ne garde que les relectures portant une vraie valeur : un « libellé
        # seul » (ex. zone de signature, bande CIN sans valeur) n'apporte que
        # du bruit au formulaire.
        if not new_text:
            continue
        if not (_row_has_value(new_text) or handwritten_ok):
            continue
        if item is not None:
            item["text"] = new_text
            item["confidence"] = round(conf, 4)
            if label:
                item["label"] = label
            repaired.append(item)
        else:
            repaired.append(
                {
                    "text": new_text,
                    "confidence": round(conf, 4),
                    "box": _row_box(idx),
                    **({"label": label} if label else {}),
                }
            )
    return repaired


#: Champs rédigés à la main (l'OCR PP-OCR y est faible → HTR TrOCR).
#: ``etablissement`` s'y ajoute quand le lecteur VLM est actif (le sigle
#: manuscrit exige le prompt contextuel des acronymes).
def _re_recognize_handwritten(
    image: np.ndarray,
    pairs: list[tuple[FieldBand, np.ndarray]],
    items: list[dict[str, Any]],
    handwritten_reader: HandwrittenReader,
    handwritten_fields: tuple[str, ...] = _HTR_FIELDS,
) -> list[dict[str, Any]]:
    """Retranscrit les champs manuscrits avec le lecteur ``handwritten_reader``.

    Le réseau PP-OCR est excellent sur l'imprimé mais faible sur les lettres
    manuscrites (ex. ``Nom : ... Ellmi`` → ``"Nom:.D"``). Pour chaque rangée
    étiquetée dans ``handwritten_fields`` (``nom``, ``prenom``,
    ``etablissement``), la zone de valeur (à droite de la ligne pointillée)
    est découpée puis transcrite par le lecteur — qui reçoit **le type du
    champ** (``field_type``) pour son prompt contextuel (VLM local) et bascule
    sur son repli (TrOCR/PP-OCR) en cas d'échec. Si la lecture est fiable
    (≥ 2 lettres, confiance ≥ 0.5), elle remplace la lecture composite.
    """
    if not items or not pairs:
        return items
    by_key = {entry.get("label"): entry for entry in items}
    changed: dict[str, tuple[str, float]] = {}
    for label in handwritten_fields:
        entry = by_key.get(label)
        if entry is None:
            continue
        box = entry.get("box")
        if not box:
            continue
        band = _band_for_box(pairs, box)
        if band is None:
            continue
        crop = crop_field_value(image, band)
        if crop.size == 0:
            continue
        try:
            text, conf = handwritten_reader(crop, label)
        except Exception:
            logger.warning("Relecture manuscrite %s échouée.", label, exc_info=True)
            continue
        text = str(text or "").strip()
        letters = sum(1 for ch in text if ch.isalpha())
        if len(text) >= 2 and letters >= 2 and conf >= 0.5:
            changed[label] = (text, conf)
    if not changed:
        return items
    result: list[dict[str, Any]] = []
    for entry in items:
        label = entry.get("label")
        if label in changed:
            text, conf = changed[label]
            replacement = dict(entry)
            replacement["text"] = text
            replacement["confidence"] = round(conf, 4)
            result.append(replacement)
        else:
            result.append(entry)
    return result


def _band_for_box(
    pairs: list[tuple[FieldBand, np.ndarray]], box: Any
) -> Optional[FieldBand]:
    """Bande la plus proche du centre Y d'une boîte (repère page)."""
    if not box:
        return None
    cy = sum(pt[1] for pt in box) / 4.0
    best: Optional[FieldBand] = None
    best_dist = float("inf")
    for band, _crop in pairs:
        dist = abs(band.y_center - cy)
        if dist < best_dist:
            best, best_dist = band, dist
    return best


def _band_row_crop(image: np.ndarray, band: FieldBand) -> np.ndarray:
    """Crop d'une rangée entière de formulaire (sans masquage pointillés).

    Le masquage des pointillés dans la zone binarisée à petite échelle
    avale aussi les traits manuscrits — on transcrit donc la ligne brute :
    Paddle lit ``"Nom :. Ellommi..."`` et ``form_analyzer`` nettoie.
    """
    height, width = image.shape[:2]
    pad_up, pad_down = _line_pad_y(height)
    y0 = max(0, band.y0 - pad_up)
    y1 = min(height, band.y1 + pad_down)
    return image[y0:y1, 0 : int(width * _VALUE_MAX_X)]


def _band_overlaps_grid(band: FieldBand, grid: DigitGrid) -> bool:
    """True si la bande recouvre verticalement la rangée d'une grille."""
    if not grid.cells:
        return False
    top = min(c.y for c in grid.cells)
    bottom = max(c.y + c.h for c in grid.cells)
    return band.y_center >= top - _GRID_BAND_TOL_PX and band.y_center <= bottom + _GRID_BAND_TOL_PX


def _crop_label(image: np.ndarray, band: FieldBand) -> np.ndarray:
    """Zone du libellé : gauche de la ligne pointillée (jusqu'à 60 % largeur)."""
    height, width = image.shape[:2]
    pad_up, pad_down = _line_pad_y(height)
    x1 = max(10, band.dots_x0 + 20)
    x1 = min(x1, int(width * _LABEL_MAX_X))
    y0 = max(0, band.y0 - pad_up)
    y1 = min(height, band.y1 + pad_down)
    if x1 <= 0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y0:y1, 0:x1]


def _recognize_text(
    recognize_crop: Callable[[np.ndarray], list[dict[str, Any]]],
    crop: np.ndarray,
) -> str:
    if crop is None or crop.size == 0:
        return ""
    items = recognize_crop(crop)
    return " ".join(str(item.get("text", "")).strip() for item in (items or [])).strip()


def _value_box(band: FieldBand, width: int, height: int) -> list[list[int]]:
    """Boîte rectangle de la zone de valeur (repère page)."""
    pad_up, pad_down = _line_pad_y(height)
    x0 = max(0, band.dots_x0 - _VALUE_PAD_X)
    x1 = min(width, int(width * _VALUE_MAX_X))
    y0 = max(0, band.y0 - pad_up)
    y1 = min(height, band.y1 + pad_down)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _union_box(boxes: list[list[list[int]]]) -> list[list[int]]:
    """Rectangle englobant des cases (repère page)."""
    if not boxes:
        return []
    x0 = min(pt[0] for box in boxes for pt in box)
    y0 = min(pt[1] for box in boxes for pt in box)
    x1 = max(pt[0] for box in boxes for pt in box)
    y1 = max(pt[1] for box in boxes for pt in box)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
