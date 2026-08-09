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
]

logger = logging.getLogger("scriptvault.image_processing")

#: Taille de travail commune aux détecteurs OpenCV (économie CPU).
_WORK_SIDE = 1400

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
) -> list[dict[str, Any]]:
    """Lit une feuille d'examen zone par zone (jamais la page entière).

    Args:
        image: Page (BGR). Les coordonnées retournées sont alignées dessus.
        recognize_crop: transcrit une petite zone → ``[{"text", "confidence"}]``.
        recognize_crops: variante **batch** (une seule passe pour plusieurs
            zones, ex. Paddle) — utilisée dans l'ordre si fournie.
        digit_classifier: classifieur MNIST (sinon auto + repli rec).
        max_side: plus grand côté de l'image de travail OpenCV.

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
            image, crop_pairs, recognize_crop, recognize_crops
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
) -> list[dict[str, Any]]:
    """Transcrit les rangées en une seule passe réseau (image composite).

    Chaque rangée est réduite à ``_COMPOSITE_MAX_W`` puis empilée avec un
    écart ; le réseau OCR (détection + reconnaissance) ne traite qu'UNE
    image. Les items sont ensuite redistribués à leur bande par position Y
    et les coordonnées sont remises à l'échelle du repère page.

    Repli : si la passe composite échoue (ex. modèle HTR saturé), les rangées
    sont transcrites individuellement (comportement historique).
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
                cx = sum(pt[0] for pt in box) / 4.0
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
            key = match_field_label(text) if ":" in text else None
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
        if recognize_crops is not None:
            results = recognize_crops([composite])
            mapped = _map_back(results[0] if results else [])
        else:
            mapped = _map_back(recognize_crop(composite))
        return _assign_labels(mapped)
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
            key = match_field_label(text) if ":" in text else None
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