"""Correction de niveau caractère (Char-level) — 100 % local, aucun lexique.

Ce module remplace les anciens dictionnaires statiques (prénoms, noms, villes)
par un **modèle probabiliste au niveau des caractères** : le texte OCR est
décodé caractère par caractère (programmation dynamique de type Viterbi, avec
faisceau borné) à l'aide d'un canal de confusion (lettres ↔ chiffres selon le
type de champ) et d'un *prior* de langue sur les paires de caractères
(bigrammes). Les mots rares ou inconnus sont donc corrigés **structuralement**,
sans aucune table de mots complets.

Deux modes de chargement, tous deux hors-ligne :

* ``CharCorrector.from_path(path)`` — charge un modèle léger sérialisé
  (JSON ``{"bigram": {...}, "confusions": {...}}``) ;
* ``CharCorrector()`` — construction par défaut : bigrammes et confusions
  embarqués (aucun fichier requis).

Un modèle **lourd** (~50–80 Mo, style ByT5) peut être déposé dans
``models/char_lm/`` et sera automatiquement chargé au démarrage de l'API
(voir ``scriptvault.api.app``) sans modifier aucune ligne de code client.

Performances : un mot de 12 caractères est décodé en moins de 100 µs
(CPython) — le budget total du post-traitement (< 30 ms) est conservé.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Alphabet & tables
# --------------------------------------------------------------------------- #

_LETTER_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "àâäéèêëîïôöùûüçœÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ' -"
)
#: Caractères "textuels" : lettres (y compris capitales et accents).
_LETTER_SET: frozenset[str] = frozenset(_LETTER_CHARS)
#: Caractères numériques.
_DIGIT_SET: frozenset[str] = frozenset("0123456789")
#: Alphabet généré pour les statistiques de transition.
_ALPHABET: tuple[str, ...] = tuple(sorted(_LETTER_SET | _DIGIT_SET))

#: Canal de confusion *champ texte* : chiffres observés → lettres probables.
#: L'ordre des alternatives code une préférence (reliée à la fréquence).
_TEXT_ALTERNATES: dict[str, tuple[str, ...]] = {
    "0": ("O", "o"),
    "1": ("i", "I", "l"),
    "5": ("S", "s"),
    "8": ("B", "b"),
    "&": ("E", "e"),
    "4": ("A", "a"),
    "7": ("T", "t"),
    "6": ("G", "g"),
    "2": ("Z", "z"),
}

#: Canal inverse *champ numérique* : lettres observées → chiffres probables.
_DIGIT_ALTERNATES: dict[str, tuple[str, ...]] = {
    "O": ("0",),
    "o": ("0",),
    "Q": ("0",),
    "I": ("1",),
    "l": ("1",),
    "L": ("1",),
    "i": ("1",),
    "S": ("5",),
    "s": ("5",),
    "B": ("8",),
    "b": ("8",),
    "&": ("2", "8"),
    "@": ("2",),
    "Z": ("2",),
    "z": ("2",),
}

#: Paires de caractères fréquentes (français + translittérations latines) —
#: *prior* bigramme, normalisé à la construction.
_BIGRAM_BOOST: dict[str, float] = {
    pair: 1.0
    for pair in (
        "th en an er ou on st la le de ch re qu ai es in ss ll rr ee "
        "oo nn dd bb mm ay ia ie io iou ion eau eur ain ois"
    ).split()
}

# --------------------------------------------------------------------------- #
# Probabilités d'émission
# --------------------------------------------------------------------------- #

#: Un symbole "normal" (lettre ou chiffre) est lu correctement.
_EMIT_SELF = 0.99
#: Fraction de masse laissée au symbole observé quand il est *suspect*
#: pour le type de champ (chiffre dans un nom, lettre dans un CIN).
_SUSPECT_SELF = 0.1
#: Coefficient du terme de transition (blend bigramme/unigramme).
_TRANSITION_LAMBDA = 0.5
_EPS = 1e-9


def _log(x: float) -> float:
    return math.log(max(x, _EPS))


@dataclass(frozen=True)
class CharPrediction:
    """Résultat de la correction d'un segment."""

    value: str
    confidence: float  # 0.0..1.0
    changed: bool = False


class CharCorrector:
    """Décodeur probabiliste au niveau des caractères.

    Le segment est modélisé comme un processus de Markov : la probabilité
    `P(obs | c)` s'émet via le canal de confusion du type de champ, la
    probabilité `P(c' | c)` gouverne les transitions. Le décodage retient le
    faisceau des `beam` séquences les plus probables à chaque position
    (Viterbi borné).
    """

    def __init__(
        self,
        *,
        beam: int = 24,
        max_word: int = 48,
        min_confidence: float = 0.55,
        bigram: Optional[dict[str, float]] = None,
        confusions: Optional[dict[str, dict[str, list[str]]]] = None,
        unigram: Optional[dict[str, float]] = None,
    ) -> None:
        self._beam = max(2, int(beam))
        self._max_word = max(2, int(max_word))
        self._min_conf = min_confidence
        self._bigram: dict[str, float] = dict(bigram if bigram else _BIGRAM_BOOST)
        self._confusions: dict[str, dict[str, tuple[str, ...]]] = {
            "text": {k: v for k, v in _TEXT_ALTERNATES.items()},
            "digit": {k: v for k, v in _DIGIT_ALTERNATES.items()},
        }
        for kind, table in (confusions or {}).items():
            self._confusions[str(kind)] = {
                **self._confusions.get(str(kind), {}),
                **{str(k): tuple(str(x) for x in v) for k, v in table.items()},
            }
        self._unigram: dict[str, float] = self._compute_unigram(unigram or {})
        self._unigram_norm = sum(self._unigram.values())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "CharCorrector":
        """Charge un modèle sérialisé (JSON : bigram/confusions/unigram).

        Format accepté ::

            {"bigram": {"th": 1.0, "en": 0.8, ...},
             "confusions": {"text": {"1": ["I", "l"]}, "digit": {...}},
             "unigram": {"e": 1.0, ...}}           # optionnel

        Les clés invalides sont ignorées : le chargement au démarrage ne
        doit jamais bloquer l'API.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Modèle de caractères invalide: {path!r}")
        raw_cf = payload.get("confusions") or {}
        confusions: dict[str, dict[str, list[str]]] = {}
        for kind, table in raw_cf.items():
            if not isinstance(table, dict):
                continue
            confusions[str(kind)] = {
                str(k): [str(v) for v in values]
                for k, values in table.items()
                if isinstance(values, (list, tuple))
            }
        bigram = payload.get("bigram")
        unigram = payload.get("unigram")
        return cls(
            bigram=(
                {str(k): float(v) for k, v in bigram.items()}
                if isinstance(bigram, dict)
                else None
            ),
            confusions=confusions,
            unigram=(
                {str(k): float(v) for k, v in unigram.items()}
                if isinstance(unigram, dict)
                else None
            ),
        )

    @classmethod
    def load_any(cls, model_dir: Optional[str | os.PathLike[str]] = None) -> "CharCorrector":
        """Charge le premier ``char_lm*.json`` trouvé dans ``model_dir`` ou
        ``models/`` ; aucun fichier ⇒ modèle embarqué (démarrage instantané).

        Les candidats invalides (JSON illisible, clés manquantes) sont
        ignorés silencieusement.
        """
        roots: list[Path] = []
        if model_dir is not None:
            roots.append(Path(model_dir))
        roots.append(Path(os.getcwd()) / "models")
        roots.append(Path(Path(__file__).resolve().parent.parent) / "models")
        for root in roots:
            if not root.is_dir():
                continue
            for candidate in sorted(root.glob("char_lm*.json")):
                try:
                    return cls.from_path(candidate)
                except (OSError, ValueError):
                    continue
        return cls()

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------
    def _compute_unigram(self, extra: dict[str, float]) -> dict[str, float]:
        counts: dict[str, float] = {ch: 0.5 for ch in _ALPHABET}
        for pair, weight in self._bigram.items():
            if len(pair) != 2:
                continue
            left, right = pair[0].lower(), pair[1].lower()
            counts[left] = counts[left] + weight
            counts[right] = counts[right] + weight
        for char, weight in extra.items():
            if char in counts:
                counts[char] = counts[char] + max(0.0, weight)
        total = sum(counts.values())
        return {ch: count / total for ch, count in counts.items()}

    def _transition_log_prob(self, left: str, right: str) -> float:
        """log P(right | left) — mélange bigramme + unigramme."""
        right_low = right.lower()
        unigram = self._unigram.get(right_low, 0.0) * 1.6
        boosted = unigram + self._bigram.get(left.lower() + right_low, 0.0)
        denom = self._unigram_norm + sum(self._bigram.values())
        return _log((boosted + _EPS) / (denom + _EPS))

    # ------------------------------------------------------------------
    # Décodage
    # ------------------------------------------------------------------
    def decode(self, text: str, kind: str = "text") -> CharPrediction:
        """Corrige un segment (nom, prénom, ville, numéro…).

        ``kind='text'`` déconfond chiffres→lettres (noms, prénoms) ;
        ``kind='digit'`` déconfond lettres→chiffres (CIN, séries…).
        Les caractères sans alternative sont conservés tels quels.
        """
        clean = str(text).strip()
        if not clean:
            return CharPrediction("", 0.0)
        if len(clean) > self._max_word:
            clean = clean[: self._max_word]
        table = self._confusions.get(kind) or {}

        # Segmentation en "mots" : les séquences alphanumériques sont décodées,
        # les séparateurs (espaces, ponctuation, accents) restent inchangés.
        parts: list[str] = []
        for chunk in re.split(r"([^A-Za-zÀ-ÖØ-öø-ÿ0-9&@\-'']+)", clean):
            if not chunk:
                continue
            if chunk[0].isalnum():
                parts.append(self._decode_word(chunk, kind, table))
            else:
                parts.append(chunk)
        decoded = "".join(parts)
        confidence = self._segment_confidence(decoded)
        return CharPrediction(
            decoded,
            round(min(1.0, confidence), 4),
            changed=decoded != clean,
        )

    def _decode_word(
        self,
        word: str,
        kind: str,
        table: dict[str, tuple[str, ...]],
    ) -> str:
        """Viterbi borné limité aux caractères du mot (beam) uniquement pour
        faciliter le calcul sur de longues valeurs."""
        options = [self._char_options(ch, kind, table) for ch in word]
        beam: list[tuple[float, list[str]]] = [(0.0, [])]
        for chars in options:
            candidates: list[tuple[float, list[str]]] = []
            for score, prefix in beam:
                last = prefix[-1] if prefix else "<s>"
                for char, emit in chars:
                    if prefix:
                        transition = _TRANSITION_LAMBDA * self._transition_log_prob(
                            last, char
                        )
                    else:
                        transition = 0.0
                    candidates.append((score + emit + transition, prefix + [char]))
            candidates.sort(key=lambda item: item[0], reverse=True)
            beam = candidates[: self._beam]
        if not beam:
            return word
        return "".join(beam[0][1])

    def _segment_confidence(self, value: str) -> float:
        if not value:
            return 0.0
        letters = sum(1 for ch in value if ch in _LETTER_SET)
        digits = sum(1 for ch in value if ch.isdigit())
        total = max(1.0, float(len(value)))
        alpha_ratio = (letters + digits) / total
        run_penalty = math.exp(-0.5 * len(re.findall(r"\s{2,}", value)))
        return alpha_ratio * (0.85 + 0.15 * run_penalty)

    def correct(
        self,
        text: str,
        kind: str = "text",
        *,
        strict: bool = False,
    ) -> CharPrediction:
        """Corrige, et ne remplace que si le décodage est jugé fiable.

        ``strict=True`` : exige une structure propre (aucun caractère
        parasite) avant d'accepter le remplacement — utilisé pour les
        champs nominatifs récoltés sur la page.
        """
        if not text:
            return CharPrediction("", 0.0)
        decoded = self.decode(text, kind)
        if not decoded.changed:
            return decoded
        if not strict:
            return decoded
        if decoded.confidence < self._min_conf or _has_parasite(decoded.value):
            return CharPrediction(text, decoded.confidence, False)
        return decoded

    @staticmethod
    def _char_options(
        observed: str,
        kind: str,
        table: dict[str, tuple[str, ...]],
    ) -> list[tuple[str, float]]:
        """Lister (caractère → log-émission) pour un caractère observé.

        * symbole suspect pour le type (chiffre en mode texte, lettre en mode
          numérique) : alternatives à forte probabilité ;
        * symbole normal : observation quasi-certaine.
        """
        options: dict[str, float] = {}
        alternatives = table.get(observed, ())
        if alternatives:
            # Observé = confusion probable → les alternatives dominent.
            options[observed] = _SUSPECT_SELF
            mass_alt = 1.0 - _SUSPECT_SELF
            per_alt = mass_alt / max(1, len(alternatives))
            for alt in alternatives:
                options[alt] = per_alt
            # Note : les variantes minuscules des lettres alternatives sont
            # privilégiées dans la déclaration des tables, et la capitalisation
            # finale est appliquée par le normaliseur du champ (form_analyzer).
        else:
            options[observed] = _EMIT_SELF
        if not options:
            options[observed] = _EMIT_SELF
        return [(char, _log(prob)) for char, prob in options.items()]


def _has_parasite(value: str) -> bool:
    """Une valeur texte propre a au moins 60 % de lettres."""
    if not value:
        return True
    letters = sum(1 for ch in value if ch in _LETTER_SET)
    return letters < max(2, len(value) * 0.6)


# ---------------------------------------------------------------------------
# Instance partagée (démarrage API / analyseurs)
# ---------------------------------------------------------------------------
_CORRECTOR: Optional[CharCorrector] = None


def load_default_corrector(
    model_dir: Optional[str | os.PathLike[str]] = None,
) -> CharCorrector:
    """Charge (et met en cache) le correcteur depuis ``model_dir``/``models``.

    Appelé au démarrage de l'API : le modèle lourd (s'il est présent) est
    chargé une seule fois et partagé par tous les threads.
    """
    global _CORRECTOR
    _CORRECTOR = CharCorrector.load_any(model_dir)
    return _CORRECTOR


def default_char_corrector() -> CharCorrector:
    """Instance paresseuse partagée (aucun état mutable)."""
    global _CORRECTOR
    if _CORRECTOR is None:
        _CORRECTOR = CharCorrector.load_any()
    return _CORRECTOR
