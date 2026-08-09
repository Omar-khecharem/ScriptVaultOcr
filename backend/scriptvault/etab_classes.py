"""Liste officielle des établissements (concours) + rapprochement flou.

Le champ « Établissement d'origine » est rempli par un acronyme manuscrit
que l'OCR lit souvent bruité (ex. ``I S EI B``). Au lieu de garder le texte
brut, on rapproche la lecture de la liste officielle des établissements
tunisiens via un ratio de similarité (SequenceMatcher) : l'acronyme canonique
est renvoyé si la similarité dépasse le seuil, sinon ``None`` (la valeur
OCR reste alors inchangée — jamais inventée).
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

#: Acronymes officiels des établissements tunisiens (instituts préparatoires,
#: facultés, ISET, écoles d'ingénieurs…). La valeur lue par l'OCR est
#: rapprochée de cette liste ; ``AUTRE`` sert de classe de repli.
ETAB_CLASSES: tuple[str, ...] = (
    # Instituts Préparatoires
    "IPEIT",
    "IPEIN",
    "IPEIM",
    "IPEIS",
    "IPEIB",
    "IPEIK",
    "IPEIG",
    "IPEIEM",
    # Facultés des Sciences
    "FST",
    "FSB",
    "FSM",
    "FSS",
    "FSG",
    "FSGF",
    # ISSAT & Écoles Techniques
    "INSAT",
    "ISSATSO",
    "ISSATK",
    "ISSATG",
    "ISSATM",
    "ISSATMH",
    "ISSATGF",
    "ESSTHS",
    "ESSTT",
    # Informatique
    "ISI",
    "ISIMA",
    "ISIMS",
    "ISITCOM",
    "ISAMM",
    "ISIK",
    "ISIMGB",
    # Grandes Écoles / Ingénieurs
    "ENIT",
    "ENSI",
    "EPT",
    "SUPCOM",
    "ENICAR",
    "ENISO",
    "ENIM",
    "ENIS",
    "ENIB",
    "ENIG",
    "INAT",
    "ESIAT",
    "ESTI",
    # Réseau ISET
    "ISETR",
    "ISETN",
    "ISETSO",
    "ISETS",
    "ISETB",
    "ISETK",
    "ISETG",
    "ISETGF",
    "ISETKR",
    "ISETMH",
    "ISETCOM",
    "ISETAR",
    "ISETKF",
    "ISETJ",
    "ISETSL",
    "ISETTZ",
    "ISETTT",
    "ISETMD",
    "ISETZ",
    "ISETSB",
    # Classe Fallback
    "AUTRE",
)

#: Seuil minimal de similarité pour accepter un rapprochement.
_ETAB_MIN_RATIO = 0.72


def _normalize_acronym(value: str) -> str:
    """Acronyme canonique : majuscules, sans espaces ni ponctuation."""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def closest_etab(value: str) -> Optional[str]:
    """Retourne l'acronyme officiel le plus proche de ``value``, sinon ``None``.

    La comparaison ignore l'ordre des lettres confondues (ex. ``ISEIB`` →
    ``ISETB``) mais exige une similarité ≥ ``_ETAB_MIN_RATIO`` : une lecture
    trop bruitée n'est jamais remplacée par un acronyme arbitraire.
    """
    probe = _normalize_acronym(value)
    if len(probe) < 2:
        return None
    best: Optional[tuple[str, float]] = None
    for candidate in ETAB_CLASSES:
        canonical = _normalize_acronym(candidate)
        ratio = float(SequenceMatcher(None, probe, canonical).ratio())
        if best is None or ratio > best[1]:
            best = (candidate, ratio)
    if best is None or best[1] < _ETAB_MIN_RATIO:
        return None
    return best[0]
