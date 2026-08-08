"""Moteur de post-traitement OCR : analyse de formulaire clé/valeur locale.

Transforme la liste brute des items PaddleOCR
``[{"text", "confidence", "box"}]`` en paires *Champ → Valeur* structurées,
valide chaque champ avec des règles métier 100 % locales (regex, checksums,
cohérences), puis classe le niveau de risque :

===================  ================================  ======================
Statut               Condition                         Affichage client
===================  ================================  ======================
``valid``            confiance > 85 % et règles OK     vert
``warning``          confiance 70–85 % et format OK    orange
``error``            confiance < 70 % OU règle violée  rouge
===================  ================================  ======================

Aucun appel réseau : le parsing spatial et la validation s'exécutent en
moins de 30 ms par document sur CPU (pur Python, aucune dépendance lourde).

Exemple::

    from form_analyzer import LocalFormAnalyzer

    analyzer = LocalFormAnalyzer()
    response = analyzer.analyze_page(
        items=[
            {"text": "Nom :", "confidence": 0.98, "box": [...]},
            {"text": "Didi", "confidence": 0.96, "box": [...]},
        ],
        file_name="scan_001.png",
    )
    print(response.fields[0].status)  # FieldStatus.VALID
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from scriptvault.lexicons import (
    TUNISIAN_CITIES,
    TUNISIAN_FIRST_NAMES,
    TUNISIAN_LAST_NAMES,
)
from scriptvault.schemas import (
    SECTION_LABELS,
    AnalyzedFormResponse,
    FieldStatus,
    FormFieldResult,
    FormSection,
)

logger: logging.Logger = logging.getLogger("scriptvault.form_analyzer")
logger.addHandler(logging.NullHandler())

__all__ = [
    "LocalFormAnalyzer",
    "FieldSpec",
    "CIN_RE",
    "DATE_RE",
    "DUREE_RE",
    "NAME_RE",
    "COUNT_RE",
    "SERIE_RE",
    "IDENTIFIANT_RE",
]

# --------------------------------------------------------------------------- #
# Regex de validation (Règles Métier)
# --------------------------------------------------------------------------- #
#: CIN / passeport tunisien : exactement 8 chiffres (espaces tolérés).
CIN_RE = re.compile(r"^\d{8}$")

#: Date JJ/MM/AAAA (ou JJ.MM.AAAA ou JJ-MM-AAAA), années 1900–2099.
DATE_RE = re.compile(
    r"^(0[1-9]|[12][0-9]|3[01])[/.-](0[1-9]|1[012])[/.-](?P<year>(?:19|20)\d{2})$"
)

#: Durée d'épreuve : chiffres + unité "h"/"h30"/"heure(s)" (ex. "2", "2h",
#: "2h30", "3 h", "4 Heures").
DUREE_RE = re.compile(r"^\d{1,2}\s*(h(\.)?\s*\d{0,2}|heures?)?$", re.IGNORECASE)

#: Série du sujet : 3 ou 4 chiffres (ex. "531", "5401").
SERIE_RE = re.compile(r"^\d{3,4}$")

#: Identifiant d'inscription : exactement 6 chiffres (ex. "540117").
IDENTIFIANT_RE = re.compile(r"^\d{6}$")

#: Nombre de cahiers remis : entier de 1 à 5 (anomalie au-delà).
COUNT_RE = re.compile(r"^[1-5]$")

#: Nom / Prénom : uniquement alphabétique (accents compris) + tirets/espaces.
NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ' -]{2,}$")

#: Années acceptées pour une date de naissance (cohérence métier).
DATE_YEAR_RANGE = (1950, 2026)

#: Nombres : valeurs reconnues comme purement numériques.
_VALUE_KINDS: frozenset[str] = frozenset(
    {"cin", "serie", "identifiant", "nombre_cahiers", "anonyme"}
)

#: Établissements officiels (prépas tunisiennes + facultés à prépa intégrée)
#: pour la validation par distance de Levenshtein.
_ETABLISSEMENTS: tuple[str, ...] = (
    # Instituts Préparatoires aux Études d'Ingénieurs (tous gouvernorats).
    "IPEIN",
    "IPEIT",
    "IPEIM",
    "IPEIS",
    "IPEIB",
    "IPEST",
    "IPEES",
    # Facultés avec cycle prépa intégré / filières d'ingénieur.
    "FST",
    "FSM",
    "FSB",
    "FSG",
    "FSS",
    # Autres établissements officiels (écoles d'ingénieurs, ISET, etc.).
    "FGES",
    "FSEG",
    "FSEGN",
    "FSEGT",
    "INSAT",
    "ENIT",
    "ESPRIT",
    "ISI",
    "ISET",
    "ISSAT",
    "ISIT",
)

_ETABLISSEMENTS_SET: frozenset[str] = frozenset(_ETABLISSEMENTS)

#: Marge d'encre minimale (fraction de pixels sombres) dans la zone de
#: signature des enseignants pour considérer que la signature est présente.
SIGNATURE_INK_MIN = 0.004

#: Mois français (sans accents) → numéro de mois.
_FRENCH_MONTHS: dict[str, str] = {
    "janvier": "01",
    "fevrier": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "aout": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "decembre": "12",
}

#: Date en toutes lettres : "Lundi 03 Juin 2024 à 8 H", "3 juin 2024"…
_FRENCH_DATE_RE = re.compile(
    r"(?P<weekday>lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)?\s*"
    r"(?P<day>\d{1,2})\s*(?P<month>[A-Za-zÀ-ÖØ-öø-ÿ]+)\s*"
    r"(?P<year>(?:19|20)\d{2})",
    re.IGNORECASE,
)

#: Date numérique libre : jj/mm/aaaa, jj-mm-aa, jj.mm.aaaa…
_DATE_TOKEN_RE = re.compile(
    r"(?P<d1>\d{1,2})\s*[/.\-]\s*(?P<d2>\d{1,2})\s*[/.\-]\s*(?P<y>\d{2,4})"
)

#: Date au format ISO : aaaa-mm-jj.
_ISO_TOKEN_RE = re.compile(
    r"(?P<y>\d{4})\s*[/.\-]\s*(?P<m>\d{1,2})\s*[/.\-]\s*(?P<d>\d{1,2})"
)

#: Année à 4 chiffres apparente dans un texte (dates en toutes lettres).
_YEAR_LIKE_RE = re.compile(r"(?:19|20)\d{2}")

#: Structure complète jour/mois/année (triplet numérique) — la découpe
#: empêche de transformer une date IMPOSSIBLE (« 31/02/2003 ») en simple
#: avertissement : seule une absence de triplet autorise le mode partiel.
_FULL_DATE_TRIPLET_RE = re.compile(
    r"\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*(?:19|20)\d{2}"
)


def _has_full_date_triplet(text: str) -> bool:
    """Vrai si ``text`` porte une structure jour/mois/année complète."""
    probe = text.translate(_OCR_CONFUSION_TABLE).replace("&", "2")
    return _FULL_DATE_TRIPLET_RE.search(probe) is not None


def _year_hint(text: str) -> Optional[str]:
    """Année 4 chiffres lisible après déconfusion OCR (``B``→``8``, ``o``→``0``).

    ``"A2/2oo3.Sax"`` porte bien l'année ``2003`` même si le ``2o`` est
    confondu avec la lettre ``o`` : seule la variante déconfonduée est
    sondée, la valeur originale reste intacte.
    """
    probe = text.translate(_OCR_CONFUSION_TABLE).replace("&", "2")
    match = _YEAR_LIKE_RE.search(probe)
    return match.group() if match is not None else None


#: Confusions chiffres ↔ lettres pour la recherche de séquence numérique.
_DIGIT_CONFUSIONS: dict[str, str] = {
    "O": "0",
    "o": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "L": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "b": "8",
}

# --------------------------------------------------------------------------- #
# Corrections OCR des confusions usuelles Chiffres ↔ Lettres
# --------------------------------------------------------------------------- #
#: Table des substitutions usuelles commises par PaddleOCR (haute confiance)
#: sur les valeurs numériques : ``O``→``0``, ``I/l``→``1``, ``S``→``5``…
_OCR_CONFUSION_TABLE = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "i": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "b": "8",
    }
)


def _normalize_keyword(text: str) -> str:
    """Minuscules, sans accents ni ponctuation (comparaison de labels).

    ``"N° C.I.N :"`` devient ``"n cin"`` : le ``°``, les points et le ``:``
    final sont neutralisés pour un appariement robuste aux variantes.
    """
    value = unicodedata.normalize("NFKD", text.strip().lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("&", "et")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _clean_value(text: str) -> str:
    """Nettoie une valeur OCR : espaces, ponctuation finale parasite."""
    return re.sub(r"[\s_]+", " ", text).strip().strip(".:;,«»\"'")


def _is_noise_value(text: str) -> bool:
    """Fragments de label OCR capturés par erreur comme valeur.

    PaddleOCR confond parfois l'étiquette d'un champ adjacent avec la
    valeur du champ courant (ex. ``"N0m8re de"`` pour « Nombre de »), ou lit
    une parcelle manuscrite comme un glyphe CJK (ex. ``"二"`` pour un
    prénom) : ces valeurs ne doivent jamais remplir un champ.
    """
    lowered = text.strip().lower()
    if re.search(r"\bnombre\b|\bnumber\b|\bn?0?m8?re\b", lowered):
        return True
    # Glyphes CJK / coréens : jamais de valeur exploitable sur une copie
    # d'examen tunisienne (latin uniquement).
    return bool(
        re.search(
            r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
            r"\uac00-\ud7af]",
            text,
        )
    )


def _box_rect(box: Any) -> tuple[float, float, float, float]:
    """Rectangle englobant ``(x0, y0, x1, y1)`` d'une boîte OCR.

    La boîte PaddleOCR est un polygone de 4 points. Les boîtes mal formées ou
    vides retombent sur le centre (0, 0) — le champ reste analysable au
    minimum par le texte seul.
    """
    try:
        points = [(float(p[0]), float(p[1])) for p in box]
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0, 0.0, 0.0
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _levenshtein(a: str, b: str) -> int:
    """Distance d'édition minimale entre deux chaînes (pure Python)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def _french_date_value(text: str) -> Optional[str]:
    """Date en toutes lettres → JJ/MM/AAAA (ex. « Lundi 03 Juin 2024 à 8 H »)."""
    span = _french_date_span(text)
    return span[0] if span is not None else None


def _french_date_span(text: str) -> Optional[tuple[str, int, int]]:
    """Comme :func:`_french_date_value` mais avec les positions du texte."""
    match = _FRENCH_DATE_RE.search(text)
    if match is None:
        return None
    month = unicodedata.normalize("NFKD", match.group("month"))
    month = "".join(ch for ch in month if not unicodedata.combining(ch)).lower()
    try:
        month_num = _FRENCH_MONTHS[month]
    except KeyError:
        return None
    day = int(match.group("day"))
    if not 1 <= day <= 31:
        return None
    normalized = f"{day:02d}/{month_num}/{match.group('year')}"
    return normalized, match.start(), match.end()


#: Paire "jour/mois" : séparés par des séparateurs OCR bruyants.
_DAY_SEP_MONTH_RE = re.compile(r"(?P<d1>\d{1,2})\s*[/.\-|,]\s*(?P<d2>\d{1,2})")

#: Année 4 chiffres qui suit, éventuellement entrecoupée de séparateurs OCR
#: (ex. "200.3" → 2003).
_YEAR_AFTER_RE = re.compile(r"(?:19|20)[\s./\-|,]*\d[\s./\-|,]*\d")


def _day_month_pair(
    day_raw: str, month_raw: str, allow_reverse: bool
) -> tuple[Optional[int], Optional[int]]:
    """Couple (jour, mois) valide, avec repli sur les chiffres inversés.

    PaddleOCR inverse parfois l'ordre des chiffres d'un jour brouillon
    (``"80/09/2003"`` lu pour ``08/09/2003``). Le repli n'est autorisé que
    sur les variantes OCR, jamais sur le texte original : ``"31/02/2003"``
    reste bien une erreur de calendrier.
    """
    d1 = int(day_raw)
    m1 = int(month_raw)
    if 1 <= d1 <= 31 and 1 <= m1 <= 12:
        return d1, m1
    if allow_reverse and len(day_raw) == 2 and len(month_raw) == 2:
        d2 = int(day_raw[::-1])
        m2 = int(month_raw[::-1])
        for day, month in ((d2, m1), (d1, m2), (d2, m2)):
            if 1 <= day <= 31 and 1 <= month <= 12:
                return day, month
    return None, None


def _best_date_parse(text: str, allow_reverse: bool) -> Optional[tuple[str, int, int]]:
    """Extrait une date plausible de ``text`` (JJ/MM/AAAA + positions)."""
    # 1) ISO strict : aaaa-mm-jj (ex. "2003-12-04" → 04/12/2003).
    iso = _ISO_TOKEN_RE.search(text)
    if iso is not None:
        try:
            iso_day, iso_month, iso_year = (
                int(iso.group("d")),
                int(iso.group("m")),
                int(iso.group("y")),
            )
            datetime(iso_year, iso_month, iso_day)
            normalized = f"{iso_day:02d}/{iso_month:02d}/{iso_year}"
            return normalized, iso.start(), iso.end()
        except ValueError:
            pass

    # 2) jj/mm/aaaa : première paire jour/mois suivie d'une année plausible.
    for match in _DAY_SEP_MONTH_RE.finditer(text):
        tail = text[match.end() :]
        year_match = _YEAR_AFTER_RE.search(tail)
        if year_match is None:
            continue
        year = int("".join(ch for ch in year_match.group() if ch.isdigit())[:4])
        day, month = _day_month_pair(
            match.group("d1"), match.group("d2"), allow_reverse
        )
        if day is None or month is None:
            continue
        try:
            datetime(year, month, day)
        except ValueError:
            continue
        normalized = f"{day:02d}/{month:02d}/{year}"
        return normalized, match.start(), match.end() + year_match.end()

    # 3) Forme en toutes lettres (« Lundi 03 Juin 2024 à 8 H »).
    span = _french_date_span(text)
    if span is not None:
        return span[0], span[1], span[2]

    # 4) Variante OCR uniquement : année 4 chiffres + couple jour/mois juste
    #    avant (ex. « 6.1.421.2003 » → 06/01/2003). Jamais de devinette
    #    d'un mois absent : « 12/2003 » reste partiel (voir _normalize_birth).
    if allow_reverse:
        tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\d+", text)]
        for yi, (_, _, year_raw) in enumerate(tokens):
            if not re.fullmatch(r"(?:19|20)\d{2}", year_raw):
                continue
            before = [token for token in tokens[:yi] if len(token[2]) <= 2][-2:]
            if len(before) < 2:
                continue
            day, month = _day_month_pair(before[0][2], before[1][2], True)
            if day is None or month is None:
                continue
            try:
                datetime(int(year_raw), month, day)
            except ValueError:
                continue
            normalized = f"{day:02d}/{month:02d}/{year_raw}"
            return normalized, before[0][0], tokens[yi][1]
    return None


def _best_date(text: str) -> Optional[tuple[str, int, int]]:
    """Meilleure date plausible trouvée dans ``text`` (JJ/MM/AAAA + positions).

    Recherche ISO, jj/mm/aaaa et forme en toutes lettres sur le texte brut,
    puis sur les variantes OCR déconfondues (``B``→``8``, ``O``→``0``,
    ``I``→``1``, ``&``→``2``). Seules les variantes autorisent la lecture
    inversée d'un jour brouillon : ``"Bo/09|&00.3"`` → ``"08/09/2003"``
    tandis que ``"31/02/2003"`` reste impossible (erreur calendrier).
    """
    variants: list[str] = [text]
    translated = text.translate(_OCR_CONFUSION_TABLE)
    if translated != text:
        variants.append(translated)
    for variant in list(variants):
        if "&" in variant:
            replaced = variant.replace("&", "2")
            if replaced not in variants:
                variants.append(replaced)
    for index, variant in enumerate(variants):
        result = _best_date_parse(variant, allow_reverse=index > 0)
        if result is not None:
            return result
    return None


def _nearest_digit_window(text: str, length: int = 8) -> Optional[str]:
    """La séquence de ``length`` chiffres la plus proche dans ``text``.

    Fenêtre glissante sur le texte sans espaces : chaque caractère est soit
    un chiffre (gardé), soit une confusion OCR (``O``→``0``…), soit une
    erreur. La fenêtre ayant le moins d'erreurs est retenue si elle reste
    raisonnable (au plus 2 erreurs) — ex. ``"0 9 7 28 320"`` → ``"09728320"``.
    """
    stripped = "".join(text.split())
    if len(stripped) < length:
        return None
    best: Optional[tuple[int, str]] = None
    for index in range(len(stripped) - length + 1):
        window = stripped[index : index + length]
        errors = 0
        digits: list[str] = []
        for ch in window:
            if ch.isdigit():
                digits.append(ch)
            elif ch in _DIGIT_CONFUSIONS:
                digits.append(_DIGIT_CONFUSIONS[ch])
                errors += 1
            else:
                errors += 2
        if errors <= 2 and (best is None or errors < best[0]):
            best = (errors, "".join(digits))
    return best[1] if best is not None else None


def _norm_key(text: str) -> str:
    """Texte → MAJUSCULES sans accents ni ponctuation (clé de comparaison).

    Les chiffres sont conservés : une lecture OCR comme ``"D1d1"`` reste
    comparable au ``"Didi"`` du lexique.
    """
    value = unicodedata.normalize("NFKD", text.strip())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _closest_lexicon(value: str, lexicon: frozenset[str]) -> Optional[str]:
    """L'entrée du lexique la plus proche de ``value`` (distance <= 2).

    Les mots très courts (< 3 caractères) ne déclenchent pas de correction
    (sauf correspondance exacte) : ``"a"`` ne doit jamais devenir une ville.
    """
    target = _norm_key(value)
    if not target:
        return None
    best: Optional[tuple[str, int]] = None
    for candidate in lexicon:
        distance = _levenshtein(target, _norm_key(candidate))
        if best is None or distance < best[1]:
            best = (candidate, distance)
    if best is None or best[1] > 2:
        return None
    if best[1] > 0 and len(target) < 3:
        return None
    return best[0]


# --------------------------------------------------------------------------- #
# Registre des champs du formulaire (Form Rules)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldSpec:
    """Spécification d'un champ : clé, label d'affichage, alias de détection.

    ``value_re`` — regex *préférentielle* de la valeur (ex. date, CIN) : si
    plusieurs candidats se disputent le champ, celui qui matche l'emporte ;
    ``kind`` — typage de la valeur (``text``, ``cin``, ``date``, ``count``) ;
    ``section`` — bloc du formulaire (concours, candidat, codification).
    """

    key: str
    label: str
    aliases: tuple[str, ...]
    section: FormSection = FormSection.CANDIDAT
    value_re: Optional[re.Pattern[str]] = None
    kind: str = "text"


FORM_FIELDS: tuple[FieldSpec, ...] = (
    # 1 — Concours & Session
    FieldSpec(
        key="nom_concours",
        label="Nom du concours",
        aliases=("nom du concours", "nom concours"),
        section=FormSection.CONCOURS,
    ),
    FieldSpec(
        key="session",
        label="Session",
        aliases=("session",),
        section=FormSection.CONCOURS,
    ),
    FieldSpec(
        key="concours",
        label="Concours",
        aliases=("concours", "type de concours"),
        section=FormSection.CONCOURS,
    ),
    FieldSpec(
        key="epreuve",
        label="Épreuve de",
        aliases=("epreuve de", "epreuve"),
        section=FormSection.CONCOURS,
    ),
    FieldSpec(
        key="date_concours",
        label="Date",
        aliases=("date", "date du concours", "date de l epreuve"),
        section=FormSection.CONCOURS,
        value_re=DATE_RE,
        kind="date",
    ),
    FieldSpec(
        key="duree",
        label="Durée",
        aliases=("duree", "duree de l epreuve"),
        section=FormSection.CONCOURS,
        value_re=DUREE_RE,
        kind="duree",
    ),
    # 2. Informations du Candidat
    FieldSpec(
        key="nom",
        label="Nom",
        aliases=("nom", "nom complet", "nom de famille", "nom et prenom"),
    ),
    FieldSpec(
        key="prenom",
        label="Prénom",
        aliases=("prenom", "prenom et nom"),
    ),
    FieldSpec(
        key="date_naissance",
        label="Date & lieu de naissance",
        aliases=(
            "date et lieu de naissance",
            "date de naissance",
            "date et lieu",
            "date nee",
        ),
        value_re=DATE_RE,
        kind="date",
    ),
    FieldSpec(
        key="etablissement",
        label="Établissement d'origine",
        aliases=(
            "etablissement",
            "etablissement d origine",
            "etablissement scolaire",
            "institut",
            "lycee",
        ),
        kind="etablissement",
    ),
    FieldSpec(
        key="cin",
        label="N° C.I.N ou N° du passeport pour les étrangers",
        aliases=(
            "numero cin",
            "n cin",
            "cin",
            "cin ou passeport",
            "n cin ou n du passeport",
            "numero du passeport",
            "numero passeport",
            "n passeport",
            "cin ou passeport pour les etrangers",
            "n cin ou n du passeport pour les etrangers",
            "n du passeport pour les etrangers",
        ),
        value_re=CIN_RE,
        kind="cin",
    ),
    # 3. Codification & Traçabilité Administrative
    FieldSpec(
        key="serie",
        label="Série",
        aliases=("serie", "serie du sujet", "n serie"),
        section=FormSection.CODIFICATION,
        value_re=SERIE_RE,
        kind="serie",
    ),
    FieldSpec(
        key="identifiant",
        label="Identifiant",
        aliases=("identifiant", "identifiant d inscription", "numero identifiant"),
        section=FormSection.CODIFICATION,
        kind="identifiant",
    ),
    FieldSpec(
        key="nombre_cahiers",
        label="Nombre de cahiers remis",
        aliases=(
            "nombre de cahiers",
            "nombre de cahiers remis",
            "nb de cahiers",
            "n cahiers",
        ),
        section=FormSection.CODIFICATION,
        value_re=COUNT_RE,
        kind="count",
    ),
    FieldSpec(
        key="zone_signature",
        label="Zone de signature des enseignants",
        aliases=(
            "zone de signature des enseignants",
            "zone de signature",
            "signatures des enseignants",
        ),
        section=FormSection.CODIFICATION,
    ),
    FieldSpec(
        key="anonyme",
        label="Bloc code-barres / Identifiant d'anonymat",
        aliases=(
            "bloc code barres identifiant d anonymat",
            "bloc code barres identifiant",
            "bloc code barres",
            "identifiant d anonymat",
            "identifiant anonymat",
            "code barres",
            "anonymat",
        ),
        section=FormSection.CODIFICATION,
        kind="anonyme",
    ),
)

_ALIAS_INDEX: dict[str, FieldSpec] = {}
for _spec in FORM_FIELDS:
    for _alias in _spec.aliases:
        _ALIAS_INDEX[_alias] = _spec

#: Ordre canonique d'affichage des champs (l'ordre du gabarit physique).
_FORM_ORDER: dict[str, int] = {spec.key: i for i, spec in enumerate(FORM_FIELDS)}

#: Gabarit FIXE affiché par les clients : ces champs existent toujours
#: (feuille d'examen type), l'OCR ne fait que remplir leur contenu.
GABARIT_FIELDS: tuple[str, ...] = (
    "concours",  # Concours
    "epreuve",  # Épreuve de
    "date_concours",  # Date
    "duree",  # Durée
    "nom",  # Nom
    "prenom",  # Prénom
    "date_naissance",  # Date & lieu de naissance (un seul champ)
    "etablissement",  # Établissement d'origine
    "cin",  # N° C.I.N ou N° du passeport
    "serie",  # Série
    "identifiant",  # Identifiant
)

_SPEC_BY_KEY: dict[str, FieldSpec] = {spec.key: spec for spec in FORM_FIELDS}


# --------------------------------------------------------------------------- #
# Analyseur de formulaire
# --------------------------------------------------------------------------- #
class LocalFormAnalyzer:
    """Post-traitement OCR : extraction clé/valeur spatiale + validation.

    Pipeline par document :

    1. **Détection de formulaire** — chaque item est comparé au registre
       d'alias : les labels statiques (``"Nom :"``, ``"Série :"``…) sont
       repérés (avec support des items ``"label : valeur"`` en ligne) ;
    2. **Appariement spatial** — la valeur d'un champ est cherchée à droite
       sur la même ligne, puis en dessous (distance euclidienne minimale),
       avec filtrage des autres labels et préférence des candidats matchant
       le pattern du champ (ex. date, CIN) ;
    3. **Validation & scoring** — seuils de confiance (70 % / 85 %) et règles
       métier (CIN à 8 chiffres, date JJ/MM/AAAA 1950–2026, incohérence
       Série vs Identifiant, nombre de cahiers) ;
    4. **Classification du risque** — ``valid`` / ``warning`` / ``error``.

    Les seuils sont configurables à la construction (tests, métiers
    différents) mais restent inchangés par défaut.
    """

    def __init__(
        self,
        *,
        warning_threshold: float = 0.85,
        error_threshold: float = 0.65,
        min_fields_for_form: int = 2,
        min_item_confidence: float = 0.50,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.warning_threshold = warning_threshold
        self.error_threshold = error_threshold
        self.min_fields_for_form = max(1, min_fields_for_form)
        #: Les lignes OCR trop incertaines sont ignorées : elles polluent la
        #: recherche spatiale (une ligne parasite devient la "valeur" du champ)
        #: et produisent des extractions fausses.
        self.min_item_confidence = min_item_confidence
        self.logger = logger or logging.getLogger("scriptvault.form_analyzer")

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #
    def analyze_page(
        self,
        items: Sequence[dict[str, Any]],
        *,
        file_name: str = "",
        image: Optional[Any] = None,
        include_placeholders: bool = False,
    ) -> AnalyzedFormResponse:
        """Analyse complète d'une page : extraction + validation + risque.

        Args:
            items: Items OCR bruts ``[{"text", "confidence", "box"}, ...]``
                (conforme à la sortie de :class:`LocalOCREngine`).
            file_name: Nom du document source (libre).
            image: Image (numpy ``uint8``) telle qu'analysée, pour les
                contrôles visuels locaux (densité d'encre de la zone de
                signature des enseignants). ``None`` désactive ces contrôles.
            include_placeholders: Si ``True``, tous les champs du gabarit
                sont retournés — les champs non lus par l'OCR portent le
                statut ``empty`` (affichage permanent du formulaire).

        Returns:
            Le formulaire structuré : champs, statuts, confiance globale et
            temps de post-traitement (budget < 30 ms).
        """
        started = time.perf_counter()
        fields = self.analyze_extracted_items(list(items))
        is_form = len(fields) >= self.min_fields_for_form

        # Contrôle visuel : zone de signature des enseignants (si image fournie).
        if image is not None:
            ink = self._signature_ink_ratio(list(items), image)
            if ink is not None and ink < SIGNATURE_INK_MIN:
                fields = self._mark_signature_missing(fields)

        # Mode gabarit permanent : complète les champs non détectés.
        if include_placeholders:
            fields = self._fill_placeholders(fields)

        detected = [field for field in fields if field.confidence > 0.0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return AnalyzedFormResponse(
            file_name=file_name,
            is_form=is_form,
            global_confidence=round(
                sum(field.confidence for field in detected) / len(detected), 4
            )
            if detected
            else 0.0,
            processing_time_ms=round(elapsed_ms, 2),
            fields=fields,
        )

    def analyze_extracted_items(
        self, raw_items: list[dict[str, Any]]
    ) -> list[FormFieldResult]:
        """Extrait puis valide les champs d'une page (voir :meth:`analyze_page`)."""
        parsed_data = self._extract_key_values(raw_items)
        by_key = {field["key"]: field for field in parsed_data}
        fields = [self._validate_field(field, by_key) for field in parsed_data]
        # Ordre canonique du gabarit (Concours → Candidat → Codification).
        fields.sort(key=lambda field: _FORM_ORDER.get(field.key, 999))
        # RÈGLE 4b : incohérence Série vs Identifiant -> les DEUX champs en rouge.
        return self.flag_identifier_mismatch(fields)

    # ------------------------------------------------------------------ #
    # Étape 1 & 2 : détection du formulaire et appariement spatial
    # ------------------------------------------------------------------ #
    def _extract_key_values(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transforme les items OCR en paires ``{key, label, value, conf, box}``.

        Heuristique spatiale : la valeur d'un label est l'item situé
        immédiatement **à droite** sur la même ligne, sinon l'item **en
        dessous** le plus proche (distance euclidienne des centres).
        """
        if not items:
            return []
        parsed: list[dict[str, Any]] = []
        consumed: set[int] = set()

        # Pré-traitement : normalisation + rectangles englobants.
        entries: list[dict[str, Any]] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            confidence = float(item.get("confidence", 0.0))
            box = item.get("box") or []
            # Rejet du bruit : une ligne trop incertaine ne doit ni devenir
            # un label ni être choisie comme valeur d'un champ.
            if box and confidence < self.min_item_confidence:
                continue
            entries.append(
                {
                    "raw": text,
                    "text": _clean_value(text),
                    "norm": _normalize_keyword(text),
                    "confidence": confidence,
                    "box": box,
                    "rect": _box_rect(box),
                }
            )

        # 1) Labels "inline" : item unique "Nom : Didi".
        for index, entry in enumerate(entries):
            if index in consumed:
                continue
            spec, value = self._split_inline_label(entry["raw"])
            if spec is None or not value:
                continue
            consumed.add(index)
            parsed.append(
                self._build_field(
                    spec,
                    value,
                    entry["confidence"],
                    entry["box"],
                )
            )

        # 2) Labels isolés : recherche de la valeur dans le voisinage spatial.
        for index, entry in enumerate(entries):
            if index in consumed:
                continue
            spec = self._match_label(entry["norm"])
            if spec is None:
                continue
            consumed.add(index)
            value_index = self._find_value_index(entries, index, consumed)
            if value_index is None:
                continue
            consumed.add(value_index)
            value_entry = entries[value_index]
            parsed.append(
                self._build_field(
                    spec,
                    value_entry["raw"],
                    value_entry["confidence"],
                    value_entry["box"],
                )
            )

        # 3) Relecture ciblée des numéros manquants (CIN/série/identifiant).
        parsed = self._harvest_numeric_fields(entries, parsed, consumed)

        # 4) Relecture ciblée des noms illisibles (prénom/nom) sur la page.
        parsed = self._harvest_name_fields(entries, parsed, consumed)

        return parsed

    # ------------------------------------------------------------------ #
    # Relecture ciblée : CIN / Série / Identifiant récupérés sur la page
    # ------------------------------------------------------------------ #
    def _harvest_numeric_fields(
        self,
        entries: list[dict[str, Any]],
        parsed: list[dict[str, Any]],
        consumed: set[int],
    ) -> list[dict[str, Any]]:
        """Recherche les numéros manquants sur toute la page.

        Quand un champ numérique n'a pas pu être lu dans la ligne de son
        label (ex. un CIN écrit dans sa cellule de saisie), on balaie la
        page à la recherche d'une suite de chiffres de la bonne longueur
        (CIN 8, Série 4, Identifiant 6) en privilégiant les items proches du
        label. Les confusions OCR (``O``→``0``, ``B``→``8``…) s'appliquent.
        """
        by_key = {field["key"]: field for field in parsed}
        for key, length, slack in (
            ("cin", 8, 1),
            ("serie", 4, 0),
            ("identifiant", 6, 1),
        ):
            field = by_key.get(key)
            if field is not None:
                digits_count = len("".join(ch for ch in field["value"] if ch.isdigit()))
                if key == "serie":
                    if digits_count in (3, 4):
                        continue
                elif not self._needs_harvest(field["value"], length):
                    continue
            spec = _SPEC_BY_KEY[key]
            label_index = self._find_label_index(entries, spec)
            if label_index is None:
                continue
            consumed.add(label_index)
            label_rect = entries[label_index]["rect"]
            best: Optional[tuple[float, int, str, dict[str, Any]]] = None
            for index, entry in enumerate(entries):
                if index in consumed:
                    continue
                if self._match_label(entry["norm"]) is not None:
                    continue
                if _is_noise_value(entry["text"]):
                    continue
                deconf = entry["text"].translate(_OCR_CONFUSION_TABLE)
                deconf = "".join(
                    ch if ch.isdigit() or ch.isspace() else " " for ch in deconf
                )
                windows = self._digit_windows(deconf, length, slack)
                for window, penalty in windows:
                    score = penalty * 1000.0 + (
                        self._rect_distance(label_rect, entry["rect"]) / 2000.0
                    )
                    if best is None or score < best[0]:
                        best = (score, index, window, entry)
            if best is None:
                continue
            _score, index, run, entry = best
            digits = "".join(
                _DIGIT_CONFUSIONS.get(ch, ch)
                for ch in entry["text"]
                if ch.isdigit() or ch in _DIGIT_CONFUSIONS
            )
            if len(digits) <= length + 2:
                consumed.add(index)  # un seul numéro dans l'item
            parsed.append(
                self._build_field(spec, run, entry["confidence"], entry["box"])
            )
            by_key[key] = parsed[-1]
        return parsed

    @staticmethod
    def _needs_harvest(value: str, expected: int) -> bool:
        """Vrai si le champ numérique est vide ou mal lu (taille != attendue)."""
        if not value:
            return True
        digits = "".join(ch for ch in value if ch.isdigit())
        return len(digits) != expected

    @staticmethod
    def _find_label_index(
        entries: list[dict[str, Any]], spec: FieldSpec
    ) -> Optional[int]:
        """Index du premier item correspondant au label du champ ``spec``."""
        for index, entry in enumerate(entries):
            if entry["norm"] and LocalFormAnalyzer._match_label(entry["norm"]) is spec:
                return index
        return None

    @staticmethod
    def _digit_windows(text: str, length: int, slack: int = 1) -> list[tuple[str, int]]:
        """Fenêtres de ``length`` chiffres (valeur, pénalité d'alignement).

        ``text`` conserve les espaces OCR entre les chiffres (« 5400 17 » =
        identifiant « 540017 ») : on regroupe les suites contiguës séparées
        par de simples espaces et on énumère les fenêtres d'exactement
        ``length`` chiffres. Une fenêtre commençant ET finissant sur une
        borne de suite obtient une pénalité nulle.
        """
        runs = [(m.start(), m.end()) for m in re.finditer(r"\d+", text)]
        groups: list[list[tuple[int, int]]] = []
        for s, e in runs:
            if groups:
                prev_s, prev_e = groups[-1][-1]
                gap = text[prev_e:s]
                if gap and not gap.strip(" \t./-|,"):
                    groups[-1].append((s, e))
                    continue
            groups.append([(s, e)])
        windows: list[tuple[str, int]] = []
        for group in groups:
            start0, end0 = group[0][0], group[-1][1]
            digits = "".join(ch for ch in text[start0:end0] if ch.isdigit())
            n = len(digits)
            bounds: list[tuple[int, int]] = []
            offset = 0
            for ts, te in group:
                run_n = len("".join(ch for ch in text[ts:te] if ch.isdigit()))
                bounds.append((offset, offset + run_n))
                offset += run_n
            starts = {s for s, _e in bounds}
            ends = {e for _s, e in bounds}
            for a in range(0, n - length + 1):
                penalty = (0 if a in starts else 1) + (0 if a + length in ends else 1)
                if penalty <= 1:
                    windows.append((digits[a : a + length], penalty))
            for s, e in bounds:
                run = digits[s:e]
                if abs(len(run) - length) <= slack:
                    windows.append((run, 0))
        return windows

    @staticmethod
    def _rect_distance(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        """Distance de Manhattan entre les centres de deux rectangles."""
        return abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) + abs(
            (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
        )

    def _harvest_name_fields(
        self,
        entries: list[dict[str, Any]],
        parsed: list[dict[str, Any]],
        consumed: set[int],
    ) -> list[dict[str, Any]]:
        """Recherche un Nom/Prénom illisible sur toute la page.

        Quand la valeur OCR d'un nom est un glyphe parasite (ex. ``"二"``)
        ou absente, on balaie la page à la recherche du mot le plus proche
        du lexique tunisien (prénom ou nom), en privilégiant les items
        proches du label du champ. L'entrée retenue est consommée.
        """
        by_key = {field["key"]: field for field in parsed}
        for key, lexicons in (
            ("prenom", (TUNISIAN_FIRST_NAMES, TUNISIAN_LAST_NAMES)),
            ("nom", (TUNISIAN_LAST_NAMES, TUNISIAN_FIRST_NAMES)),
        ):
            field = by_key.get(key)
            if field is None or field["value"]:
                continue
            spec = _SPEC_BY_KEY[key]
            label_index = self._find_label_index(entries, spec)
            if label_index is None:
                continue
            consumed.add(label_index)
            label_rect = entries[label_index]["rect"]
            best: Optional[tuple[float, int, str, dict[str, Any]]] = None
            for index, entry in enumerate(entries):
                if index in consumed:
                    continue
                if self._match_label(entry["norm"]) is not None:
                    continue
                if _is_noise_value(entry["text"]):
                    continue
                match: Optional[str] = None
                for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", entry["text"]):
                    for lexicon in lexicons:
                        match = _closest_lexicon(word, lexicon)
                        if match is not None:
                            break
                    if match is not None:
                        break
                if match is None:
                    continue
                distance = _levenshtein(_norm_key(word), _norm_key(match))
                score = distance * 1000.0 + (
                    self._rect_distance(label_rect, entry["rect"]) / 2000.0
                )
                if best is None or score < best[0]:
                    best = (score, index, match, entry)
            if best is None:
                continue
            _, index, match, entry = best
            consumed.add(index)
            parsed.append(
                self._build_field(
                    spec, match, max(entry["confidence"], 0.95), entry["box"]
                )
            )
            by_key[key] = parsed[-1]
        return parsed

    def _build_field(
        self,
        spec: FieldSpec,
        value: str,
        confidence: float,
        box: Any,
    ) -> dict[str, Any]:
        """Construit l'entrée clé/valeur brute avant validation.

        La valeur brute OCR est **corrigée** au préalable : confusions
        chiffres/lettres pour les champs numériques, date de naissance
        structurée (meilleure date plausible + lieu rapproché des villes
        tunisiennes), noms/prénoms rapprochés du lexique tunisien.
        """
        if spec.key == "date_naissance":
            corrected = self._normalize_birth_value(value)
        elif spec.key in ("nom", "prenom"):
            corrected = self._correct_name_value(value)
        else:
            corrected = self._correct_ocr_value(spec.kind, value)
        if corrected != value:
            # Correction fiable par lexique/déchiffrement : la lecture brute
            # (faible confiance) est remplacée par une valeur canonique sûre.
            confidence = max(confidence, 0.92)
        return {
            "key": spec.key,
            "label": spec.label,
            "value": corrected,
            "confidence": confidence,
            "box": box,
            "section": spec.section,
        }

    def _normalize_birth_value(self, value: str) -> str:
        """Date & lieu de naissance : meilleure date + ville tunisienne.

        La date la plus adéquate (JJ/MM/AAAA) est extraite du texte OCR
        bruité ; le reste du texte est interprété comme le lieu et rapproché
        de la ville tunisienne la plus proche (ex. ``"05.12.2003 Tnus"`` →
        ``"05/12/2003 · Tunis"``).
        """
        best = _best_date(value)
        date_str: Optional[str] = None
        start = end = 0
        if best is not None:
            date_str, start, end = best
        else:
            longhand = _french_date_value(value)
            if longhand is not None:
                date_str = longhand
                span = _french_date_span(value)
                if span is not None:
                    start, end = span[1], span[2]
        if date_str is None:
            if not _has_full_date_triplet(value):
                year_hint = _year_hint(value)
                if year_hint is not None:
                    # Jamais de devinette du jour/mois : la valeur reste
                    # lisible mais passe en « partielle » (warning). Une date
                    # structurée mais impossible (« 31/02/2003 ») reste rouge.
                    return f"{year_hint} (jour/mois illisibles)"
            return value
        raw_lieu = (value[:start] + value[end:]).strip()
        lieu = re.sub(r"\s+", " ", re.sub(r"^[\s,.;:·-]+|[\s,.;:·-]+$", "", raw_lieu))
        lieu = _clean_value(lieu).strip("·- ,.;:")
        city = None
        if lieu:
            city = _closest_lexicon(lieu, TUNISIAN_CITIES)
            if city is None:
                for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", lieu):
                    city = _closest_lexicon(word, TUNISIAN_CITIES)
                    if city is not None:
                        break
        if city is not None:
            lieu = city
        if not lieu or len(lieu) <= 1:
            return date_str
        return f"{date_str} · {lieu}"

    @staticmethod
    def _correct_name_value(value: str) -> str:
        """Rapproche un nom/prénom de la liste tunisienne la plus proche.

        Correction uniquement si l'écart (distance de Levenshtein) est <= 2 :
        ``"D1d1"`` → ``"Didi"``. Une valeur inconnue reste inchangée (elle
        sera signalée par la validation, jamais inventée).
        """
        if not value:
            return value
        if any(ch.isdigit() for ch in value):
            # Valeur avec chiffres : la comparaison porte sur le mot entier
            # (« D1d1 » → « Didi »), jamais mot par mot (« Q1z9x » intact).
            match = _closest_lexicon(value, TUNISIAN_FIRST_NAMES)
            if match is None:
                match = _closest_lexicon(value, TUNISIAN_LAST_NAMES)
            return match if match is not None else value
        corrected: list[str] = []
        for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", value):
            match = _closest_lexicon(word, TUNISIAN_FIRST_NAMES)
            if match is None:
                match = _closest_lexicon(word, TUNISIAN_LAST_NAMES)
            corrected.append(match if match is not None else word)
        return " ".join(corrected).strip().strip("'")

    @staticmethod
    def _correct_ocr_value(kind: str, value: str) -> str:
        """Corrige les lectures OCR erronées d'une valeur numérique.

        PaddleOCR confond souvent les lettres et les chiffres sur les copies
        scannées : ``O`` lu ``0``, ``I/l`` lu ``1``, ``S`` lu ``5``, ``B`` lu
        ``8``. La correction n'est appliquée qu'aux champs numériques
        (CIN, série, identifiant, anonymat…) **et seulement si la valeur est
        majoritairement composée de chiffres** : un passeport alphanumérique
        (ex. ``"OMAR097"``) doit conserver ses lettres intactes plutôt que de
        les défigurer en ``"0M491097"``. Le CIN (8 chiffres) est reconstruit à
        partir de la séquence de chiffres la plus proche (espaces, ``O``→``0``,
        ``I``→``1``…). Les dates en toutes lettres (« Lundi 03 Juin 2024 »)
        sont normalisées en JJ/MM/AAAA. Un établissement est rapproché de la
        liste officielle des prépas tunisiennes (ex. ``"TLEIB"`` → ``"IPEIB"``).
        """
        if kind == "etablissement":
            match = _closest_lexicon(value, _ETABLISSEMENTS_SET)
            return match if match is not None else value
        if kind == "date":
            date_span = _best_date(value)
            if date_span is not None:
                return date_span[0]
            longhand = _french_date_value(value)
            return longhand if longhand is not None else value
        if kind == "cin":
            nearest = _nearest_digit_window(value)
            if nearest is not None:
                return nearest
        if kind not in _VALUE_KINDS or not value:
            return value
        digits = sum(1 for char in value if char.isdigit())
        if digits == 0 or digits * 2 < len(value):
            return value
        corrected = value.translate(_OCR_CONFUSION_TABLE)
        if kind == "cin":
            nearest = _nearest_digit_window(corrected)
            if nearest is not None:
                return nearest
        return corrected

    @staticmethod
    def _match_label(norm_text: str) -> Optional[FieldSpec]:
        """Retourne le champ dont un alias matche un texte normalisé, sinon None."""
        if norm_text in _ALIAS_INDEX:
            return _ALIAS_INDEX[norm_text]
        # Repli : alias suivi d'un mot (ex. "cin ou passeport" tronqué).
        # Un alias court (ex. "cin") ne déclenche jamais de préfixe pour ne
        # pas confondre une valeur (ex. "cinq") avec un label.
        for alias, spec in _ALIAS_INDEX.items():
            if len(alias) >= 4 and norm_text.startswith(alias + " "):
                return spec
        return None

    @staticmethod
    def _split_inline_label(
        text: str,
    ) -> tuple[Optional[FieldSpec], str]:
        """Extrait ``(champ, valeur)`` d'un item ``"Label : Valeur"``."""
        if ":" not in text:
            return None, ""
        label_part, _, value_part = text.partition(":")
        spec = LocalFormAnalyzer._match_label(_normalize_keyword(label_part))
        if spec is None:
            return None, ""
        return spec, _clean_value(value_part)

    def _find_value_index(
        self,
        entries: list[dict[str, Any]],
        label_index: int,
        consumed: set[int],
    ) -> Optional[int]:
        """Cherche la valeur d'un label : à droite, puis en dessous."""
        label = entries[label_index]
        rect = label["rect"]
        x0, y0, x1, y1 = rect
        center_y = (y0 + y1) / 2.0
        label_height = max(1.0, y1 - y0)

        # --- Passe 1 : à droite sur la même ligne ----------------------- #
        same_line: list[tuple[float, int]] = []
        for index, entry in enumerate(entries):
            if index == label_index or index in consumed:
                continue
            if self._match_label(entry["norm"]) is not None:
                continue  # un autre label n'est jamais une valeur
            if _is_noise_value(entry["text"]):
                continue  # fragment de label (ex. « N0m8re de ») : jamais une valeur
            ex0, ey0, ex1, ey1 = entry["rect"]
            if ex0 < x1 - 2.0:
                continue
            vertical_gap = abs((ey0 + ey1) / 2.0 - center_y)
            if vertical_gap > max(label_height, 8.0) * 0.5:
                continue
            gap = ex0 - x1
            if gap > 400.0:
                continue
            same_line.append((gap, index))
        if same_line:
            same_line.sort(key=lambda pair: (pair[0], entries[pair[1]]["text"]))
            return self._pick_value_index(entries, same_line, label_index)

        # --- Passe 2 : en dessous (ligne suivante / cellule) ----------- #
        below: list[tuple[float, int]] = []
        for index, entry in enumerate(entries):
            if index == label_index or index in consumed:
                continue
            if self._match_label(entry["norm"]) is not None:
                continue
            if _is_noise_value(entry["text"]):
                continue
            ex0, ey0, ex1, ey1 = entry["rect"]
            if ey0 <= y1:
                continue
            if ex1 < x0 - label_height * 2.0 or ex0 > x1 + label_height * 2.0:
                continue  # trop éloigné horizontalement du label
            dy = ey0 - y1
            if dy > 300.0:
                continue
            below.append((dy, index))
        if below:
            below.sort(key=lambda pair: (pair[0], entries[pair[1]]["text"]))
            return self._pick_value_index(entries, below, label_index)
        return None

    def _pick_value_index(
        self,
        entries: list[dict[str, Any]],
        candidates: list[tuple[float, int]],
        label_index: int,
    ) -> Optional[int]:
        """Choisit le meilleur candidat-valeur.

        Privilégie le candidat qui matche le pattern du champ (ex. une date
        pour ``date_naissance``) ; à égalité, le premier classé spatialement.
        """
        spec = self._match_label(entries[label_index]["norm"])
        for _gap, index in candidates:
            value = entries[index]["text"]
            if spec is not None and spec.value_re is not None:
                if spec.value_re.match(value):
                    return index
        return candidates[0][1]

    # ------------------------------------------------------------------ #
    # Étape 3 & 4 : validation métier et classification du risque
    # ------------------------------------------------------------------ #
    def _validate_field(
        self,
        field: dict[str, Any],
        all_fields: dict[str, dict[str, Any]],
    ) -> FormFieldResult:
        """Valide un champ : seuils OCR puis règles métier (regex/cohérence)."""
        key = field["key"]
        value = field["value"]
        confidence = field["confidence"]

        status = FieldStatus.VALID
        err_msg: Optional[str] = None

        # RÈGLE 1 : Seuil de confiance OCR (douteux / critique).
        if confidence < self.error_threshold:
            status = FieldStatus.ERROR
            err_msg = (
                f"Confiance OCR critique "
                f"({int(confidence * 100)}% < {int(self.error_threshold * 100)}%)"
            )
        elif confidence < self.warning_threshold:
            status = FieldStatus.WARNING

        # RÈGLE 2 : CIN / Passeport — 8 chiffres exactement (Tunisie).
        if key == "cin":
            ok, message = self._validate_cin(value)
            if not ok:
                status, err_msg = FieldStatus.ERROR, message

        # RÈGLE 3 : Dates — format JJ/MM/AAAA + cohérence année (naissance
        # ou date du concours). Une année seule (OCR bruité) reste un
        # avertissement « partiel », jamais une invention.
        elif key in ("date_naissance", "date_concours"):
            ok, message = self._validate_date(
                value, all_fields if key == "date_naissance" else None
            )
            if not ok:
                year_hint = _year_hint(value)
                if (
                    year_hint is not None
                    and not _has_full_date_triplet(value)
                    and status == FieldStatus.VALID
                ):
                    status, err_msg = (
                        FieldStatus.WARNING,
                        "Date partielle : "
                        f"année {year_hint} lue (jour/mois illisibles).",
                    )
                else:
                    status, err_msg = FieldStatus.ERROR, message

        # RÈGLE 4 : Cohérence Série vs Identifiant (préfixe).
        elif key == "identifiant":
            ok, message = self._validate_identifier(value, all_fields)
            if not ok:
                status, err_msg = FieldStatus.ERROR, message
            else:
                digits = "".join(ch for ch in value if ch.isdigit())
                if re.search(r"[A-Za-z]", value) or not re.fullmatch(
                    r"\d{5,8}", digits
                ):
                    status, err_msg = (
                        FieldStatus.ERROR,
                        "Identifiant invalide (6 chiffres attendus, ex. 540117).",
                    )
                elif len(digits) != 6 and status == FieldStatus.VALID:
                    status, err_msg = (
                        FieldStatus.WARNING,
                        (f"Identifiant sur 6 chiffres attendu (lu: {value})."),
                    )

        # RÈGLE 5 : Nombre de cahiers — entier de 1 à 5.
        elif key == "nombre_cahiers":
            if not COUNT_RE.match(value):
                status, err_msg = (
                    FieldStatus.ERROR,
                    "Nombre de cahiers invalide (attendu: entier de 1 à 5).",
                )

        # RÈGLE 6 : Série — 3 ou 4 chiffres (ex. "531", "5401").
        elif key == "serie":
            digits = "".join(ch for ch in value if ch.isdigit())
            if re.search(r"[A-Za-z]", value) or not re.fullmatch(r"\d{3,4}", digits):
                status, err_msg = (
                    FieldStatus.ERROR,
                    "Série invalide (3 ou 4 chiffres attendus, ex. 5401).",
                )

        # RÈGLE 7 : Durée — "2", "2h", "2h30", "4 Heures" (texte toléré).
        elif key == "duree":
            if DUREE_RE.match(value):
                pass
            elif re.search(r"\d", value):
                pass  # durée mixte (ex. "4 Heures") : numérique présent
            elif len(value) >= 2:
                status, err_msg = (
                    FieldStatus.WARNING,
                    "Durée en toutes lettres (non standardisée).",
                )
            else:
                status, err_msg = (
                    FieldStatus.ERROR,
                    "Durée illisible.",
                )

        # RÈGLE 8 : Anonymat — réplication exacte du couple Série/Identifiant.
        elif key == "anonyme":
            ok, message = self._validate_anonyme(value, all_fields)
            if not ok:
                status, err_msg = FieldStatus.ERROR, message

        # RÈGLE 9 : Nom / Prénom — alphabétique uniquement (accents, tirets).
        elif key in ("nom", "prenom"):
            if not NAME_RE.match(value):
                status, err_msg = (
                    FieldStatus.ERROR,
                    "Présence de chiffres ou de symboles parasites "
                    "(ex. D1d1 au lieu de Didi).",
                )

        # RÈGLE 10 : Établissement d'origine — liste officielle (Levenshtein).
        elif key == "etablissement":
            ok, message = self._validate_etablissement(value)
            if not ok:
                status = FieldStatus.WARNING if status == FieldStatus.VALID else status
                err_msg = message

        # RÈGLE 11 : Texte (session, concours, épreuve…) — non vide.
        else:
            if len(value) < 2:
                status, err_msg = (
                    FieldStatus.ERROR,
                    "Valeur trop courte pour être un texte exploitable.",
                )

        return FormFieldResult(
            key=key,
            label=field["label"],
            value=value,
            confidence=confidence,
            status=status,
            error_message=err_msg,
            bounding_box=field.get("box") or None,
            section=field["section"],
            section_label=SECTION_LABELS[field["section"]],
        )

    # ------------------------------------------------------------------ #
    # Validateurs métier
    # ------------------------------------------------------------------ #
    def _validate_cin(self, value: str) -> tuple[bool, str]:
        """CIN/passeport : numérique et exactement 8 chiffres."""
        cleaned = value.replace(" ", "")
        if CIN_RE.match(cleaned):
            return True, ""
        if any(ch.isalpha() for ch in cleaned):
            return (
                False,
                "Numéro CIN invalide : des lettres ont été lues "
                "(ex. O972832O) — 8 chiffres attendus.",
            )
        return False, "Format CIN invalide (8 chiffres attendus)."

    def _validate_date(
        self,
        value: str,
        all_fields: Optional[dict[str, dict[str, Any]]] = None,
    ) -> tuple[bool, str]:
        """Date : JJ/MM/AAAA préféré, texte (jour mois année) accepté aussi.

        Une date en toutes lettres (« Lundi 03 Juin 2024 à 8 H ») est une
        valeur légitime : elle n'est jamais rejetée tant qu'une année (19xx
        ou 20xx) y figure. Une date impossible (ex. 31/02) reste une erreur.
        """
        best = _best_date(value)
        if best is None:
            # Des nombres jour/mois/année existent mais sont impossibles
            # (ex. 31/02) : vraie erreur, pas une date au format libre.
            if _DATE_TOKEN_RE.search(value) or _ISO_TOKEN_RE.search(value):
                return (
                    False,
                    "Date de naissance invalide (jour/mois inexistant).",
                )
            if _YEAR_LIKE_RE.search(value):
                return True, ""  # date au format libre acceptée
            return (
                False,
                "Date non reconnue (aucun jour/mois/année visible).",
            )
        year = int(best[0][-4:])
        month = int(best[0][3:5])
        day = int(best[0][0:2])
        low, high = DATE_YEAR_RANGE
        if not (low <= year <= high):
            return (
                False,
                f"Année de naissance incohérente ({year} hors plage {low}–{high}).",
            )
        try:
            datetime(year, month, day)
        except ValueError:
            return (
                False,
                "Date de naissance invalide (jour/mois inexistant).",
            )
        # Cohérence chronologique avec la session (naissance seulement) :
        # pour la session 2026, une naissance doit précéder 2010.
        if all_fields is not None:
            session_year = self._session_year(all_fields)
            if session_year is not None and year > session_year - 17:
                return (
                    False,
                    f"Année de naissance incohérente avec la session "
                    f"{session_year} (attendu avant {session_year - 16}).",
                )
        return True, ""

    @staticmethod
    def _session_year(
        all_fields: dict[str, dict[str, Any]],
    ) -> Optional[int]:
        """Année de session extraite de ``session`` ou ``date_concours``."""
        for key in ("session", "date_concours"):
            match = re.search(
                r"(?:19|20)\d{2}", all_fields.get(key, {}).get("value", "")
            )
            if match:
                return int(match.group())
        return None

    @staticmethod
    def _validate_etablissement(value: str) -> tuple[bool, str]:
        """Établissement d'origine : liste officielle + distance de Levenshtein."""
        norm = re.sub(r"[^A-Z]", "", value.upper())
        if not norm:
            return True, ""
        if norm in _ETABLISSEMENTS:
            return True, ""
        best = min(_levenshtein(norm, known) for known in _ETABLISSEMENTS)
        if len(norm) >= 3 and best <= 2:
            return True, ""
        return (
            False,
            "Établissement suspect : ne correspond à aucun établissement "
            "officiel (IPEIN, IPEIT, IPEIM, FGES…).",
        )

    def _validate_anonyme(
        self,
        value: str,
        all_fields: dict[str, dict[str, Any]],
    ) -> tuple[bool, str]:
        """Anonymat : suite numérique + réplication du couple Série/Identifiant."""
        if not re.match(r"^[0-9][0-9 -]{3,19}$", value.replace(" ", "")):
            return (
                False,
                "Identifiant d'anonymat invalide (suite numérique attendue).",
            )
        serie = all_fields.get("serie", {}).get("value", "").strip()
        identifiant = all_fields.get("identifiant", {}).get("value", "").strip()
        if serie and identifiant and SERIE_RE.match(serie) and identifiant.isdigit():
            if "".join(ch for ch in value if ch.isdigit()) != f"{serie}{identifiant}":
                return (
                    False,
                    f"Ne correspond pas au couple Série/Identifiant "
                    f"(attendu: {serie} {identifiant}).",
                )
        return True, ""

    def _validate_identifier(
        self,
        value: str,
        all_fields: dict[str, dict[str, Any]],
    ) -> tuple[bool, str]:
        """Identifiant : doit commencer par la Série (ex. 514 -> 514017).

        La comparaison ignore les espaces insérés par l'OCR (ex. Série
        ``"514"`` + Identifiant ``"514 017"`` restent cohérents).
        """
        serie = all_fields.get("serie", {}).get("value", "").strip()
        if not serie:
            return True, ""  # série absente : règle non applicable
        digits = "".join(ch for ch in value if ch.isdigit())
        serie_digits = "".join(ch for ch in serie if ch.isdigit())
        if serie_digits and digits.startswith(serie_digits):
            return True, ""
        return (
            False,
            f"L'identifiant doit commencer par la série ({serie}).",
        )

    # ------------------------------------------------------------------ #
    # Annotation croisée : la série et l'identifiant s'affichent en rouge
    # ------------------------------------------------------------------ #
    def flag_identifier_mismatch(
        self, fields: list[FormFieldResult]
    ) -> list[FormFieldResult]:
        """Colore en ``error`` la Série ET l'Identifiant en cas d'incohérence.

        Post-traitement appelé par les clients après :meth:`analyze_page` :
        si l'identifiant ne commence pas par la série, les deux champs sont
        marqués en rouge (alerte visuelle conjointe, exigence métier).
        """
        by_key = {field.key: field for field in fields}
        identifier = by_key.get("identifiant")
        serie = by_key.get("serie")
        if (
            identifier is None
            or serie is None
            or identifier.status != FieldStatus.ERROR
            or "doit commencer par la série" not in (identifier.error_message or "")
        ):
            return list(fields)
        for field in fields:
            if field.key in ("serie", "identifiant"):
                field.status = FieldStatus.ERROR
                if field.key == "serie":
                    field.error_message = (
                        f"Incohérence : l'identifiant ne commence pas par "
                        f"la série ({serie.value})."
                    )
        return list(fields)

    # ------------------------------------------------------------------ #
    # Gabarit permanent & contrôles visuels locaux
    # ------------------------------------------------------------------ #
    def _fill_placeholders(
        self, fields: list[FormFieldResult]
    ) -> list[FormFieldResult]:
        """Construit le gabarit FIXE (statut ``empty`` pour les champs non lus).

        Les champs listés dans :data:`GABARIT_FIELDS` existent toujours dans
        la réponse — l'extraction OCR ne fait que remplir leur contenu. Tout
        champ hors gabarit est retiré, sauf l'alerte rouge « Signature
        manquante » qui reste visible en dernière position.
        """
        by_key = {field.key: field for field in fields}
        kept: list[FormFieldResult] = []
        for key in GABARIT_FIELDS:
            field = by_key.get(key)
            if field is not None:
                kept.append(field)
                continue
            spec = _SPEC_BY_KEY[key]
            kept.append(
                FormFieldResult(
                    key=spec.key,
                    label=spec.label,
                    value="",
                    confidence=0.0,
                    status=FieldStatus.EMPTY,
                    error_message=None,
                    bounding_box=None,
                    section=spec.section,
                    section_label=SECTION_LABELS[spec.section],
                )
            )
        signature = by_key.get("zone_signature")
        if signature is not None and signature.status == FieldStatus.ERROR:
            kept.append(signature)
        return kept

    @staticmethod
    def _signature_ink_ratio(
        items: Sequence[dict[str, Any]],
        image: Any,
    ) -> Optional[float]:
        """Fraction de pixels sombres sous l'étiquette « Zone de signature ».

        La zone de signature des enseignants est la région située sous
        l'étiquette : un fond quasi blanc (fraction < :data:`SIGNATURE_INK_MIN`)
        déclenche l'alerte « Signature enseignante manquante ».
        """
        try:
            if image is None or getattr(image, "ndim", 0) not in (2, 3):
                return None

            if image.ndim == 3:
                gray = image[:, :, 0]  # BGR : canal bleu ≈ luminance
            else:
                gray = image
            height, width = gray.shape[:2]
            for item in items:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                spec = LocalFormAnalyzer._match_label(_normalize_keyword(text))
                if spec is None or spec.key != "zone_signature":
                    continue
                x0, y0, x1, y1 = _box_rect(item.get("box") or [])
                if x1 - x0 < 4.0 or y1 - y0 < 4.0:
                    continue
                # Région en dessous du label signature.
                rx0 = max(0, int(x0 - 60))
                rx1 = min(width, int(x1 + 120))
                ry0 = int(y1)
                ry1 = min(height, int(y1 + max(90.0, 2.5 * (y1 - y0))))
                if ry1 - ry0 < 10 or rx1 - rx0 < 10:
                    continue
                region = gray[ry0:ry1, rx0:rx1]
                if region.size == 0:
                    continue
                return float((region < 128).mean())
        except Exception:  # pragma: no cover - contrôle purement informatif
            return None
        return None

    def _mark_signature_missing(
        self, fields: list[FormFieldResult]
    ) -> list[FormFieldResult]:
        """Passe le champ « zone de signature » en erreur rouge."""
        message = "Signature enseignante manquante (zone vide)."
        for field in fields:
            if field.key == "zone_signature":
                field.status = FieldStatus.ERROR
                field.error_message = message
                return fields
        fields.append(
            FormFieldResult(
                key="zone_signature",
                label="Zone de signature des enseignants",
                value="",
                confidence=0.0,
                status=FieldStatus.ERROR,
                error_message=message,
                bounding_box=None,
                section=FormSection.CODIFICATION,
                section_label=SECTION_LABELS[FormSection.CODIFICATION],
            )
        )
        return fields


# --------------------------------------------------------------------------- #
# Raccourci module : analyse complète en une ligne
# --------------------------------------------------------------------------- #
_ANALYZER: Optional[LocalFormAnalyzer] = None


def analyze_form_items(
    items: Sequence[dict[str, Any]],
    *,
    file_name: str = "",
    image: Optional[Any] = None,
    include_placeholders: bool = False,
) -> AnalyzedFormResponse:
    """Analyse une page OCR complète via une instance partagée.

    L'instance ``LocalFormAnalyzer`` est sans état : elle peut être réutilisée
    en toute sécurité par tous les threads/clients.

    Args:
        items: Items OCR bruts ``[{"text", "confidence", "box"}, ...]``.
        file_name: Nom du document source (libre).
        image: Image (numpy ``uint8``) analysée, pour les contrôles visuels
            locaux (zone de signature des enseignants).
        include_placeholders: Si ``True``, retourne tous les champs du gabarit
            (champs non lus en statut ``empty``).

    Raises:
        ValueError: Si ``items`` est invalide (pas une séquence de dicts).
    """
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = LocalFormAnalyzer()
    return _ANALYZER.analyze_page(
        items,
        file_name=file_name,
        image=image,
        include_placeholders=include_placeholders,
    )


# --------------------------------------------------------------------------- #
# CLI de démonstration (aucune dépendance Paddle requise)
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """Ligne de commande : ``python form_analyzer.py [--json <items.json>]``.

    Sans argument, exécute un scénario de démonstration représentant une
    feuille d'examen type (nom, prénom, date, CIN, série, identifiant).
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyseur de formulaire OCR (post-traitement local)."
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Fichier JSON contenant les items OCR (text/confidence/box).",
    )
    args = parser.parse_args(argv)

    if args.json:
        with open(args.json, encoding="utf-8") as handle:
            payload = json.load(handle)
        items = payload if isinstance(payload, list) else payload.get("items", [])
    else:
        items = demo_items()

    response = analyze_form_items(items, file_name=args.json or "demo.png")
    for field in response.fields:
        flag = {
            "valid": "OK",
            "warning": "!!",
            "error": "!!!",
            "empty": "--",
        }.get(field.status.value)
        print(f"[{flag}] {field.label:<28} {field.value:<16} {field.status.value}")
        if field.error_message:
            print(f"      -> {field.error_message}")
    print(
        f"\n{len(response.fields)} champ(s), is_form={response.is_form}, "
        f"confiance={response.global_confidence:.2f}, "
        f"post-traitement={response.processing_time_ms:.2f} ms"
    )
    return 0


def demo_items() -> list[dict[str, Any]]:
    """Items OCR simulant une feuille d'examen scannée (données de test)."""
    base_y = 60
    rows = [
        "Nom du concours :",
        "Session :",
        "Concours :",
        "Épreuve de :",
        "Date :",
        "Durée :",
        "Nom :",
        "Prénom :",
        "Date & lieu de naissance :",
        "Établissement d'origine :",
        "N° CIN :",
        "Série :",
        "Identifiant :",
        "Nombre de cahiers remis :",
        "Bloc code-barres / Identifiant d'anonymat :",
    ]
    values = [
        "Baccalauréat 2026",
        "Principale",
        "Sciences expérimentales",
        "Mathématiques",
        "04/06/2026",
        "2h",
        "Didi",
        "Mayssa",
        "04/12/2003",
        "IPEIN",
        "09728320",
        "5140",
        "514017",
        "2",
        "5140 514017",
    ]
    items: list[dict[str, Any]] = []
    for row, (label, value) in enumerate(zip(rows, values)):
        items.append(
            {
                "text": label,
                "confidence": 0.97,
                "box": [
                    [50, base_y + row * 60],
                    [300, base_y + row * 60],
                    [300, base_y + row * 60 + 34],
                    [50, base_y + row * 60 + 34],
                ],
            }
        )
        items.append(
            {
                "text": value,
                "confidence": 0.96,
                "box": [
                    [320, base_y + row * 60],
                    [560, base_y + row * 60],
                    [560, base_y + row * 60 + 34],
                    [320, base_y + row * 60 + 34],
                ],
            }
        )
    return items


if __name__ == "__main__":
    raise SystemExit(main())
