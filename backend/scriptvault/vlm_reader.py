"""Lecture directe des champs manuscrits par un modèle Vision-Langage (VLM) local.

Remplace l'OCR classique (TrOCR / PP-OCR) sur les zones manuscrites du
formulaire par une lecture **directe** via un VLM local (Qwen2-VL /
Qwen2.5-VL) desservi par un serveur local Ollama / llama.cpp. Le module
expose :

* :class:`LocalVLMReader` — client asynchrone haut performance : encodage
  base64 JPEG du crop, appel unique au serveur local, timeout strict
  (2.0 s par défaut), sortie JSON structurée ``{"text", "confidence"}``.
* :func:`read_handwritten_crop` — fonction asynchrone autonome qui encode
  le crop et renvoie le dict de lecture :class:`VLMResult`.
* :meth:`LocalVLMReader.read_handwritten_crops_batch` — lecture **par lot**
  des 3 champs manuscrits (Nom, Prénom, Établissement) via
  ``asyncio.gather`` (requêtes parallèles, timeout individuel).
* **Repli d'urgence** : si le VLM échoue ou dépasse le délai, le moteur
  bascule automatiquement sur le reconnaisseur de secours (TrOCR / PP-OCR)
  fourni par l'appelant (``fallback``).

Les contraintes contextuelles sont injectées dans le *prompt system* en
fonction de ``field_type`` : liste des acronymes officiels d'établissements
tunisiens (:data:`etab_classes.ETAB_CLASSES`) pour ``"etablissement"``,
dictionnaire de noms/prénoms tunisiens pour ``"nom"`` / ``"prenom"``. La
réponse attendue du VLM est un **JSON strict** sans aucun texte explicatif.

Configuration (variables ``SCRIPTVAULT_*``) :

* ``SCRIPTVAULT_VLM_URL`` — base URL du serveur (défaut
  ``http://127.0.0.1:11434``, Ollama).
* ``SCRIPTVAULT_VLM_MODEL`` — modèle VLM local (défaut ``qwen2.5vl:2b``).
* ``SCRIPTVAULT_VLM_TIMEOUT_S`` — délai maximal d'un appel (défaut 2.0).
* ``SCRIPTVAULT_VLM_JSON`` — impose ``format: "json"`` (défaut 1).
* ``SCRIPTVAULT_VLM_MAX_TOKENS`` / ``SCRIPTVAULT_VLM_TEMPERATURE`` /
  ``SCRIPTVAULT_VLM_MAX_SIDE`` — génération et taille de l'image.

Exemple::

    from scriptvault.vlm_reader import LocalVLMReader

    reader = LocalVLMReader(fallback=lambda crop: ("", 0.0))
    result = await reader.read_handwritten_crop(crop_nom, "nom")
    print(result["text"], result["confidence"], result["engine"])
    reader.close()

Dépendances : ``httpx`` (client asynchrone), ``opencv-contrib-python``,
``numpy``. Aucun poids de modèle n'est embarqué : le VLM tourne sur le
serveur local Ollama / llama.cpp.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, TypedDict, Union

import cv2
import numpy as np

from .etab_classes import ETAB_CLASSES, closest_etab

__version__ = "1.0.0"

__all__ = [
    "LocalVLMReader",
    "VLMConfig",
    "VLMResult",
    "FieldType",
    "FallbackRecognizer",
    "encode_image_base64",
    "build_system_prompt",
    "build_user_prompt",
    "parse_vlm_json",
    "sanitize_vlm_text",
    "read_handwritten_crop",
    "VLMBaseError",
    "VLMInitError",
    "VLMTimeoutError",
    "VLMRequestError",
    "VLMResultError",
]

PathLike = Union[str, os.PathLike[str]]
FieldType = Literal["nom", "prenom", "etablissement"]

#: Un reconnaisseur de secours (TrOCR / PP-OCR) : ``(crop) -> (texte, conf)``.
FallbackRecognizer = Callable[[np.ndarray], tuple[str, float]]

logger: logging.Logger
logger = logging.getLogger("scriptvault.vlm_reader")
logger.addHandler(logging.NullHandler())

_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "qwen2.5vl:7b"
_DEFAULT_TIMEOUT_S = 2.0  # Timeout strict d'un appel VLM (spéc. <= 2 s)
_DEFAULT_MAX_TOKENS = 32
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_IMAGE_SIDE = 1024
_JPEG_QUALITY = 92

#: Grille de bandes (lecture du formulaire complet en UN appel VLM).
#: À chaud sur GPU, le 7B lit une grille en ~13 s ; le délai couvre surtout
#: le chargement à froid du modèle dans Ollama (2-7 min selon la VRAM).
_GRID_TIMEOUT_S = 240.0
_GRID_MAX_TOKENS = 1024
_GRID_MAX_IMAGE_SIDE = 1280

#: Temps de garde du pont synchrone au-delà du timeout VLM (marge du réseau).
_SYNC_BRIDGE_EXTRA_S = 0.5

_FIELD_LABELS: dict[str, str] = {
    "nom": "Nom de famille",
    "prenom": "Prénom",
    "etablissement": "Établissement d'origine",
}

#: Extrait le premier objet JSON d'une réponse (fences et bruit tolérés).
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCES_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Exceptions métier
# --------------------------------------------------------------------------- #
class VLMBaseError(Exception):
    """Erreur racine du module de lecture VLM."""


class VLMInitError(VLMBaseError):
    """Dépendance (httpx) absente ou client non initialisable."""


class VLMTimeoutError(VLMBaseError):
    """Délai maximal d'un appel VLM dépassé."""


class VLMRequestError(VLMBaseError):
    """Réponse HTTP du serveur VLM inexploitable (statut, transport, JSON)."""


class VLMResultError(VLMBaseError):
    """Réponse du VLM non conforme au JSON structuré attendu."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class VLMConfig:
    """Configuration du lecteur VLM (dérivée de l'environnement)."""

    base_url: str = _DEFAULT_BASE_URL
    model: str = _DEFAULT_MODEL
    timeout_s: float = _DEFAULT_TIMEOUT_S
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE
    json_mode: bool = True
    max_image_side: int = _DEFAULT_MAX_IMAGE_SIDE
    grid_timeout_s: float = _GRID_TIMEOUT_S
    grid_max_tokens: int = _GRID_MAX_TOKENS
    grid_max_image_side: int = _GRID_MAX_IMAGE_SIDE
    #: Fenêtre de contexte Ollama (``options.num_ctx``). Le manifest officiel
    #: de qwen2.5vl annonce 128k : Ollama alloue alors un KV cache qui déborde
    #: des 6-8 Go VRAM d'un laptop et déporte des couches vers le CPU (lenteur
    #: extrême). 8k suffisent largement à une grille (~2 k tokens) et gardent
    #: le modèle entièrement sur GPU. ``None`` = valeur par défaut du modèle.
    num_ctx: Optional[int] = 8192
    #: Couches GPU explicites (``options.num_gpu``). 99 = toutes les couches :
    #: sans lui, Ollama ne déporte que ~2 Go sur 6 Go de VRAM (réglage WDDM
    #: conservateur) et 70 % de l'inférence tourne sur CPU (~99 s/grille au
    #: lieu de ~13 s). ``None`` = auto Ollama.
    num_gpu: Optional[int] = 99
    #: Durée de résidence du modèle dans Ollama (``keep_alive``) : évite le
    #: rechargement à froid (~2 min) entre deux fichiers du même lot.
    keep_alive: str = "30m"

    @classmethod
    def from_env(cls) -> "VLMConfig":
        """Construit la configuration depuis les variables ``SCRIPTVAULT_VLM_*``."""
        raw_url = os.environ.get("SCRIPTVAULT_VLM_URL", "").strip()
        return cls(
            base_url=(raw_url or _DEFAULT_BASE_URL).rstrip("/"),
            model=os.environ.get("SCRIPTVAULT_VLM_MODEL", _DEFAULT_MODEL).strip()
            or _DEFAULT_MODEL,
            timeout_s=max(0.1, _env_float("SCRIPTVAULT_VLM_TIMEOUT_S", _DEFAULT_TIMEOUT_S)),
            max_tokens=max(1, _env_int("SCRIPTVAULT_VLM_MAX_TOKENS", _DEFAULT_MAX_TOKENS)),
            temperature=max(0.0, min(2.0, _env_float("SCRIPTVAULT_VLM_TEMPERATURE", 0.0))),
            json_mode=_env_bool("SCRIPTVAULT_VLM_JSON", True),
            max_image_side=max(
                256, _env_int("SCRIPTVAULT_VLM_MAX_SIDE", _DEFAULT_MAX_IMAGE_SIDE)
            ),
            grid_timeout_s=max(
                5.0, _env_float("SCRIPTVAULT_VLM_GRID_TIMEOUT_S", _GRID_TIMEOUT_S)
            ),
            grid_max_tokens=max(
                256, _env_int("SCRIPTVAULT_VLM_GRID_MAX_TOKENS", _GRID_MAX_TOKENS)
            ),
            grid_max_image_side=max(
                256, _env_int("SCRIPTVAULT_VLM_GRID_MAX_SIDE", _GRID_MAX_IMAGE_SIDE)
            ),
            num_ctx=max(2048, _env_int("SCRIPTVAULT_VLM_NUM_CTX", 8192)),
            num_gpu=_env_int("SCRIPTVAULT_VLM_NUM_GPU", 99) or None,
            keep_alive=_env("SCRIPTVAULT_VLM_KEEP_ALIVE", "30m"),
        )


# --------------------------------------------------------------------------- #
# Résultat structuré
# --------------------------------------------------------------------------- #
class VLMResult(TypedDict):
    """Lecture d'un champ manuscrit (VLM ou repli de secours).

    ``source`` distingue la lecture VLM directe (``"vlm"``) du repli
    d'urgence (``"fallback"``) ; ``engine`` nomme le moteur réellement
    utilisé (ex. ``"vlm:qwen2.5vl:2b"`` ou ``"htr"``).
    """

    text: str
    confidence: float
    field_type: str
    source: Literal["vlm", "fallback"]
    engine: str
    latency_ms: float


# --------------------------------------------------------------------------- #
# Lexique de contraintes contextuelles
# --------------------------------------------------------------------------- #
#: Dictionnaire non exhaustif de noms / prénoms tunisiens : contrainte
#: contextuelle injectée dans le prompt ``nom`` / ``prenom`` (guide du VLM,
#: jamais un correcteur post-lecture — une valeur inconnue reste valide).
TUNISIAN_NAMES_LEXICON: tuple[str, ...] = (
    "MOHAMED",
    "AHMED",
    "ALI",
    "HASSEN",
    "YOUSSEF",
    "OMAR",
    "AMINE",
    "KHALIL",
    "MAHER",
    "ANIS",
    "MALEK",
    "SALMA",
    "NADIA",
    "RIM",
    "INES",
    "AMIRA",
    "MARIEM",
    "SIHEM",
    "HOUDA",
    "FATMA",
    "KHADIJA",
    "SARRA",
    "NIHEL",
    "EMNA",
    "TRABELSI",
    "GHARBI",
    "ELLOUMI",
    "JAZIRI",
    "BOUAZIZI",
    "HAMDI",
    "KACEM",
    "MABROUK",
    "SAIDI",
    "JEBALI",
    "BEN SALEM",
    "BEN ALI",
    "BEN AMOR",
    "DALI",
    "KRIDENE",
    "KSIBI",
    "ZAGHDOUD",
    "EL FAKHRY",
)


# --------------------------------------------------------------------------- #
# Encodage du crop (base64)
# --------------------------------------------------------------------------- #
def encode_image_base64(
    crop: np.ndarray, *, max_side: int = _DEFAULT_MAX_IMAGE_SIDE, quality: int = _JPEG_QUALITY
) -> str:
    """Encode un crop BGR/gris en JPEG base64 (prêt pour le payload Ollama).

    La zone est réduite (ratio conservé) à ``max_side`` px au plus grand côté
    avant encodage : les crops de pages haute résolution (plusieurs Mo) sont
    compressés en quelques dizaines de Ko — gain de latence majeur sans perte
    de lisibilité pour un VLM.
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        raise ValueError("Crop d'image invalide (None ou vide).")
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    height, width = crop.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        crop = cv2.resize(
            crop,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))]
    )
    if not ok:
        raise ValueError("Encodage JPEG du crop impossible.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# Prompts contextuels (Stratégie de lecture directe)
# --------------------------------------------------------------------------- #
def _field_label(field_type: str) -> str:
    return _FIELD_LABELS.get(field_type, field_type.capitalize())


def _constraint_section(field_type: str) -> str:
    """Contraintes contextuelles du champ, injectées dans le prompt system."""
    if field_type == "etablissement":
        acronyms = ", ".join(ETAB_CLASSES)
        return (
            "Le champ est le sigle de l'établissement d'origine (document "
            "tunisien). Rends la valeur en MAJUSCULES, sans espaces.\n"
            f"Acronymes officiels possibles (liste non exhaustive) : {acronyms}.\n"
            "Si l'écriture est illisible, choisis l'acronyme le plus proche de "
            "cette liste — n'invente JAMAIS un acronyme hors liste."
        )
    if field_type in ("nom", "prenom"):
        lexicon = ", ".join(TUNISIAN_NAMES_LEXICON)
        return (
            "Le champ est un nom de famille / prénom tunisien (document en "
            "latin). La valeur doit ressembler à un nom ou prénom tunisien "
            "plausible.\n"
            f"Lexique de référence (liste non exhaustive) : {lexicon}.\n"
            "Lis précisément l'écriture : ne corrige pas une valeur déjà "
            "lisible (un nom hors lexique reste un nom valide)."
        )
    return "Lis le texte manuscrit tel quel, sans interpréter."


def build_system_prompt(field_type: str) -> str:
    """Prompt system : rôle du VLM + contraintes contextuelles du champ.

    Le VLM n'a jamais le document entier : uniquement le crop ciblé et le
    type de champ, pour une lecture directe focalisée (zéro bruit visuel).
    """
    return (
        "Tu es un expert en lecture de formulaires d'examen tunisiens "
        "(écriture manuscrite en français/latin).\n"
        "Contrainte de sortie stricte : réponds UNIQUEMENT au format JSON "
        '{\"text\": \"valeur_lue\", \"confidence\": 0.92} — la confiance dans '
        "[0, 1], aucune explication, aucun texte avant ou après le JSON.\n"
        + _constraint_section(field_type)
    )


def build_user_prompt(field_type: str) -> str:
    """Prompt utilisateur : désigne le champ et rappelle la sortie JSON."""
    return (
        f"Voici l'image de la zone manuscrite du champ « {_field_label(field_type)} ».\n"
        "Lis l'écriture et réponds au format JSON strict "
        '{"text": "valeur_lue", "confidence": 0.0-1.0} sans rien d\'autre.'
    )


# --------------------------------------------------------------------------- #
# Grille de bandes : lecture du formulaire complet en UN appel VLM
# --------------------------------------------------------------------------- #
def build_band_grid_system_prompt() -> str:
    """Prompt system : lecture d'une grille de lignes de formulaire.

    L'image contient plusieurs lignes numérotées (``1..N`` de haut en bas) ;
    chaque ligne = un libellé imprimé (ex. ``Nom :``) suivi d'une valeur
    (souvent manuscrite). Le VLM retourne un JSON unique avec toutes les
    lignes — c'est la lecture « comme un humain/Gemini » du formulaire.
    """
    return (
        "Tu es un expert en lecture de formulaires d'examen tunisiens.\n"
        "L'image est une grille de lignes de formulaire, numérotées de 1 à N "
        "de haut en bas, dans la marge gauche.\n"
        "Chaque ligne contient un libellé imprimé (ex. « Nom : », « Prénom : ») "
        "suivi d'une valeur souvent manuscrite.\n"
        "Lis CHAQUE ligne numérotée avec précision : le libellé ET la valeur, "
        "exactement tels qu'ils sont écrits (accents, majuscules, ponctuation).\n"
        "Ne traduis pas, ne corrige pas, n'invente rien. Si une ligne est vide "
        "ou illisible, écris une chaîne vide pour son « text ».\n"
        "Contrainte de sortie stricte : réponds UNIQUEMENT au format JSON "
        '{"rows": [{"row": 1, "text": "...", "confidence": 0.9}, ...]} — la '
        "confiance dans [0, 1], aucune explication, aucun texte avant/après."
    )


def build_band_grid_user_prompt(first_row: int, last_row: int) -> str:
    """Prompt utilisateur : désigne la grille et rappelle le JSON attendu."""
    count = last_row - first_row + 1
    return (
        f"Voici une grille de {count} lignes de formulaire numérotées de "
        f"{first_row} à {last_row} (de haut en bas). Lis chaque ligne et "
        "réponds au JSON strict "
        '{"rows": [{"row": N, "text": "libellé : valeur", "confidence": 0.0-1.0}]} '
        "sans rien d'autre."
    )


def parse_band_grid_json(
    raw: str, first_row: int, last_row: int
) -> list[tuple[int, str, float]]:
    """Extrait ``{"rows": [{"row", "text", "confidence"}]}`` de la réponse VLM.

    Tolère fences markdown, bruit autour du bloc JSON, lignes manquantes ou
    hors bornes (elles sont simplement ignorées). Lève :class:`VLMResultError`
    si aucun bloc JSON ou aucune ligne exploitable n'est trouvée.
    """
    text = (raw or "").strip()
    text = _FENCES_RE.sub("", text).strip()
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        raise VLMResultError(f"Réponse VLM grille sans bloc JSON: {raw!r:.80}")
    try:
        payload = json.loads(match.group())
    except (json.JSONDecodeError, TypeError) as exc:
        raise VLMResultError(f"JSON VLM grille invalide: {exc}") from exc
    if not isinstance(payload, dict):
        raise VLMResultError(f"Réponse VLM grille non objet: {type(payload).__name__}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise VLMResultError("Absence de liste 'rows' dans la réponse VLM grille.")
    out: list[tuple[int, str, float]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        raw_row = entry.get("row")
        try:
            index = int(raw_row) if raw_row is not None else 0
        except (TypeError, ValueError):
            continue
        if not first_row <= index <= last_row:
            continue
        value = str(entry.get("text", "")).strip()
        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not 0.0 <= confidence <= 1.0:
            confidence = 0.0
        out.append((index, value, round(confidence, 4)))
    if not out:
        raise VLMResultError("Aucune ligne exploitable dans la réponse VLM grille.")
    return out


# --------------------------------------------------------------------------- #
# Parsing strict de la sortie JSON du VLM
# --------------------------------------------------------------------------- #
def parse_vlm_json(raw: str) -> tuple[str, float]:
    """Extrait ``{"text", "confidence"}`` de la réponse brute du VLM.

    Tolère les fences markdown, les espaces parasites et le bruit autour du
    bloc JSON (les petits modèles répondent parfois `````json ...`````).
    Lève :class:`VLMResultError` si le JSON est absent, non objet, sans
    ``text`` non vide, ou avec une confiance hors ``[0, 1]``.
    """
    text = (raw or "").strip()
    text = _FENCES_RE.sub("", text).strip()
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        raise VLMResultError(f"Réponse VLM sans bloc JSON: {raw!r:.80}")
    try:
        payload = json.loads(match.group())
    except (json.JSONDecodeError, TypeError) as exc:
        raise VLMResultError(f"JSON VLM invalide: {exc}") from exc
    if not isinstance(payload, dict):
        raise VLMResultError(f"Réponse VLM non objet JSON: {type(payload).__name__}")
    value = str(payload.get("text", "")).strip()
    if not value:
        raise VLMResultError("Champ 'text' vide dans la réponse VLM.")
    confidence = float(payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise VLMResultError(f"Confiance VLM hors bornes [0, 1]: {confidence!r}")
    return value, round(confidence, 4)


def _fold_accent(value: str) -> str:
    """Minuscules sans accents (comparaison uniquement, jamais de sortie)."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def sanitize_vlm_text(text: str, field_type: str) -> str:
    """Normalise la valeur lue selon la contrainte du champ (jamais inventée).

    * ``etablissement`` — majuscules, seul l'alphanumérique est conservé,
      puis rapprochement flou vers les acronymes officiels
      (:func:`etab_classes.closest_etab`) : un sigle hors liste reste lisible
      tel quel.
    * ``nom`` / ``prenom`` — seules lettres/espaces/tirets sont conservés,
      chaque mot reçoit une capitale initiale.
    """
    if field_type == "etablissement":
        probe = "".join(ch for ch in str(text).upper() if ch.isalnum())
        canonical = closest_etab(probe)
        return canonical if canonical is not None else probe
    if field_type in ("nom", "prenom"):
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’]+", str(text))
        out = [word[0].upper() + word[1:] for word in words if word]
        return " ".join(out) if out else str(text).strip()
    return str(text).strip()


# --------------------------------------------------------------------------- #
# Client HTTP commun (Ollama / llama.cpp)
# --------------------------------------------------------------------------- #
def _default_client_factory(config: Optional[VLMConfig] = None) -> Any:
    """Fabrique du client `httpx` asynchrone (timeout réseau > timeout strict)."""
    import httpx

    timeout_s = (config.timeout_s if config is not None else VLMConfig().timeout_s)
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s + _SYNC_BRIDGE_EXTRA_S),
        limits=httpx.Limits(max_connections=8),
    )


# --------------------------------------------------------------------------- #
# Lecteur VLM local
# --------------------------------------------------------------------------- #
class LocalVLMReader:
    """Lecture directe de crops manuscrits par un VLM local (Ollama/llama.cpp).

    Caractéristiques :

    * Encodage base64 JPEG du crop (``encode_image_base64``) puis envoi
      unique au serveur local — prompt system contextuel par ``field_type``
      (acronymes d'établissements, lexique de noms tunisiens).
    * **Timeout strict** (``VLMConfig.timeout_s``, 2.0 s par défaut) appliqué
      par :func:`asyncio.wait_for` à chaque appel.
    * **Repli d'urgence** : si le VLM échoue (transport, HTTP, JSON invalide)
      ou dépasse le délai, la prédiction du reconnaisseur ``fallback``
      (TrOCR / PP-OCR) remplace la lecture — la chaîne OCR ne casse jamais.
    * Lecture par lot des 3 champs manuscrits via ``asyncio.gather``
      (``read_handwritten_crops_batch``) : chaque crop indépendant, timeout
      individuel, repli par champ.
    * **Lecture grille** (``read_form_band_grid``) : toutes les lignes du
      formulaire sont lues en UN appel (image numérotée), comme le ferait
      Gemini — c'est le chemin privilégié quand le moteur OCR local ne lit
      pas correctement les bandes (TrOCR hallucine).
    * Pont synchrone ``sync_read_handwritten_crop`` (boucle de fond dédiée)
      pour l'intégration dans le pipeline OCR synchrone de ``core_ocr``.

    L'objet est thread-safe (client ``httpx`` + boucle de fond dédiée au
    pont synchrone) et doit être fermé par :meth:`close`.
    """

    def __init__(
        self,
        config: Optional[VLMConfig] = None,
        *,
        fallback: Optional[FallbackRecognizer] = None,
        client_factory: Optional[Callable[[], Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise le lecteur.

        Args:
            config: Configuration (``None`` → ``SCRIPTVAULT_VLM_*``).
            fallback: Reconnaisseur de secours ``(crop) -> (texte, confiance)``
                (TrOCR ou PP-OCR). ``None`` → repli vide (``text: ""``).
            client_factory: Fabrique du client HTTP asynchrone (injectable
                pour les tests). ``None`` → client ``httpx`` par défaut.
            logger: Logger optionnel.
        """
        self.config = config or VLMConfig.from_env()
        self.fallback = fallback
        self.logger = logger or logging.getLogger("scriptvault.vlm_reader.reader")
        self._client: Any = None
        self._client_factory: Callable[[], Any] = (
            client_factory if client_factory is not None else _default_client_factory
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._closed = False
        self.logger.info(
            "VLM local configuré: %s (%s, timeout %.1f s).",
            self.config.base_url,
            self.config.model,
            self.config.timeout_s,
        )

    # ------------------------------------------------------------------ #
    # État
    # ------------------------------------------------------------------ #
    @property
    def is_enabled(self) -> bool:
        """Le lecteur n'est pas fermé."""
        return not self._closed

    def _get_client(self) -> Any:
        if self._closed:
            raise VLMInitError("Le lecteur VLM est fermé.")
        if self._client is None:
            try:
                self._client = self._client_factory()
            except Exception as exc:
                raise VLMInitError(
                    f"Client HTTP VLM non initialisable: {type(exc).__name__}: {exc}"
                ) from exc
        return self._client

    # ------------------------------------------------------------------ #
    # API publique asynchrone
    # ------------------------------------------------------------------ #
    async def read_handwritten_crop(
        self, crop: np.ndarray, field_type: str
    ) -> VLMResult:
        """Lit UN crop manuscrit : VLM direct, repli d'urgence sinon.

        Args:
            crop: Zone manuscrite découpée (BGR ou gris).
            field_type: ``"nom"``, ``"prenom"`` ou ``"etablissement"``.

        Returns:
            Un :class:`VLMResult` — ``source`` vaut ``"vlm"`` si la lecture
            directe a abouti, ``"fallback"`` si le repli de secours a pris le
            relais (échec ou timeout).
        """
        field = str(field_type).strip().lower()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._read_vlm(crop, field), timeout=self.config.timeout_s
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "VLM %s en timeout (%.1f s) ; repli %s.",
                self.config.model,
                self.config.timeout_s,
                self._fallback_name(),
            )
            return self._fallback_result(crop, field, started)
        except VLMBaseError as exc:
            self.logger.warning("VLM %s en échec (%s) ; repli.", self.config.model, exc)
            return self._fallback_result(crop, field, started)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return result

    async def read_handwritten_crops_batch(
        self,
        crops: list[np.ndarray],
        field_types: list[str],
    ) -> list[VLMResult]:
        """Lit plusieurs crops manuscrits en parallèle (cas des 3 champs).

        Les requêtes sont exécutées concurremment avec :func:`asyncio.gather`
        — un timeout individuel par champ, aucun échec global : chaque crop
        échoué bascule isolément sur le repli. L'ordre des résultats suit
        l'ordre des entrées.

        Raises:
            ValueError: ``crops`` et ``field_types`` de longueurs différentes.
        """
        if len(crops) != len(field_types):
            raise ValueError(
                "crops (%d) et field_types (%d) de tailles différentes."
                % (len(crops), len(field_types))
            )
        results = await asyncio.gather(
            *(
                self.read_handwritten_crop(crop, field)
                for crop, field in zip(crops, field_types)
            )
        )
        return list(results)

    async def read_form_band_grid(
        self,
        grid: np.ndarray,
        first_row: int,
        last_row: int,
    ) -> Optional[list[tuple[int, str, float]]]:
        """Lit TOUTES les lignes d'une grille de formulaire en un appel VLM.

        Args:
            grid: Grille de lignes numérotées (construite par
                ``image_processing._build_band_grid``) : ligne ``first_row``
                en haut.
            first_row: Numéro absolu de la première ligne de la grille.
            last_row: Numéro absolu de la dernière ligne de la grille.

        Returns:
            ``[(index, texte, confiance), ...]`` pour les lignes lues, ou
            ``None`` si l'appel VLM échoue (timeout, HTTP, JSON inexploitable)
            — le pipeline bascule alors sur son chemin de secours (TrOCR).
        """
        started = time.perf_counter()
        try:
            rows = await asyncio.wait_for(
                self._read_band_grid(grid, first_row, last_row),
                timeout=self.config.grid_timeout_s,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Grille VLM %s en timeout (%.1f s) ; repli TrOCR.",
                self.config.model,
                self.config.grid_timeout_s,
            )
            return None
        except VLMBaseError as exc:
            self.logger.warning("Grille VLM %s en échec (%s) ; repli TrOCR.", self.config.model, exc)
            return None
        self.logger.info(
            "Grille VLM lue: %d lignes (rangées %d..%d) en %.1f s.",
            len(rows),
            first_row,
            last_row,
            time.perf_counter() - started,
        )
        return rows

    async def _read_band_grid(
        self, grid: np.ndarray, first_row: int, last_row: int
    ) -> list[tuple[int, str, float]]:
        image_b64 = encode_image_base64(grid, max_side=self.config.grid_max_image_side)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_band_grid_system_prompt()},
            {
                "role": "user",
                "content": build_band_grid_user_prompt(first_row, last_row),
                "images": [image_b64],
            },
        ]
        content = await self._chat(
            messages,
            max_tokens=self.config.grid_max_tokens,
            timeout=self.config.grid_timeout_s,
        )
        return parse_band_grid_json(content, first_row, last_row)

    def sync_read_form_band_grid(
        self, grid: np.ndarray, first_row: int, last_row: int
    ) -> Optional[list[tuple[int, str, float]]]:
        """Version synchrone de :meth:`read_form_band_grid` (pipeline OCR).

        Retourne ``None`` si le lecteur est fermé ou si la boucle de fond ne
        répond pas — le pipeline garde son chemin TrOCR historique.
        """
        if self._closed:
            return None
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.read_form_band_grid(grid, first_row, last_row), loop
        )
        timeout = self.config.grid_timeout_s + _SYNC_BRIDGE_EXTRA_S
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            self.logger.warning("Pont synchrone grille VLM en timeout.")
            return None

    async def is_available(self) -> bool:
        """Vérifie que le serveur VLM local répond (``GET /api/tags``)."""
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.get(f"{self.config.base_url}/api/tags"), timeout=1.5
            )
            return bool(response) and getattr(response, "status_code", 200) == 200
        except Exception as exc:
            self.logger.debug("VLM indisponible (%s).", exc)
            return False

    # ------------------------------------------------------------------ #
    # Pré-chargement du modèle (démarrage serveur)
    # ------------------------------------------------------------------ #
    async def _warm_up(self) -> bool:
        """Appel trivial : force Ollama à charger le modèle en mémoire.

        Sans lui, le **premier** formulaire d'un lot subit le chargement à
        froid (2-3 min selon la VRAM) ; en le lançant au démarrage du
        serveur, chaque page est lue à pleine vitesse dès le départ.
        """
        try:
            content = await asyncio.wait_for(
                self._chat(
                    [{"role": "user", "content": "ok"}],
                    max_tokens=1,
                    timeout=_GRID_TIMEOUT_S,
                ),
                timeout=_GRID_TIMEOUT_S,
            )
            return bool(content)
        except Exception as exc:
            self.logger.debug("Pré-chauffage VLM ignoré (%s).", exc)
            return False

    def warm_up(self) -> bool:
        """Version synchrone de :meth:`_warm_up` (thread de démarrage).

        Non bloquante pour l'appelant en cas d'échec : le pipeline conserve
        son repli TrOCR si Ollama est absent.
        """
        if self._closed:
            return False
        try:
            loop = self._ensure_loop()
            future = asyncio.run_coroutine_threadsafe(self._warm_up(), loop)
            return bool(future.result(timeout=_GRID_TIMEOUT_S + 5.0))
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Implémentation VLM
    # ------------------------------------------------------------------ #
    async def _read_vlm(self, crop: np.ndarray, field_type: str) -> VLMResult:
        image_b64 = encode_image_base64(crop, max_side=self.config.max_image_side)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(field_type)},
            {
                "role": "user",
                "content": build_user_prompt(field_type),
                "images": [image_b64],
            },
        ]
        content = await self._chat(messages)
        text, confidence = parse_vlm_json(content)
        return {
            "text": sanitize_vlm_text(text, field_type),
            "confidence": confidence,
            "field_type": field_type,
            "source": "vlm",
            "engine": f"vlm:{self.config.model}",
            "latency_ms": 0.0,
        }

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Appelle ``POST /api/chat`` (protocole Ollama).

        ``max_tokens``/``timeout`` permettent de déroger aux valeurs par
        défaut (ex. grille de bandes, beaucoup plus verbeuse qu'un champ).
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }
        if self.config.num_ctx is not None:
            payload["options"]["num_ctx"] = self.config.num_ctx
        if self.config.num_gpu is not None:
            payload["options"]["num_gpu"] = self.config.num_gpu
        if self.config.keep_alive:
            payload["keep_alive"] = self.config.keep_alive
        if self.config.json_mode:
            payload["format"] = "json"
        client = self._get_client()
        effective_timeout = timeout or self.config.timeout_s
        started = time.perf_counter()
        try:
            import httpx

            response = await asyncio.wait_for(
                client.post(
                    f"{self.config.base_url}/api/chat",
                    json=payload,
                    timeout=httpx.Timeout(effective_timeout + _SYNC_BRIDGE_EXTRA_S),
                ),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise VLMTimeoutError("Appel VLM en timeout réseau.") from exc
        except VLMBaseError:
            raise
        except Exception as exc:
            raise VLMRequestError(
                f"Transport VLM en échec: {type(exc).__name__}: {exc}"
            ) from exc
        if getattr(response, "status_code", None) != 200:
            raise VLMRequestError(
                f"HTTP {getattr(response, 'status_code', '?')} du serveur VLM."
            )
        try:
            data = response.json()
        except Exception as exc:
            raise VLMResultError(f"Corps de réponse VLM non JSON: {exc}") from exc
        message = data.get("message") or {}
        content = str(message.get("content", "")).strip()
        if not content:
            raise VLMResultError("Réponse VLM sans contenu de message.")
        self.logger.debug(
            "VLM ok en %.1f ms (%d caractères).",
            (time.perf_counter() - started) * 1000.0,
            len(content),
        )
        return content

    # ------------------------------------------------------------------ #
    # Repli d'urgence
    # ------------------------------------------------------------------ #
    def _fallback_name(self) -> str:
        fn = self.fallback
        if fn is None:
            return "aucun"
        return str(getattr(fn, "name", None) or getattr(fn, "__name__", "fallback"))

    def _fallback_result(self, crop: np.ndarray, field_type: str, started: float) -> VLMResult:
        engine = self._fallback_name()
        text, confidence = "", 0.0
        if self.fallback is not None:
            try:
                text, confidence = self.fallback(crop)
            except Exception as exc:
                self.logger.warning("Repli %r en échec (%s).", engine, exc)
                text, confidence = "", 0.0
        return {
            "text": str(text or "").strip(),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "field_type": field_type,
            "source": "fallback",
            "engine": engine,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }

    # ------------------------------------------------------------------ #
    # Pont synchrone (intégration pipeline OCR / core_ocr)
    # ------------------------------------------------------------------ #
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Boucle asyncio dédiée (thread de fond) pour les appels synchrones."""
        if self._loop is None or self._loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="scriptvault-vlm-loop",
                daemon=True,
            )
            thread.start()
            self._loop = loop
            self._loop_thread = thread
        return self._loop

    def sync_read_handwritten_crop(
        self, crop: np.ndarray, field_type: str
    ) -> VLMResult:
        """Version synchrone de :meth:`read_handwritten_crop`.

        Exécute la coroutine sur la boucle de fond dédiée (sûr depuis n'importe
        quel thread, y compris un worker ``ProcessPool`` ou le pipeline OCR
        synchrone). En dernier recours (blocage du thread de fond), le repli
        est exécuté directement dans le thread appelant.
        """
        if self._closed:
            raise VLMInitError("Le lecteur VLM est fermé.")
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.read_handwritten_crop(crop, field_type), loop
        )
        timeout = self.config.timeout_s + _SYNC_BRIDGE_EXTRA_S
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            started = time.perf_counter()
            return self._fallback_result(crop, str(field_type).lower(), started)

    # ------------------------------------------------------------------ #
    # Cycle de vie
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Ferme le client HTTP et la boucle de fond (idempotent)."""
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            try:
                close = getattr(client, "aclose", None)
                if callable(close):
                    loop = self._loop
                    if loop is not None and not loop.is_closed():
                        try:
                            asyncio.run_coroutine_threadsafe(
                                close(), loop
                            ).result(timeout=2.0)
                        except Exception as exc:
                            self.logger.debug("Fermeture client HTTP : %s", exc)
            except Exception:
                pass
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            self._loop = None
        self._loop_thread = None

    def __enter__(self) -> "LocalVLMReader":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Fonction utilitaire autonome
# --------------------------------------------------------------------------- #
async def read_handwritten_crop(
    crop: np.ndarray,
    field_type: str,
    *,
    fallback: Optional[FallbackRecognizer] = None,
    config: Optional[VLMConfig] = None,
) -> VLMResult:
    """Encode un crop manuscrit et le lit via le VLM local (appel unique).

    Fonction autonome : construit un :class:`LocalVLMReader` par appel
    (``fallback`` : reconnaisseur de secours TrOCR / PP-OCR). Pour un usage
    intensif (pages par lots), préférez l'instance réutilisée
    :class:`LocalVLMReader` et :meth:`LocalVLMReader.read_handwritten_crops_batch`.
    """
    reader = LocalVLMReader(config=config, fallback=fallback)
    try:
        return await reader.read_handwritten_crop(crop, field_type)
    finally:
        reader.close()
