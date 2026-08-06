"""Règles de gestion métier SWIT : validation, vérification BD, archivage.

Le flux de confiance d'un document OCR se déroule en trois étapes :

1. :func:`validate_ocr_fields` — contrôle des formats (regex) : CIN à 8
   chiffres, année de session, identifiant, code-barres ;
2. :func:`verify_with_database` — comparaison des données lues avec le
   référentiel :class:`~scriptvault.database.DocumentStore` et calcul d'une
   note de concordance ``match_score`` (0–100) ;
3. :class:`DocumentArchiver` — réorganisation du fichier validé sous
   ``STORAGE/{ANNEE}/{IDENTIFIANT}_{HORODATAGE}.{ext}``, empreinte SHA-256 et
   enregistrement du registre (option : chiffrement AES-256-GCM au repos).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from scriptvault.database import CandidateRecord, DocumentMeta, DocumentStore
from scriptvault.security import Aes256GcmCipher

# ---------------------------------------------------------------------------
# Règles de format (regex) — contexte Tunisie / SWIT
# ---------------------------------------------------------------------------

#: CIN tunisien : exactement 8 chiffres (12345678).
CIN_PATTERN = r"^\d{8}$"
CIN_RE = re.compile(CIN_PATTERN)

#: Année de session : 2025 ou 2024/2025 ou 2024-2025 (seconde = première + 1).
SESSION_YEAR_PATTERN = r"^(?P<year>\d{4})(?:[/-](?P<end>\d{4}))?$"
SESSION_YEAR_RE = re.compile(SESSION_YEAR_PATTERN)

#: Identifiant d'inscription SWIT : 4 à 8 chiffres, optionnellement "AAAA-NNNN".
IDENTIFIANT_PATTERN = r"^(?:\d{4}-\d{3,6}|\d{4,8})$"
IDENTIFIANT_RE = re.compile(IDENTIFIANT_PATTERN)

#: Code-barres lisible (EAN-13 / Code 128 / QR alphanumérique court).
BARCODE_PATTERN = r"^[A-Z0-9]{6,20}$"
BARCODE_RE = re.compile(BARCODE_PATTERN)


@dataclass(frozen=True)
class FieldRule:
    """Règle de validation d'un champ OCR."""

    name: str
    value: str | None
    pattern: str | None
    matched: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Données extraites & résultats
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFields:
    """Champs extraits par l'OCR, prêts pour la vérification métier."""

    nom: str | None = None
    prenom: str | None = None
    cin: str | None = None
    session_year: str | None = None
    identifiant: str | None = None
    barcode: str | None = None

    @property
    def has_identity(self) -> bool:
        return bool(self.cin or self.identifiant)


@dataclass(frozen=True)
class FieldMatch:
    """Comparaison champ à champ entre l'OCR et le référentiel."""

    field: str
    ocr_value: str
    db_value: str
    matched: bool
    weight: int
    score: float


@dataclass(frozen=True)
class VerificationResult:
    """Résultat de la vérification BD : note de concordance et statut."""

    match_score: float  # 0..100
    status: str  # "valid" | "review" | "reject"
    matched_candidate: CandidateRecord | None
    field_matches: list[FieldMatch] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "valid"


@dataclass(frozen=True)
class ArchiveResult:
    """Sortie de l'archivage : fichier réorganisé + métadonnées persistées."""

    meta: DocumentMeta
    source_path: Path
    dest_path: Path
    encrypted: bool
    sha256: str


# ---------------------------------------------------------------------------
# Utilitaires de normalisation
# ---------------------------------------------------------------------------


def _normalize(text: str | None) -> str:
    """Majuscules, sans accents ni espaces parasites (comparaison robuste)."""
    if not text:
        return ""
    value = unicodedata.normalize("NFKD", text.strip().upper())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value).strip()


def _similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, _normalize(a), _normalize(b)).ratio())


# ---------------------------------------------------------------------------
# 1. Validation des formats
# ---------------------------------------------------------------------------


def validate_ocr_fields(fields: ExtractedFields) -> list[FieldRule]:
    """Applique les regex métier à chaque champ fourni.

    * CIN : 8 chiffres ;
    * année de session : ``AAAA`` ou ``AAAA/AAAA`` (année suivante) ;
    * identifiant : ``AAAA-NNNN`` ou 4–8 chiffres ;
    * code-barres : 6–20 caractères alphanumériques.
    """
    rules: list[FieldRule] = []
    if fields.cin is not None:
        rules.append(
            _check(fields.cin, CIN_RE, "cin", "CIN à 8 chiffres (ex. 12345678)")
        )
    if fields.session_year is not None:
        valid = _valid_session_year(fields.session_year)
        rules.append(
            FieldRule(
                name="session_year",
                value=fields.session_year,
                pattern=SESSION_YEAR_PATTERN,
                matched=valid,
                message="" if valid else "Année de session invalide (ex. 2024/2025)",
            )
        )
    if fields.identifiant is not None:
        rules.append(
            _check(
                fields.identifiant,
                IDENTIFIANT_RE,
                "identifiant",
                "Identifiant invalide (ex. 2024-0001)",
            )
        )
    if fields.barcode is not None:
        rules.append(
            _check(
                fields.barcode,
                BARCODE_RE,
                "barcode",
                "Code-barres invalide (6-20 caractères)",
            )
        )
    return rules


def _check(
    value: str, regex: re.Pattern[str], name: str, error_message: str
) -> FieldRule:
    valid = bool(regex.match(value.strip()))
    return FieldRule(
        name=name,
        value=value,
        pattern=regex.pattern,
        matched=valid,
        message="" if valid else error_message,
    )


def _valid_session_year(value: str) -> bool:
    """Valide ``AAAA`` ou ``AAAA/AAAA`` avec année de fin = début + 1."""
    match = SESSION_YEAR_RE.match(value.strip())
    if not match:
        return False
    end = match.group("end")
    if end is None:
        return True
    try:
        return int(end) == int(match.group("year")) + 1
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# 2. Vérification BD & note de concordance
# ---------------------------------------------------------------------------


def compute_match_score(
    fields: ExtractedFields,
    candidates: Sequence[CandidateRecord],
) -> VerificationResult:
    """Compare les champs OCR aux candidats du référentiel.

    Pondération (total 100) : CIN 40, nom 30, prénom 20, session 10.

    * ``status`` = ``valid`` si score ≥ 85 (acceptation automatique),
    * ``review`` si score ≥ 60,
    * ``reject`` sinon — ou si le CIN est présent mais introuvable en BD.
    """
    violations: list[str] = [
        rule.message for rule in validate_ocr_fields(fields) if rule.message
    ]

    best: tuple[float, CandidateRecord | None, list[FieldMatch]] = (
        0.0,
        None,
        [],
    )
    for candidate in candidates:
        matches = _match_candidate(fields, candidate)
        score = sum(item.score for item in matches)
        if score > best[0]:
            best = (score, candidate, matches)

    score, matched_candidate, field_matches = best

    cin = _normalize(fields.cin)
    # CIN fourni mais absent du référentiel → rejet immédiat.
    if cin and matched_candidate is None:
        status = "reject"
        violations.append("CIN introuvable dans le référentiel")
    elif score >= 85:
        status = "valid"
    elif score >= 60:
        status = "review"
    else:
        status = "reject"

    details: dict[str, Any] = {
        "weights": {"cin": 40, "nom": 30, "prenom": 20, "session_year": 10},
        "fields_checked": [m.field for m in field_matches],
    }
    return VerificationResult(
        match_score=round(score, 2),
        status=status,
        matched_candidate=matched_candidate,
        field_matches=field_matches,
        violations=violations,
        details=details,
    )


def _match_candidate(
    fields: ExtractedFields, candidate: CandidateRecord
) -> list[FieldMatch]:
    """Compare un candidat précis : CIN, nom, prénom, session (pondérés)."""
    cin = _normalize(fields.cin)
    nom = _normalize(fields.nom)
    prenom = _normalize(fields.prenom)
    session_year = _normalize(fields.session_year)

    cin_ok = bool(cin) and cin == _normalize(candidate.cin)
    cin_score = 40.0 if cin_ok else 0.0

    nom_score = 0.0
    nom_ok = False
    if nom:
        ratio = _similarity(nom, candidate.nom)
        nom_score = 30.0 if ratio >= 0.9 else round(30.0 * ratio, 1)
        nom_ok = ratio >= 0.85

    prenom_score = 0.0
    prenom_ok = False
    if prenom:
        ratio = _similarity(prenom, candidate.prenom)
        prenom_score = 20.0 if ratio >= 0.9 else round(20.0 * ratio, 1)
        prenom_ok = ratio >= 0.85

    session_ok = bool(session_year) and session_year == _normalize(
        candidate.session_year
    )
    session_score = 10.0 if session_ok else 0.0

    return [
        FieldMatch("cin", cin or "-", _normalize(candidate.cin), cin_ok, 40, cin_score),
        FieldMatch("nom", nom or "-", _normalize(candidate.nom), nom_ok, 30, nom_score),
        FieldMatch(
            "prenom",
            prenom or "-",
            _normalize(candidate.prenom),
            prenom_ok,
            20,
            prenom_score,
        ),
        FieldMatch(
            "session_year",
            session_year or "-",
            _normalize(candidate.session_year),
            session_ok,
            10,
            session_score,
        ),
    ]


def verify_with_database(
    store: DocumentStore,
    fields: ExtractedFields,
    *,
    accept_threshold: int = 85,
) -> VerificationResult:
    """Orchestration : recherche candidat(s) en BD puis calcul du score."""
    cin = fields.cin.strip() if fields.cin else ""
    if cin:
        candidate = store.find_candidate_by_cin(cin)
        candidates: list[CandidateRecord] = [candidate] if candidate is not None else []
    else:
        candidates = store.find_candidates_by_name(
            nom=fields.nom or "", prenom=fields.prenom or None
        )
    result = compute_match_score(fields, candidates)
    if result.status == "valid" and result.match_score < accept_threshold:
        return VerificationResult(
            match_score=result.match_score,
            status="review",
            matched_candidate=result.matched_candidate,
            field_matches=result.field_matches,
            violations=result.violations,
            details=result.details,
        )
    return result


# ---------------------------------------------------------------------------
# 3. Réorganisation & archivage (STORAGE/{ANNEE}/{ID}_{TIMESTAMP}.ext)
# ---------------------------------------------------------------------------


def _safe_token(value: str) -> str:
    """Nettote un identifiant pour un nom de fichier (alnum, tiret, _)."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", value)
    return cleaned or "INCONNU"


def sha256_file(path: Path | str) -> str:
    """Empreinte SHA-256 d'un fichier (flux 1 Mo, mémoire constante)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_year(session_year: str | None) -> str:
    """Année de classement : session (première année) sinon année courante."""
    if session_year:
        match = SESSION_YEAR_RE.match(session_year.strip())
        if match:
            return match.group("year")
    return str(datetime.now().year)


class DocumentArchiver:
    """Réorganise les fichiers validés selon les règles de nommage SWIT.

    Cible : ``STORAGE_ROOT/{ANNEE}/{IDENTIFIANT}_{HORODATAGE}.{ext}`` où
    l'identifiant est le code-barres, sinon le CIN, sinon l'identifiant
    d'inscription. L'empreinte SHA-256 et les métadonnées sont enregistrées
    dans le :class:`DocumentStore`.
    """

    def __init__(
        self,
        store: DocumentStore,
        storage_root: Path | str = "STORAGE",
        *,
        cipher: Aes256GcmCipher | None = None,
    ) -> None:
        self._store = store
        self._root = Path(storage_root)
        self._cipher = cipher

    @property
    def storage_root(self) -> Path:
        return self._root

    def archive(
        self,
        source_path: Path | str,
        fields: ExtractedFields,
        verification: VerificationResult,
        *,
        page_count: int = 1,
        keep_original: bool = True,
    ) -> ArchiveResult:
        """Copie le fichier validé vers sa destination réorganisée.

        * le fichier est copié (source conservée) puis chiffré sur place si un
          chiffreur est configuré ;
        * le nom de destination suit ``{ANNEE}/{ID}_{TIMESTAMP}.{ext}`` ;
        * le registre (métadonnées + SHA-256) est mis à jour en BD.
        """
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Fichier source absent : {source}")

        year = infer_year(fields.session_year)
        identifier = _safe_token(
            fields.barcode or fields.cin or fields.identifiant or "INCONNU"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self._root / year / f"{identifier}_{timestamp}{source.suffix.lower()}"

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

        encrypted = False
        if self._cipher is not None:
            encrypted_path = dest.with_suffix(dest.suffix + ".enc")
            self._cipher.encrypt_file(dest, encrypted_path)
            dest.unlink()
            dest = encrypted_path
            encrypted = True

        digest = sha256_file(dest)
        meta = DocumentMeta(
            original_filename=source.name,
            stored_path=str(dest.relative_to(self._root))
            if dest.is_relative_to(self._root)
            else str(dest),
            year=year,
            identifier=identifier,
            barcode=fields.barcode,
            cin=fields.cin,
            sha256=digest,
            size_bytes=dest.stat().st_size,
            page_count=page_count,
            encrypted=encrypted,
            status=verification.status,
            match_score=verification.match_score,
            match_details=verification.details,
        )
        self._store.save_document(meta)

        if not keep_original and source != dest:
            source.unlink(missing_ok=True)

        return ArchiveResult(
            meta=meta,
            source_path=source,
            dest_path=dest,
            encrypted=encrypted,
            sha256=digest,
        )
