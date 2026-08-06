"""Persistance SQLAlchemy : référentiel candidats + registre des documents.

Deux tables :

* ``candidates``     — référentiel BD (CIN, nom, prénom, année de session)
  contre lequel les données OCR sont comparées (note de concordance) ;
* ``stored_documents`` — registre d'archivage : chemin réorganisé, code-barres,
  CIN, empreinte SHA-256 (inviolabilité), score de correspondance, etc.

Bases de données supportées :

* SQLite (défaut, ``sqlite:///scriptvault.db``) ;
* SQLite chiffré via SQLCipher (``SCRIPTVAULT_DB_ENCRYPT=1`` + ``DB_KEY``) —
  exige le pilote ``sqlcipher3`` / ``pysqlcipher3`` ;
* PostgreSQL (toute URL ``postgresql://...``).
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence
from urllib.parse import quote_plus

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    create_engine,
    event,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class DatabaseError(RuntimeError):
    """Erreur d'initialisation ou d'accès à la base de données."""


class Base(DeclarativeBase):
    """Base déclarative commune aux modèles du référentiel."""


class CandidateRecord(Base):
    """Enregistrement de référence (ex. candidat au bac, Tunisie / SWIT)."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cin: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    nom: Mapped[str] = mapped_column(String(120), index=True)
    prenom: Mapped[str] = mapped_column(String(120))
    session_year: Mapped[str] = mapped_column(String(16), index=True)
    classe: Mapped[str | None] = mapped_column(String(60), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:  # pragma: no cover - outil de debug
        return f"<Candidate cin={self.cin!r} nom={self.nom!r} {self.prenom!r}>"


class StoredDocument(Base):
    """Métadonnées d'un fichier validé et réorganisé sur disque."""

    __tablename__ = "stored_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    year: Mapped[str] = mapped_column(String(16), index=True)
    identifier: Mapped[str] = mapped_column(String(64), index=True)
    barcode: Mapped[str | None] = mapped_column(String(128), default=None)
    cin: Mapped[str | None] = mapped_column(String(16), index=True, default=None)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    encrypted: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="archived")
    match_score: Mapped[float | None] = mapped_column(Float, default=None)
    match_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True)
class CandidateInput:
    """Ligne de référence à insérer / mettre à jour dans le référentiel."""

    cin: str
    nom: str
    prenom: str
    session_year: str
    classe: str | None = None


@dataclass(frozen=True)
class DocumentMeta:
    """Métadonnées d'archivage d'un fichier OCR validé."""

    original_filename: str
    stored_path: str
    year: str
    identifier: str
    sha256: str
    size_bytes: int
    page_count: int = 1
    barcode: str | None = None
    cin: str | None = None
    encrypted: bool = False
    status: str = "archived"
    match_score: float | None = None
    match_details: dict[str, Any] = field(default_factory=dict)


def _sqlcipher_available() -> bool:
    return any(
        importlib.util.find_spec(driver) is not None
        for driver in ("sqlcipher3", "pysqlcipher3")
    )


def create_database_engine(
    url: str = "sqlite:///scriptvault.db",
    *,
    encrypt: bool = False,
    key: str = "",
) -> Engine:
    """Construit l'``Engine`` SQLAlchemy selon la configuration fournie.

    * ``url`` commençant par ``sqlite:///`` : base locale ;
    * ``encrypt=True`` : active SQLCipher (exige un pilote ``sqlcipher3`` et
      une clé, transmise via le mot de passe de l'URL ``sqlite+pysqlcipher://``) ;
    * toute autre URL (``postgresql://``…) est passée telle quelle.
    """
    if not url.strip():
        raise DatabaseError("URL de base de données vide.")
    url = url.strip()

    if url.startswith("sqlite:///") and encrypt:
        if not key:
            raise DatabaseError(
                "SQLite chiffré (SQLCipher) : SCRIPTVAULT_DB_KEY est requis."
            )
        if not _sqlcipher_available():
            raise DatabaseError(
                "SQLite chiffré requis mais le pilote sqlcipher3/pysqlcipher3 "
                "n'est pas installé (pip install sqlcipher3-binary)."
            )
        path = url[len("sqlite:///") :]
        url = f"sqlite+pysqlcipher://:{quote_plus(key)}@/{path}"

    connect_args: dict[str, Any] = {"check_same_thread": False}
    kwargs: dict[str, Any] = {"connect_args": connect_args}
    if url.startswith("postgresql://"):
        kwargs["pool_pre_ping"] = True
        kwargs.pop("connect_args")
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite+pysqlcipher://"):
        # PRAGMAs de sécurité SQLCipher (verrouillage + WAL) une fois connecté.
        @event.listens_for(engine, "connect")
        def _sqlcipher_pragmas(dbapi_connection: Any, _record: Any) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA cipher_memory_security = ON;")
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.close()

    elif url.startswith("sqlite://"):
        # Contrainte d'intégrité + mode WAL pour les accès concurrents.
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.close()

    return engine


class DocumentStore:
    """Accès typé au référentiel (candidats) et au registre d'archivage."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._factory = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Session transactionnelle (commit/rollback automatiques)."""
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all(self) -> None:
        """Crée les tables si elles n'existent pas."""
        Base.metadata.create_all(self._engine)

    # --- Référentiel candidats -------------------------------------------

    def upsert_candidates(self, rows: Sequence[CandidateInput]) -> int:
        """Insère ou met à jour le référentiel ; retourne le nombre de lignes."""
        if not rows:
            return 0
        count = 0
        with self.session() as session:
            for row in rows:
                existing = session.scalar(
                    select(CandidateRecord).where(
                        CandidateRecord.cin == row.cin.strip()
                    )
                )
                if existing is None:
                    session.add(
                        CandidateRecord(
                            cin=row.cin.strip(),
                            nom=row.nom.strip().upper(),
                            prenom=row.prenom.strip().upper(),
                            session_year=row.session_year.strip(),
                            classe=row.classe,
                        )
                    )
                    count += 1
                else:
                    existing.nom = row.nom.strip().upper()
                    existing.prenom = row.prenom.strip().upper()
                    existing.session_year = row.session_year.strip()
                    existing.classe = row.classe
                    count += 1
        return count

    def find_candidate_by_cin(self, cin: str) -> CandidateRecord | None:
        with self.session() as session:
            return session.scalar(
                select(CandidateRecord).where(CandidateRecord.cin == cin.strip())
            )

    def find_candidates_by_name(
        self,
        nom: str,
        prenom: str | None = None,
        session_year: str | None = None,
        limit: int = 20,
    ) -> list[CandidateRecord]:
        """Candidats dont le nom (et éventuellement prénom) correspond."""
        stmt = select(CandidateRecord).where(CandidateRecord.nom == nom.strip().upper())
        if prenom:
            stmt = stmt.where(CandidateRecord.prenom == prenom.strip().upper())
        if session_year:
            stmt = stmt.where(CandidateRecord.session_year == session_year.strip())
        stmt = stmt.limit(max(1, limit))
        with self.session() as session:
            return list(session.scalars(stmt).all())

    def list_candidates(
        self, limit: int = 100, offset: int = 0
    ) -> list[CandidateRecord]:
        with self.session() as session:
            stmt = (
                select(CandidateRecord)
                .order_by(CandidateRecord.cin)
                .limit(max(1, limit))
                .offset(max(0, offset))
            )
            return list(session.scalars(stmt).all())

    def count_candidates(self) -> int:
        from sqlalchemy import func

        with self.session() as session:
            return int(
                session.scalar(select(func.count()).select_from(CandidateRecord))
            )

    # --- Registre des documents archivés ----------------------------------

    def save_document(self, meta: DocumentMeta) -> StoredDocument:
        """Enregistre (ou remplace, par empreinte) un document archivé."""
        with self.session() as session:
            existing = session.scalar(
                select(StoredDocument).where(StoredDocument.sha256 == meta.sha256)
            )
            if existing is not None:
                existing.original_filename = meta.original_filename
                existing.stored_path = meta.stored_path
                existing.year = meta.year
                existing.identifier = meta.identifier
                existing.barcode = meta.barcode
                existing.cin = meta.cin
                existing.size_bytes = meta.size_bytes
                existing.page_count = meta.page_count
                existing.encrypted = meta.encrypted
                existing.status = meta.status
                existing.match_score = meta.match_score
                existing.match_details = meta.match_details
                session.flush()
                return existing
            record = StoredDocument(
                original_filename=meta.original_filename,
                stored_path=meta.stored_path,
                year=meta.year,
                identifier=meta.identifier,
                barcode=meta.barcode,
                cin=meta.cin,
                sha256=meta.sha256,
                size_bytes=meta.size_bytes,
                page_count=meta.page_count,
                encrypted=meta.encrypted,
                status=meta.status,
                match_score=meta.match_score,
                match_details=meta.match_details,
            )
            session.add(record)
            session.flush()
            return record

    def find_document_by_sha256(self, sha256: str) -> StoredDocument | None:
        with self.session() as session:
            return session.scalar(
                select(StoredDocument).where(StoredDocument.sha256 == sha256)
            )

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[StoredDocument]:
        with self.session() as session:
            stmt = (
                select(StoredDocument)
                .order_by(StoredDocument.created_at.desc())
                .limit(max(1, limit))
                .offset(max(0, offset))
            )
            return list(session.scalars(stmt).all())

    def count_documents(self) -> int:
        from sqlalchemy import func

        with self.session() as session:
            return int(session.scalar(select(func.count()).select_from(StoredDocument)))

    def statistics(self) -> dict[str, Any]:
        """Indicateurs du registre (total, chiffrés, révisés, score moyen)."""
        from sqlalchemy import func

        with self.session() as session:
            total = int(
                session.scalar(select(func.count()).select_from(StoredDocument))
            )
            encrypted = int(
                session.scalar(
                    select(func.count())
                    .select_from(StoredDocument)
                    .where(StoredDocument.encrypted.is_(True))
                )
            )
            reviewed = int(
                session.scalar(
                    select(func.count())
                    .select_from(StoredDocument)
                    .where(StoredDocument.status == "review")
                )
            )
            avg_score = session.scalar(
                select(func.avg(StoredDocument.match_score)).where(
                    StoredDocument.match_score.is_not(None)
                )
            )
            return {
                "total_documents": total,
                "encrypted_files": encrypted,
                "pending_review": reviewed,
                "avg_match_score": round(float(avg_score), 2) if avg_score else None,
            }
