"""Tests des règles métier SWIT (validation regex, concordance BD, archivage)."""

from __future__ import annotations

from pathlib import Path

import pytest
from scriptvault.database import (
    CandidateInput,
    DocumentStore,
    create_database_engine,
)
from scriptvault.rules_engine import (
    DocumentArchiver,
    ExtractedFields,
    infer_year,
    sha256_file,
    validate_ocr_fields,
    verify_with_database,
)
from scriptvault.security import Aes256GcmCipher


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    db = tmp_path / "referentiel.db"
    engine = create_database_engine(f"sqlite:///{db.as_posix()}")
    store = DocumentStore(engine)
    store.create_all()
    store.upsert_candidates(
        [
            CandidateInput(
                cin="12345678",
                nom="Ben Ali",
                prenom="Ahmed",
                session_year="2024/2025",
                classe="4eme",
            ),
            CandidateInput(
                cin="87654321",
                nom="Trabelsi",
                prenom="Sonia",
                session_year="2023/2024",
            ),
        ]
    )
    return store


class TestValidationRegex:
    def test_cin_valide(self) -> None:
        rules = validate_ocr_fields(ExtractedFields(cin="12345678"))
        assert next(r for r in rules if r.name == "cin").matched

    def test_cin_invalide(self) -> None:
        rules = validate_ocr_fields(ExtractedFields(cin="1234"))
        rule = next(r for r in rules if r.name == "cin")
        assert not rule.matched
        assert "8 chiffres" in rule.message

    def test_session_year_ok(self) -> None:
        for value in ("2024/2025", "2024-2025", "2025"):
            rules = validate_ocr_fields(ExtractedFields(session_year=value))
            assert next(r for r in rules if r.name == "session_year").matched

    def test_session_year_non_consecutive(self) -> None:
        rules = validate_ocr_fields(ExtractedFields(session_year="2017/2025"))
        assert not next(r for r in rules if r.name == "session_year").matched

    def test_identifiant_et_barcode(self) -> None:
        fields = ExtractedFields(identifiant="2024-0001", barcode="25052024")
        rules = {r.name: r for r in validate_ocr_fields(fields)}
        assert rules["identifiant"].matched
        assert rules["barcode"].matched

    def test_champs_vides_non_verifies(self) -> None:
        assert validate_ocr_fields(ExtractedFields()) == []


class TestConcordanceBD:
    def test_concordance_totale(self, store: DocumentStore) -> None:
        fields = ExtractedFields(
            nom="Ben Ali",
            prenom="Ahmed",
            cin="12345678",
            session_year="2024/2025",
        )
        result = verify_with_database(store, fields)
        assert result.status == "valid"
        assert result.match_score == 100.0
        assert result.matched_candidate is not None
        assert result.matched_candidate.cin == "12345678"

    def test_fuzzy_nom_penalise(self, store: DocumentStore) -> None:
        fields = ExtractedFields(
            nom="Benali", prenom="Ahmed", cin="12345678", session_year="2024/2025"
        )
        result = verify_with_database(store, fields)
        assert result.match_score > 95.0
        assert result.status == "valid"

    def test_cin_inconnu_rejete(self, store: DocumentStore) -> None:
        fields = ExtractedFields(cin="99999999")
        result = verify_with_database(store, fields)
        assert result.status == "reject"
        assert any("introuvable" in v for v in result.violations)

    def test_score_partiel_revision(self, store: DocumentStore) -> None:
        # CIN bon, mais nom/prénom/session complètement différents.
        fields = ExtractedFields(
            cin="12345678",
            nom="Xxx",
            prenom="Yyy",
            session_year="1999/2000",
        )
        result = verify_with_database(store, fields)
        assert result.status == "reject"
        assert result.match_score < 60.0

    def test_seuil_personnalise(self, store: DocumentStore) -> None:
        # CIN + nom + prénom (sans session) → 90/100 : valid par défaut,
        # mais passé en révision si le seuil d'acceptation est relevé à 95.
        fields = ExtractedFields(cin="12345678", nom="Ben Ali", prenom="Ahmed")
        default = verify_with_database(store, fields)
        assert default.status == "valid"
        assert default.match_score == 90.0
        strict = verify_with_database(store, fields, accept_threshold=95)
        assert strict.status == "review"
        assert strict.match_score == 90.0

    def test_recherche_par_nom(self, store: DocumentStore) -> None:
        # Sans CIN, le score plafonne à 60 (nom+prénom+session) → révision
        # systématique : un opérateur confirme l'identité.
        fields = ExtractedFields(
            nom="Trabelsi", prenom="Sonia", session_year="2023/2024"
        )
        result = verify_with_database(store, fields)
        assert result.status == "review"
        assert result.match_score == 60.0
        assert result.matched_candidate is not None


class TestArchivage:
    def test_renommage_et_registre(self, store: DocumentStore, tmp_path: Path) -> None:
        source = tmp_path / "scan_001.tif"
        source.write_bytes(b"%PDF-contenu-bacalaureat")
        fields = ExtractedFields(
            barcode="25052024",
            cin="12345678",
            nom="Ben Ali",
            prenom="Ahmed",
            session_year="2024/2025",
        )
        verification = verify_with_database(store, fields)

        archiver = DocumentArchiver(store, tmp_path / "STORAGE")
        result = archiver.archive(source, fields, verification, page_count=3)

        assert result.meta.year == "2024"
        assert result.dest_path.name.startswith("25052024_")
        assert result.dest_path.suffix == ".tif"
        assert not result.encrypted
        assert result.sha256 == sha256_file(result.dest_path)
        assert source.exists()  # copie conservée par défaut

        record = store.find_document_by_sha256(result.sha256)
        assert record is not None
        assert record.identifier == "25052024"
        assert record.page_count == 3
        assert record.match_score == 100.0

    def test_archivage_chiffre(self, store: DocumentStore, tmp_path: Path) -> None:
        source = tmp_path / "scan_002.tif"
        source.write_bytes(b"contenu-confidentiel")
        fields = ExtractedFields(cin="87654321", session_year="2023/2024")
        verification = verify_with_database(store, fields)

        cipher = Aes256GcmCipher("passphrase-de-test")
        archiver = DocumentArchiver(store, tmp_path / "STORAGE", cipher=cipher)
        result = archiver.archive(source, fields, verification)

        assert result.encrypted
        assert result.dest_path.suffix == ".enc"
        raw = result.dest_path.read_bytes()
        assert b"contenu-confidentiel" not in raw

        # Le contenu original doit être restituable avec la bonne clé.
        restored = tmp_path / "restored.tif"
        cipher.decrypt_file(result.dest_path, restored)
        assert restored.read_bytes() == b"contenu-confidentiel"

        record = store.find_document_by_sha256(result.sha256)
        assert record is not None
        assert record.encrypted is True

    def test_fichier_source_absent(self, store: DocumentStore, tmp_path: Path) -> None:
        archiver = DocumentArchiver(store, tmp_path / "STORAGE")
        with pytest.raises(FileNotFoundError):
            archiver.archive(
                tmp_path / "absent.tif",
                ExtractedFields(cin="12345678"),
                verify_with_database(store, ExtractedFields(cin="12345678")),
            )

    def test_infer_year(self) -> None:
        assert infer_year("2024/2025") == "2024"
        assert infer_year("2024-2025") == "2024"
        assert infer_year("2025") == "2025"
        assert len(infer_year(None)) == 4
