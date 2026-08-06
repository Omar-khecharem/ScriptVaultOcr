"""Tests des primitives de sécurité : hachage, JWT et chiffrement AES-256-GCM."""

from __future__ import annotations

import base64

import pytest
from scriptvault.security import (
    Aes256GcmCipher,
    PasswordHasher,
    SecurityError,
    TokenService,
)

SECRET = "une-cle-jwt-de-32-octets-minimum-svp-0123456789"


class TestPasswordHasher:
    def test_roundtrip(self) -> None:
        hasher = PasswordHasher()
        encoded = hasher.hash("MotDePasse!2024")
        assert hasher.verify("MotDePasse!2024", encoded)
        assert not hasher.verify("mauvais", encoded)

    def test_sel_aleatoire(self) -> None:
        hasher = PasswordHasher()
        assert hasher.hash("secret") != hasher.hash("secret")

    def test_condensat_du_hash(self) -> None:
        hasher = PasswordHasher()
        assert hasher.hash("secret").startswith("$argon2id$")


class TestTokenService:
    def test_issue_validate(self) -> None:
        tokens = TokenService(SECRET, expire_minutes=60)
        token = tokens.issue("agent-42", roles=["admin", "ocr"])
        claims = tokens.validate(token)
        assert claims["sub"] == "agent-42"
        assert claims["roles"] == ["admin", "ocr"]
        assert claims["iss"] == "scriptvault-ocr"

    def test_token_modifie_rejete(self) -> None:
        tokens = TokenService(SECRET)
        token = tokens.issue("agent-42")
        with pytest.raises(SecurityError):
            tokens.validate(token + "x")

    def test_mauvaise_cle_rejetee(self) -> None:
        token = TokenService(SECRET).issue("agent-42")
        with pytest.raises(SecurityError):
            TokenService("autre-cle-totalement-differente-0123456789").validate(token)


class TestAes256GcmCipher:
    def test_roundtrip_bytes(self) -> None:
        cipher = Aes256GcmCipher("passphrase")
        blob = cipher.encrypt_bytes(b"texte secret")
        assert blob[:16] != b"texte secret"
        assert cipher.decrypt_bytes(blob) == b"texte secret"

    def test_entropie_des_nonces(self) -> None:
        cipher = Aes256GcmCipher("passphrase")
        assert cipher.encrypt_bytes(b"a") != cipher.encrypt_bytes(b"a")

    def test_blob_corrompu_rejete(self) -> None:
        cipher = Aes256GcmCipher("passphrase")
        blob = bytearray(cipher.encrypt_bytes(b"donnees"))
        blob[-1] ^= 0xFF  # altération du tag
        with pytest.raises(SecurityError):
            cipher.decrypt_bytes(bytes(blob))

    def test_mauvaise_cle_rejete(self) -> None:
        blob = Aes256GcmCipher("passphrase-a").encrypt_bytes(b"donnees")
        with pytest.raises(SecurityError):
            Aes256GcmCipher("passphrase-b").decrypt_bytes(blob)

    def test_roundtrip_fichier(self, tmp_path) -> None:
        cipher = Aes256GcmCipher("passphrase")
        source = tmp_path / "scan.tif"
        source.write_bytes(b"TIF-multipage" * 300_000)
        encrypted = tmp_path / "scan.tif.enc"
        restored = tmp_path / "scan.restored.tif"

        cipher.encrypt_file(source, encrypted)
        assert encrypted.stat().st_size > source.stat().st_size
        assert b"TIF-multipage" not in encrypted.read_bytes()[:64]

        cipher.decrypt_file(encrypted, restored)
        assert restored.read_bytes() == source.read_bytes()

    def test_cle_maitresse_base64(self) -> None:
        key = base64.b64encode(b"K" * 32).decode("ascii")
        cipher = Aes256GcmCipher("inutilisee", master_key_b64=key)
        blob = cipher.encrypt_bytes(b"x")
        assert cipher.decrypt_bytes(blob) == b"x"

    def test_cle_maitresse_invalide(self) -> None:
        with pytest.raises(SecurityError):
            Aes256GcmCipher("", master_key_b64="aGVsbG8=")  # 5 octets seulement
