"""Sécurité de ScriptVault OCR : hachage des mots de passe, JWT et chiffrement.

Trois briques indépendantes :

* :class:`PasswordHasher`  — hachage Argon2id (passlib), repli PBKDF2-SHA256.
* :class:`TokenService`    — émission / validation de JWT (HS256, PyJWT).
* :class:`Aes256GcmCipher` — chiffrement authentifié AES-256-GCM des fichiers
  au repos (via ``cryptography``), clé maîtresse ou dérivée par PBKDF2.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    _CRYPTO_OK = True
except ImportError:  # pragma: no cover - environnement sans cryptography
    _CRYPTO_OK = False
    AESGCM = None  # type: ignore[assignment]
    Cipher = None  # type: ignore[assignment]
    algorithms = None  # type: ignore[assignment]
    modes = None  # type: ignore[assignment]

try:
    import jwt as _jwt  # PyJWT

    _JWT_OK = True
except ImportError:  # pragma: no cover - PyJWT absent
    _jwt = None  # type: ignore[assignment]
    _JWT_OK = False

try:
    from passlib.context import CryptContext

    _PASSLIB_OK = True
except ImportError:  # pragma: no cover - passlib absent
    _PASSLIB_OK = False
    CryptContext = None  # type: ignore[assignment]


class SecurityError(RuntimeError):
    """Erreur de sécurité générique (hachage, JWT ou chiffrement)."""


class PasswordHasher:
    """Hachage sécurisé des mots de passe (Argon2id, sel aléatoire).

    Format des condensats : PHC ``$argon2id$...`` (passlib) ; en l'absence de
    passlib, un repli PBKDF2-HMAC-SHA256 (310 000 itérations) est utilisé.
    """

    SCHEME: Final[str] = "argon2"
    _PBKDF2_ITERATIONS: Final[int] = 310_000
    _SALT_BYTES: Final[int] = 16

    def __init__(self) -> None:
        self._ctx = (
            CryptContext(schemes=["argon2"], deprecated="auto") if _PASSLIB_OK else None
        )

    def hash(self, password: str) -> str:
        """Retourne un condensat autoportant (sel + paramètres inclus)."""
        if self._ctx is not None:
            return self._ctx.hash(password)
        salt = os.urandom(self._SALT_BYTES)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self._PBKDF2_ITERATIONS
        )
        return (
            f"pbkdf2_sha256${self._PBKDF2_ITERATIONS}$"
            f"{base64.b64encode(salt).decode('ascii')}$"
            f"{base64.b64encode(digest).decode('ascii')}"
        )

    def verify(self, password: str, encoded: str) -> bool:
        """Vérifie un mot de passe contre un condensat produit par :meth:`hash`."""
        if self._ctx is not None:
            try:
                return bool(self._ctx.verify(password, encoded))
            except ValueError:
                return False
        try:
            scheme, iters, salt_b64, digest_b64 = encoded.split("$")
            if scheme != "pbkdf2_sha256":
                return False
            salt = base64.b64decode(salt_b64)
            digest = base64.b64decode(digest_b64)
            candidate = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, int(iters)
            )
            return hmac.compare_digest(candidate, digest)
        except (ValueError, TypeError):
            return False


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class TokenService:
    """Émission et validation de JSON Web Tokens (signature HMAC-SHA256)."""

    algorithm: Final[str] = "HS256"

    def __init__(
        self,
        secret: str,
        issuer: str = "scriptvault-ocr",
        audience: str = "scriptvault-web",
        expire_minutes: int = 480,
    ) -> None:
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._expire_minutes = expire_minutes

    def issue(
        self,
        subject: str,
        roles: list[str] | None = None,
        **claims: object,
    ) -> str:
        """Crée un token signé pour un sujet, avec rôles et claims libres."""
        if not _JWT_OK:
            raise SecurityError("PyJWT n'est pas installé.")
        now = _now_utc()
        payload: dict[str, object] = {
            "sub": subject,
            "iss": self._issuer,
            "aud": self._audience,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(minutes=self._expire_minutes),
            "roles": roles or [],
        }
        payload.update(claims)
        return _jwt.encode(payload, self._secret, algorithm=self.algorithm)  # type: ignore[union-attr]

    def validate(self, token: str) -> dict[str, object]:
        """Valide signature, émetteur, audience et expiration du token."""
        if not _JWT_OK:
            raise SecurityError("PyJWT n'est pas installé.")
        try:
            decoded: dict[str, object] = _jwt.decode(  # type: ignore[union-attr]
                token,
                self._secret,
                algorithms=[self.algorithm],
                issuer=self._issuer,
                audience=self._audience,
            )
            return decoded
        except Exception as exc:  # ExpiredSignatureError / InvalidTokenError...
            raise SecurityError(f"Token invalide : {exc}") from exc


class Aes256GcmCipher:
    """Chiffrement authentifié AES-256-GCM (les données chiffrées sont non
    altérables : toute modification fait échouer le déchiffrement).

    Format d'un fichier chiffré : ``salt(16) | nonce(12) | ciphertext | tag(16)``.

    * ``master_key_b64`` : clé maîtresse de 32 octets (base64) — c'est alors
      une véritable clé AES-256 persistante ;
    * sinon : clé dérivée de la passphrase par PBKDF2-HMAC-SHA256 (600 000
      itérations) avec un sel aléatoire par fichier.

    Les gros fichiers (TIF multi-page) sont traités en flux : un unique
    contexte GCM est ouvert et alimenté par blocs de 1 Mo (aucun rejeu de
    nonce, consommation mémoire constante).
    """

    KEY_BYTES: Final[int] = 32
    SALT_BYTES: Final[int] = 16
    NONCE_BYTES: Final[int] = 12
    TAG_BYTES: Final[int] = 16
    _PBKDF2_ITERATIONS: Final[int] = 600_000
    _CHUNK: Final[int] = 1024 * 1024
    _HEADER_BYTES: Final[int] = SALT_BYTES + NONCE_BYTES

    def __init__(self, passphrase: str, master_key_b64: str = "") -> None:
        if not _CRYPTO_OK:
            raise SecurityError(
                "cryptography n'est pas installé — chiffrement indisponible."
            )
        if master_key_b64:
            try:
                self._key = base64.b64decode(master_key_b64, validate=True)
            except ValueError as exc:
                raise SecurityError("Clé maîtresse base64 invalide.") from exc
            if len(self._key) != self.KEY_BYTES:
                raise SecurityError("La clé maîtresse doit contenir 32 octets.")
            self._use_master = True
            self._passphrase = ""
        else:
            if not passphrase:
                raise SecurityError(
                    "Passphrase vide : refuse d'utiliser une clé nulle."
                )
            self._use_master = False
            self._passphrase = passphrase
            self._key = b""

    def _key_for(self, salt: bytes) -> bytes:
        """Clé de session : maître fixe, ou dérivée du sel (PBKDF2)."""
        if self._use_master:
            return self._key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_BYTES,
            salt=salt,
            iterations=self._PBKDF2_ITERATIONS,
        )
        return kdf.derive(self._passphrase.encode("utf-8"))

    # --- Octets -----------------------------------------------------------

    def encrypt_bytes(self, payload: bytes) -> bytes:
        """Chiffre des octets → ``salt|nonce|ciphertext|tag``."""
        salt = os.urandom(self.SALT_BYTES)
        nonce = os.urandom(self.NONCE_BYTES)
        key = self._key_for(salt)
        ciphertext = AESGCM(key).encrypt(nonce, payload, None)
        return salt + nonce + ciphertext

    def decrypt_bytes(self, blob: bytes) -> bytes:
        """Déchiffre un blob produit par :meth:`encrypt_bytes`."""
        if len(blob) < self._HEADER_BYTES + self.TAG_BYTES:
            raise SecurityError("Blob chiffré tronqué ou corrompu.")
        salt = blob[: self.SALT_BYTES]
        nonce = blob[self.SALT_BYTES : self._HEADER_BYTES]
        ciphertext = blob[self._HEADER_BYTES :]
        key = self._key_for(salt)
        try:
            return AESGCM(key).decrypt(nonce, ciphertext, None)
        except Exception as exc:  # InvalidTag
            raise SecurityError(
                "Déchiffrement refusé (tag d'authenticité invalide)."
            ) from exc

    # --- Fichiers (flux, un seul contexte GCM) -----------------------------

    def encrypt_file(self, src: Path | str, dst: Path | str) -> Path:
        """Chiffre ``src`` vers ``dst`` en flux (en-tête puis blocs GCM)."""
        src_path = Path(src)
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        salt = os.urandom(self.SALT_BYTES)
        nonce = os.urandom(self.NONCE_BYTES)
        key = self._key_for(salt)
        encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
        with src_path.open("rb") as fin, dst_path.open("wb") as fout:
            fout.write(salt)
            fout.write(nonce)
            while chunk := fin.read(self._CHUNK):
                fout.write(encryptor.update(chunk))
            encryptor.finalize()
            fout.write(encryptor.tag)
        return dst_path

    def decrypt_file(self, src: Path | str, dst: Path | str) -> Path:
        """Déchiffre ``src`` (format ci-dessus) vers ``dst`` en flux."""
        src_path = Path(src)
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        size = src_path.stat().st_size
        if size < self._HEADER_BYTES + self.TAG_BYTES:
            raise SecurityError("Fichier chiffré tronqué ou corrompu.")
        with src_path.open("rb") as fin:
            header = fin.read(self._HEADER_BYTES)
            salt, nonce = header[: self.SALT_BYTES], header[self.SALT_BYTES :]
            payload_size = size - self._HEADER_BYTES - self.TAG_BYTES
            fin.seek(self._HEADER_BYTES + payload_size)
            tag = fin.read(self.TAG_BYTES)
            fin.seek(self._HEADER_BYTES)
            key = self._key_for(salt)
            decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).decryptor()
            try:
                remaining = payload_size
                with dst_path.open("wb") as fout:
                    while remaining > 0:
                        chunk = fin.read(min(self._CHUNK, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        fout.write(decryptor.update(chunk))
                    decryptor.finalize_with_tag(tag)
            except Exception as exc:  # InvalidTag / ValueError
                dst_path.unlink(missing_ok=True)
                raise SecurityError(
                    "Déchiffrement refusé (clé invalide ou fichier corrompu)."
                ) from exc
        return dst_path
