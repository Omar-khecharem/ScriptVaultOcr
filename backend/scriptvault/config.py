"""Configuration du serveur ScriptVault OCR, pilotée par variables d'environnement.

Toutes les options exposent un préfixe ``SCRIPTVAULT_`` et peuvent être
surchargées par la ligne de commande (``python main.py --port 9000``).

Exemple::

    SCRIPTVAULT_PORT=9000 SCRIPTVAULT_LANG=fr python main.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "1" if default else "0").lower()
    return value in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = _env(name, "")
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_roi(
    name: str,
) -> dict[str, tuple[float, float, float, float]]:
    """Analyse un profil de zones d'intérêt au format JSON.

    Exemple::

        SCRIPTVAULT_ROI='{"nom": [0.02, 0.09, 0.55, 0.14], "cin": [0.02, 0.23, 0.40, 0.28]}'

    Chaque zone est une fraction normalisée ``(x0, y0, x1, y1)`` de la page.
    """
    raw = _env(name, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    output: dict[str, tuple[float, float, float, float]] = {}
    if not isinstance(payload, dict):
        return {}
    for label, spec in payload.items():
        if not isinstance(spec, (list, tuple)) or len(spec) != 4:
            continue
        try:
            output[str(label)] = (
                float(spec[0]),
                float(spec[1]),
                float(spec[2]),
                float(spec[3]),
            )
        except (TypeError, ValueError):
            continue
    return output


@dataclass(frozen=True)
class Settings:
    """Configuration immuable du serveur (dérivée de l'environnement)."""

    # --- Réseau ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # --- Moteur OCR ------------------------------------------------------
    lang: str = "en"
    model_dir: str | None = None
    cpu_threads: int = 0  # 0 = auto (min(8, cœurs))
    max_concurrency: int = 1  # inférences simultanées (1 = recommandé CPU)
    timeout_ms: int = 300_000  # délai maximal d'une image (0 = illimité)
    preload: bool = True  # pré-charge le moteur au démarrage
    preprocess: bool = True  # pipeline OpenCV par défaut (CLAHE/deskew/Otsu)

    # --- Scalabilité multi-processing ------------------------------------
    workers: int = 0  # workers du pool ProcessPoolExecutor (0 = 1 seul, RAM-safe)
    use_processes: bool = True  # pool multi-processus (repli thread sinon)

    # --- Haute résolution ------------------------------------------------
    max_side_len: int = 0  # longueur max côté OCR (0 = SCRIPTVAULT_MAX_SIDE / 1600)

    # --- Lecture hybride (code-barres + ROI) ------------------------------
    barcode_enabled: bool = True  # scanner local code-barres / QR
    barcode_budget_ms: int = 15  # budget de détection par page
    roi_enabled: bool = False  # préchauffe le prédicteur rec-only (det=False)
    roi_profile: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )  # profil de zones d'intérêt (JSON SCRIPTVAULT_ROI)

    # --- Règles métier & stockage -----------------------------------------
    storage_root: str = "STORAGE"  # racine de réorganisation des fichiers
    archive_encrypt: bool = False  # chiffre AES-256-GCM les fichiers archivés
    accept_threshold: int = 85  # match_score (%) pour acceptation automatique

    # --- Sécurité / Authentification -------------------------------------
    auth_enabled: bool = False  # active l'authentification JWT (opt-in)
    jwt_secret: str = "scriptvault-dev-secret-change-me"
    jwt_issuer: str = "scriptvault-ocr"
    jwt_audience: str = "scriptvault-web"
    jwt_expire_minutes: int = 480
    master_key: str = ""  # clé AES-256 (base64) — dérivée de la passphrase sinon

    # --- Base de données --------------------------------------------------
    db_url: str = "sqlite:///scriptvault.db"  # SQLAlchemy (PostgreSQL supporté)
    db_encrypt: bool = False  # SQLite chiffré (SQLCipher) — exige sqlcipher3
    db_key: str = ""  # clé SQLCipher (SQLite chiffré) ou mot de passe BD

    # --- Limites d'upload ------------------------------------------------
    max_file_mb: int = 25
    allowed_extensions: tuple[str, ...] = field(
        default_factory=lambda: (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".webp",
            ".bmp",
            ".pdf",
        )
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Construit la configuration depuis les variables ``SCRIPTVAULT_*``."""
        return cls(
            host=_env("SCRIPTVAULT_HOST", "127.0.0.1"),
            port=_env_int("SCRIPTVAULT_PORT", 8000),
            cors_origins=_env_csv(
                "SCRIPTVAULT_CORS_ORIGINS",
                ["http://localhost:5173", "http://127.0.0.1:5173"],
            ),
            lang=_env("SCRIPTVAULT_LANG", "en"),
            model_dir=_env("SCRIPTVAULT_MODEL_DIR", "") or None,
            cpu_threads=_env_int("SCRIPTVAULT_CPU_THREADS", 0),
            max_concurrency=max(1, _env_int("SCRIPTVAULT_MAX_CONCURRENCY", 1)),
            timeout_ms=max(0, _env_int("SCRIPTVAULT_TIMEOUT_MS", 300_000)),
            preload=_env_bool("SCRIPTVAULT_PRELOAD", True),
            preprocess=_env_bool("SCRIPTVAULT_PREPROCESS", True),
            workers=max(0, _env_int("SCRIPTVAULT_WORKERS", 0)),
            use_processes=_env_bool("SCRIPTVAULT_USE_PROCESSES", True),
            max_side_len=max(0, _env_int("SCRIPTVAULT_MAX_SIDE", 0)),
            barcode_enabled=_env_bool("SCRIPTVAULT_BARCODE", True),
            barcode_budget_ms=max(1, _env_int("SCRIPTVAULT_BARCODE_BUDGET_MS", 15)),
            roi_enabled=_env_bool("SCRIPTVAULT_ROI_ENABLED", False),
            roi_profile=_env_roi("SCRIPTVAULT_ROI"),
            storage_root=_env("SCRIPTVAULT_STORAGE_ROOT", "STORAGE"),
            archive_encrypt=_env_bool("SCRIPTVAULT_ARCHIVE_ENCRYPT", False),
            accept_threshold=max(
                1, min(100, _env_int("SCRIPTVAULT_ACCEPT_THRESHOLD", 85))
            ),
            auth_enabled=_env_bool("SCRIPTVAULT_AUTH_ENABLED", False),
            jwt_secret=_env(
                "SCRIPTVAULT_JWT_SECRET", "scriptvault-dev-secret-change-me"
            ),
            jwt_issuer=_env("SCRIPTVAULT_JWT_ISSUER", "scriptvault-ocr"),
            jwt_audience=_env("SCRIPTVAULT_JWT_AUDIENCE", "scriptvault-web"),
            jwt_expire_minutes=max(1, _env_int("SCRIPTVAULT_JWT_EXPIRE_MINUTES", 480)),
            master_key=_env("SCRIPTVAULT_MASTER_KEY", ""),
            db_url=_env("SCRIPTVAULT_DB_URL", "sqlite:///scriptvault.db"),
            db_encrypt=_env_bool("SCRIPTVAULT_DB_ENCRYPT", False),
            db_key=_env("SCRIPTVAULT_DB_KEY", ""),
            max_file_mb=max(1, _env_int("SCRIPTVAULT_MAX_FILE_MB", 25)),
        )

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024
