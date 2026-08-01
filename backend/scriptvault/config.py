"""Configuration du serveur ScriptVault OCR, pilotée par variables d'environnement.

Toutes les options exposent un préfixe ``SCRIPTVAULT_`` et peuvent être
surchargées par la ligne de commande (``python main.py --port 9000``).

Exemple::

    SCRIPTVAULT_PORT=9000 SCRIPTVAULT_LANG=fr python main.py
"""

from __future__ import annotations

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
    timeout_ms: int = 120_000  # délai maximal d'une image (0 = illimité)
    preload: bool = True  # pré-charge le moteur au démarrage
    preprocess: bool = True  # pipeline OpenCV par défaut (CLAHE/deskew/binarize)

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
            timeout_ms=max(0, _env_int("SCRIPTVAULT_TIMEOUT_MS", 120_000)),
            preload=_env_bool("SCRIPTVAULT_PRELOAD", True),
            preprocess=_env_bool("SCRIPTVAULT_PREPROCESS", True),
            max_file_mb=max(1, _env_int("SCRIPTVAULT_MAX_FILE_MB", 25)),
        )

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024
