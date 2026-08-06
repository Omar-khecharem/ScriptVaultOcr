"""Point d'entrée du serveur ScriptVault OCR.

Usage::

    python main.py [--host 0.0.0.0] [--port 8000] [--reload]

Toutes les options peuvent aussi être passées en variables d'environnement
``SCRIPTVAULT_*`` (voir :mod:`scriptvault.config`).
"""

from __future__ import annotations

import argparse
import os
import sys

# Rend le package `scriptvault` importable quel que soit le répertoire d'appel
# (racine du dépôt ou backend/), sans dépendre d'une installation -e.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serveur OCR local ScriptVault (FastAPI + PaddleOCR CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", default=os.environ.get("SCRIPTVAULT_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SCRIPTVAULT_PORT", "8000"))
    )
    parser.add_argument(
        "--reload", action="store_true", help="Rechargement automatique (dev)."
    )
    parser.add_argument(
        "--lang",
        default=os.environ.get("SCRIPTVAULT_LANG", ""),
        help="Langue OCR par défaut.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.lang:
        os.environ["SCRIPTVAULT_LANG"] = args.lang

    import uvicorn

    uvicorn.run(
        "scriptvault.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
