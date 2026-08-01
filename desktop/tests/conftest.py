"""Met le dossier desktop/ sur le sys.path pour les imports locaux
(worker_thread, build_installer) lors des tests exécutés depuis la racine."""

from __future__ import annotations

import sys
from pathlib import Path

DESKTOP_DIR = Path(__file__).resolve().parent.parent
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))
