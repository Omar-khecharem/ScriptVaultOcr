"""ScriptVault OCR — moteur et API serveur (partagés avec le desktop).

Le package ``scriptvault`` contient le moteur OCR local (PaddleOCR CPU +
OpenCV), les services de rasterisation PDF et d'export, ainsi que l'API
HTTP FastAPI qui les expose au client web.

Il est consommé par deux clients distincts:

* ``desktop/``  — application PySide6 (``pip install -e backend`` puis
  ``pip install -r desktop/requirements.txt``) ;
* ``web/``      — interface React/Vite qui parle uniquement à l'API REST.
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = [
    "core_ocr",
    "config",
    "engines",
    "pdf",
    "exporter",
    "schemas",
    "form_analyzer",
    "api",
    "__version__",
]
