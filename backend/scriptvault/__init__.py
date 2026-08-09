"""ScriptVault OCR — moteur et API serveur.

Le package ``scriptvault`` contient le moteur HTR local (TrOCR ONNX Runtime +
OpenCV), les services de rasterisation PDF et d'export, l'analyseur de
formulaires (avec correcteur de niveau caractère, sans lexique), ainsi que
l'API HTTP FastAPI qui les expose au client web.

Clients :

* ``web/`` — interface React/Vite qui parle uniquement à l'API REST.
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
