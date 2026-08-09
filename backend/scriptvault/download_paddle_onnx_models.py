"""Télécharge les modèles ONNX PaddleOCR PP-OCRv5 (détection + reconnaissance) vers ``models/paddle_onnx/``.

Usage:
    python -m scriptvault.download_paddle_onnx_models [--dest chemin]

Sources (Hugging Face, exports ONNX officiels PaddlePaddle) :
    - PaddlePaddle/PP-OCRv5_mobile_det_onnx  → ``PP-OCRv5_mobile_det.onnx``
    - PaddlePaddle/PP-OCRv5_mobile_rec_onnx  → ``PP-OCRv5_mobile_rec.onnx``
      (le fichier ``inference.onnx`` y est renommé ``PP-OCRv5_mobile_det.onnx`` /
      ``PP-OCRv5_mobile_rec.onnx``)

Le dict de caractères du modèle de reconnaissance est récupéré depuis le
``config.json`` paddlex local (``~/.paddlex``) s'il existe, sinon depuis le
fichier ``ppocr_dict.txt`` de PaddleOCR téléchargé à la volée.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DET_REPO = "PaddlePaddle/PP-OCRv5_mobile_det_onnx"
REC_REPO = "PaddlePaddle/PP-OCRv5_mobile_rec_onnx"
BASE_URL = "https://huggingface.co/{repo}/resolve/main"


def default_dest() -> Path:
    """Emplacement par défaut : ``<racine du projet>/models/paddle_onnx``."""
    root = Path(__file__).resolve().parents[2]
    return root / "models" / "paddle_onnx"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "scriptvault-models/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def download_onnx(repo: str, dest: Path, out_name: str) -> None:
    target = dest / out_name
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"  déjà présent : {target.name} ({target.stat().st_size / 1024 / 1024:.1f} Mo) — ignoré")
        return
    print(f"  téléchargement de {out_name} …")
    data = _fetch(f"{BASE_URL.format(repo=repo)}/inference.onnx")
    target.write_bytes(data)
    print(f"    ok — {out_name} ({len(data) / 1024 / 1024:.1f} Mo)")


def ensure_dict(dest: Path) -> None:
    """Récupère le dict de caractères du modèle rec si absent.

    Priorité : 1) ~/.paddlex config.json local, 2) ppocr_dict.txt distante.
    """
    target = dest / "ppocr_dict.txt"
    if target.exists() and target.stat().st_size > 1_000:
        return
    local_config = None
    paddlex_root = Path.home() / ".paddlex"
    for cfg in paddlex_root.glob("official_models/PP-OCRv5_mobile_rec/config.json"):
        local_config = cfg
        break
    lines: list[str] = []
    if local_config is not None:
        import json

        try:
            data = json.loads(local_config.read_text(encoding="utf-8"))
            char_dict = data.get("PostProcess", {}).get("character_dict", "")
            if char_dict:
                lines = char_dict.split("\n")
        except Exception:  # noqa: BLE001
            lines = []
    if not lines:
        try:
            raw = _fetch("https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt")
            lines = io.StringIO(raw.decode("utf-8")).read().splitlines()
        except Exception as exc:  # noqa: BLE001
            print(f"  [avertissement] dict indisponible ({exc}) — vérifiez manuellement models/paddle_onnx/ppocr_dict.txt")
            return
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"    ok — ppocr_dict.txt ({len(lines)} caractères)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest(),
        help="dossier de destination (défaut: models/paddle_onnx dans la racine du projet)",
    )
    args = parser.parse_args(argv)

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement des modèles ONNX PP-OCRv5 dans {dest}")
    download_onnx(DET_REPO, dest, "PP-OCRv5_mobile_det.onnx")
    download_onnx(REC_REPO, dest, "PP-OCRv5_mobile_rec.onnx")
    ensure_dict(dest)
    print("Terminé. Le moteur ONNX (backend `ppocrv5-onnx`) est opérationnel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
