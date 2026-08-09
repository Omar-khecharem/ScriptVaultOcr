"""Télécharge les modèles ONNX TrOCR (100 % local ensuite) vers ``models/trocr/``.

Usage:
    python -m scriptvault.download_trocr_models [--dest chemin]

La source est ``Xenova/trocr-small-handwritten`` (export ONNX + tokenizer,
hors-ligne par la suite). Aucune dépendance au-delà de la bibliothèque
standard (les fichiers sont récupérés avec urllib).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "Xenova/trocr-small-handwritten"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/main"

FILES = (
    "onnx/encoder_model_quantized.onnx",
    "onnx/decoder_model_merged_quantized.onnx",
    "tokenizer.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
)


def default_dest() -> Path:
    """Emplacement par défaut : ``<racine du projet>/models/trocr``."""
    root = Path(__file__).resolve().parents[2]
    return root / "models" / "trocr"


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "scriptvault-models/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, tmp.open("wb") as out:
        size = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(chunk)
            if not block:
                break
            out.write(block)
            done += len(block)
            if size:
                print(f"    {done / 1024 / 1024:.1f} / {size / 1024 / 1024:.1f} Mo", end="\r")
    tmp.replace(dest)
    print(f"    ok — {dest.name} ({done / 1024 / 1024:.1f} Mo)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest(),
        help="dossier de destination des modèles (défaut: models/trocr dans la racine du projet)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="noms de fichiers à télécharger (défaut: tous)",
    )
    args = parser.parse_args(argv)

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPOSITORY} into {dest}")
    for name in FILES:
        if args.only and name not in args.only:
            continue
        target = dest / Path(name).name
        if target.exists() and target.stat().st_size > 1_000:
            print(f"  déjà présent : {target.name} — ignoré (utilisez --only pour forcer)")
            continue
        download_file(f"{BASE_URL}/{name}", target)
    print("Terminé. Démarrez le serveur avec SCRIPTVAULT_OCR_BACKEND=onnx.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
