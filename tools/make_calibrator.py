"""Génère tools/roi_calibrator_ready.html : template + scan encodé en base64 (JPEG).

Usage:
    python tools/make_calibrator.py <scan.tif> [--max-width 1400] [--quality 85]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "tools" / "roi_calibrator.html"
OUT = ROOT / "tools" / "roi_calibrator_ready.html"
PLACEHOLDER = "PAYLOAD_B64"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    img = cv2.imread(str(args.scan), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"impossible de lire {args.scan} (cv2)")
    h, w = img.shape[:2]
    if w > args.max_width:
        scale = args.max_width / w
        img = cv2.resize(img, (args.max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
    if not ok:
        raise SystemExit("échec de l'encodage JPEG")
    b64 = base64.b64encode(buf).decode("ascii")

    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise SystemExit(f"placeholder '{PLACEHOLDER}' introuvable dans {TEMPLATE}")
    html = html.replace(PLACEHOLDER, b64)

    OUT = ROOT / "tools" / "roi_calibrator_ready.html"
    OUT.write_text(html, encoding="utf-8")
    print(f"OK → {OUT}  ({OUT.stat().st_size / 1024:.0f} Ko, image {img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()