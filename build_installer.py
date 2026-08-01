"""Automatisation de compilation et de packaging de ScriptVault OCR.

Ce script produit un exécutable autonome (sans interpréteur Python requis)
à partir de ``gui_app.py`` via **Nuitka** (recommandé) ou **PyInstaller**,
en incluant automatiquement :

* les sous-dossiers de poids des modèles OCR (``*.pdparams``, ``*.pdmodel``,
  ``*.pdopt``, ``*.onnx``) découverts dans le projet (``models/``,
  ``weights/``, ``engine/``...),
* les DLLs OpenCV et le runtime PaddlePaddle (via les plugins Nuitka ou les
  collecteurs PyInstaller),
* les ressources graphiques (icône ``assets/icon.ico`` sur Windows,
  ``assets/icon.png`` sur Linux) si présentes.

Le mode ``--mode onefile`` (défaut) ou ``--mode onedir`` produit une
distribution **standalone** : le code source Python est compilé en C/C++
(Nuitka) ou embarqué dans l'exécutable (PyInstaller), ce qui masque
l'implémentation et protège la propriété intellectuelle.

-----------------------------------------------------------------------------
Compilation — Windows (PowerShell)
-----------------------------------------------------------------------------

    1. Prérequis (une seule fois) :

        py -3.11 -m venv .venv
        .\\.venv\\Scripts\\Activate.ps1
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install zstandard ordered-set        # recommandé par Nuitka

    2. Placer les poids OCR dans ``models\\det``, ``models\\rec``,
       ``models\\cls`` (ou toute arborescence contenant des ``.pdparams`` /
       ``.onnx`` — découverte automatique). Icône optionnelle :
       ``assets\\icon.ico``.

    3. Lancer la compilation :

        python build_installer.py --tool nuitka --mode onefile
        python build_installer.py --tool pyinstaller --mode onefile

    4. L'artefact est produit dans ``dist\\`` :
       ``dist\\ScriptVaultOCR.exe`` (onefile) ou ``dist\\ScriptVaultOCR\\``
       (onedir, exécutable ``ScriptVaultOCR.exe``).
       Pour utiliser les modèles embarqués en mode hors-ligne :
       ``ScriptVaultOCR.exe --model-dir models``

-----------------------------------------------------------------------------
Compilation — Linux (Bash)
-----------------------------------------------------------------------------

    1. Prérequis (une seule fois) :

        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install zstandard ordered-set
        sudo apt install gcc python3-dev          # compilateur requis par Nuitka

    2. Placer les poids OCR dans ``models/`` (découverte automatique).
       Icône optionnelle : ``assets/icon.png``.

    3. Lancer la compilation :

        python build_installer.py --tool nuitka --mode onefile
        python build_installer.py --tool pyinstaller --mode onefile

    4. Artefact : ``dist/ScriptVaultOCR`` (exécutable ELF autonome).
       Lancement hors-ligne avec modèles embarqués :
       ``./dist/ScriptVaultOCR --model-dir models``

-----------------------------------------------------------------------------
Options du script
-----------------------------------------------------------------------------

    --tool {nuitka,pyinstaller,auto}   Compilateur (défaut: auto → nuitka)
    --mode {onefile,onedir}            Mode de distribution (défaut: onefile)
    --name NAME                        Nom de l'exécutable (défaut: ScriptVaultOCR)
    --entry FILE.py                    Point d'entrée (défaut: gui_app.py)
    --models DIR                       Dossier de poids (découverte auto sinon)
    --icon FILE                        Icône (découverte auto sinon)
    --out-dir DIR                      Dossier de sortie (défaut: dist)
    --skip-models                      Ne pas embarquer les poids OCR
    --zip                              Archive zip finale (release)
    --dry-run                          Affiche la commande sans l'exécuter
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

__version__ = "1.0.0"

DEFAULT_NAME = "ScriptVaultOCR"
DEFAULT_ENTRY = "gui_app.py"

# Marqueurs de fichiers de poids de modèles (PaddleOCR / ONNX)
MODEL_FILE_MARKERS = (".pdparams", ".pdopt", ".pdmodel", ".onnx", ".inference")

# Dossiers candidats pour la découverte automatique des poids
MODEL_DIR_CANDIDATES = ("models", "weights", "engine", "ocr_models")

# Ressources graphiques
ICON_CANDIDATES = ("assets/icon.ico", "icon.ico", "assets/icon.png", "icon.png")

PLATFORM_LABEL = f"{platform.system().lower()}_{platform.machine().lower()}"


class BuildError(RuntimeError):
    """Erreur de compilation ou de configuration du build."""


@dataclass
class BuildConfig:
    """Configuration complète d'un build."""

    tool: str = "auto"
    mode: str = "onefile"
    name: str = DEFAULT_NAME
    entry: Path = Path(DEFAULT_ENTRY)
    models: list[Path] = field(default_factory=list)
    icon: Optional[Path] = None
    out_dir: Path = Path("dist")
    clean: bool = True
    zip_release: bool = False
    dry_run: bool = False


# --------------------------------------------------------------------------- #
# Découverte automatique des ressources
# --------------------------------------------------------------------------- #
def discover_model_dirs(base: Path, max_depth: int = 4) -> list[Path]:
    """Recherche récursivement les dossiers contenant des poids de modèles.

    Un dossier est retenu s'il contient au moins un fichier portant un
    marqueur de poids (``.pdparams``, ``.pdmodel``, ``.pdopt``, ``.onnx``...)
    ou s'il est l'un des candidats connus (``det``, ``rec``, ``cls``) avec
    des fichiers de poids dans sa hiérarchie immédiate.

    Args:
        base: Racine de recherche (répertoire du projet).
        max_depth: Profondeur maximale de descente.

    Returns:
        Dossiers de poids à embarquer (chemin absolu).
    """
    found: list[Path] = []
    base = Path(base).resolve()

    def _has_marker(dirpath: Path) -> bool:
        try:
            return any(
                item.suffix in MODEL_FILE_MARKERS or item.name == "inference.pdmodel"
                for item in dirpath.iterdir()
                if item.is_file()
            )
        except OSError:
            return False

    if not base.is_dir():
        return found

    for root in sorted(base.rglob("*")):
        if not root.is_dir() or root == base:
            continue
        rel_depth = len(root.relative_to(base).parts)
        if rel_depth > max_depth:
            continue
        if _has_marker(root) and root not in found:
            found.append(root)

    # Si seuls des sous-dossiers det/rec/cls ont été trouvés, remonter au
    # dossier racine pour embarquer l'arborescence complète.
    if found:
        common_parent = found[0].parent
        if all(p.parent == common_parent for p in found) and common_parent != base:
            if any(p.name in ("det", "rec", "cls") for p in found):
                found = [common_parent]

    return found


def find_icon() -> Optional[Path]:
    """Cherche une icône dans les emplacements usuels du projet."""
    for candidate in ICON_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def discover_project_resources(cfg: BuildConfig) -> BuildConfig:
    """Complète la configuration (modèles + icône) par découverte auto."""
    project_root = Path.cwd()

    if not cfg.models:
        cfg.models = discover_model_dirs(project_root)
        if not cfg.models:
            print(
                "[AVERTISSEMENT] Aucun dossier de poids de modèles détecté. "
                "Les modèles seront téléchargés au premier lancement — "
                "prévoir --models DIR pour un build 100% hors-ligne."
            )

    if cfg.icon is None:
        cfg.icon = find_icon()
    return cfg


# --------------------------------------------------------------------------- #
# Génération des commandes
# --------------------------------------------------------------------------- #
def _nuitka_command(cfg: BuildConfig) -> list[str]:
    """Construit la ligne de commande Nuitka (compilation en C/C++)."""
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        "--include-package=core_ocr",
        "--include-package=worker_thread",
        "--include-package=gui_app",
        "--include-package=paddle",
        "--include-package=paddleocr",
        "--include-package-data=paddle",
        "--include-package-data=paddleocr",
        "--include-package=cv2",
        "--output-dir",
        str(cfg.out_dir),
        "--output-filename",
        cfg.name,
        "--remove-output",
    ]
    if cfg.mode == "onefile":
        cmd.append("--onefile")
    else:
        cmd.append("--standalone")

    for model_dir in cfg.models:
        src = str(model_dir)
        target = "models"
        if model_dir.name in ("det", "rec", "cls"):
            target = f"models/{model_dir.name}"
        cmd.extend(["--include-data-dir", f"{src}={target}"])

    if cfg.icon is not None:
        if platform.system() == "Windows":
            cmd.extend(["--windows-icon-from-ico", str(cfg.icon)])
        else:
            cmd.extend(["--linux-icon", str(cfg.icon)])

    if platform.system() == "Windows":
        cmd.append("--windows-console-mode=disable")

    cmd.append(str(cfg.entry))
    return cmd


def _pyinstaller_command(cfg: BuildConfig) -> list[str]:
    """Construit la ligne de commande PyInstaller (bundleur classique)."""
    separator = ";" if platform.system() == "Windows" else ":"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        cfg.name,
        "--windowed",
    ]
    cmd.append("--onefile" if cfg.mode == "onefile" else "--onedir")

    # Paddle / PaddleOCR / PyMuPDF : embarque modules + données + DLLs natives
    for package in ("paddle", "paddleocr", "fitz", "PyMuPDF"):
        cmd.extend(["--collect-all", package])

    for model_dir in cfg.models:
        src = str(model_dir)
        target = "models"
        if model_dir.name in ("det", "rec", "cls"):
            target = f"models/{model_dir.name}"
        cmd.extend(["--add-data", f"{src}{separator}{target}"])

    if cfg.icon is not None:
        cmd.extend(["--icon", str(cfg.icon)])

    cmd.append(str(cfg.entry))
    return cmd


def _build_command(cfg: BuildConfig) -> tuple[str, list[str]]:
    """Retourne ``(outil, commande)`` après résolution du mode ``auto``."""
    if cfg.tool == "auto":
        cfg.tool = "nuitka" if importlib.util.find_spec("nuitka") else "pyinstaller"
    # En mode dry-run, l'outil n'a pas besoin d'être installé : on ne fait
    # que planifier la commande.
    if not cfg.dry_run:
        if cfg.tool == "nuitka" and not importlib.util.find_spec("nuitka"):
            raise BuildError("Nuitka n'est pas installé. Exécutez: pip install nuitka")
        if cfg.tool == "pyinstaller" and not importlib.util.find_spec("PyInstaller"):
            raise BuildError(
                "PyInstaller n'est pas installé. Exécutez: pip install pyinstaller"
            )

    if cfg.tool == "nuitka":
        return "nuitka", _nuitka_command(cfg)
    return "pyinstaller", _pyinstaller_command(cfg)


# --------------------------------------------------------------------------- #
# Exécution
# --------------------------------------------------------------------------- #
def _run(command: Sequence[str], dry_run: bool) -> int:
    """Exécute la commande en diffusant la sortie en temps réel."""
    if dry_run:
        print("\n>>> " + " ".join(command))
        return 0
    try:
        process = subprocess.run(command, check=False)
    except OSError as exc:
        raise BuildError(f"Exécution impossible: {exc}") from exc
    return process.returncode


def _prepare_output(cfg: BuildConfig) -> None:
    """Prépare le dossier de sortie et nettoie les artefacts antérieurs."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.clean:
        for stale in cfg.out_dir.glob(f"{cfg.name}*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)
    if cfg.mode == "onedir":
        # PyInstaller crée dist/name ; Nuitka crée dist/name.dist
        for pattern in (f"{cfg.name}", f"{cfg.name}.dist", f"{cfg.name}.onefile"):
            stale = cfg.out_dir / pattern
            if stale.exists():
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink(missing_ok=True)


def _locate_artifact(cfg: BuildConfig) -> Path:
    """Localise l'exécutable produit après le build."""
    if cfg.mode == "onefile":
        candidate = cfg.out_dir / (cfg.name + (".exe" if os.name == "nt" else ""))
        if candidate.exists():
            return candidate
    base = cfg.out_dir / cfg.name
    if base.is_dir():
        executable = base / (cfg.name + (".exe" if os.name == "nt" else ""))
        if executable.exists():
            return executable
    # Nuitka standalone crée un sous-dossier .dist
    nuitka_dir = cfg.out_dir / f"{cfg.name}.dist"
    if nuitka_dir.is_dir():
        executable = nuitka_dir / (cfg.name + (".exe" if os.name == "nt" else ""))
        if executable.exists():
            return executable
    raise BuildError(
        "Artefact introuvable — vérifiez la sortie de la compilation "
        f"dans {cfg.out_dir.resolve()}"
    )


def _zip_release(cfg: BuildConfig, artifact: Path) -> Path:
    """Crée une archive zip de la distribution (release)."""
    archive_base = cfg.out_dir / f"{cfg.name}-{PLATFORM_LABEL}-{cfg.mode}"
    if artifact.is_dir():
        shutil.make_archive(
            str(archive_base), "zip", root_dir=artifact.parent, base_dir=artifact.name
        )
    else:
        shutil.make_archive(
            str(archive_base), "zip", root_dir=artifact.parent, base_dir=artifact.name
        )
    return Path(f"{archive_base}.zip")


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #
def build(cfg: BuildConfig) -> int:
    """Exécute le pipeline complet de compilation.

    Args:
        cfg: Configuration du build.

    Returns:
        Code de retour du processus de compilation (0 = succès).

    Raises:
        BuildError: Entrée invalide ou outil de compilation indisponible.
    """
    if not cfg.entry.is_file():
        raise BuildError(
            f"Point d'entrée introuvable: {cfg.entry} — "
            "lancez le script depuis la racine du projet."
        )

    discover_project_resources(cfg)
    tool, command = _build_command(cfg)

    print("=" * 68)
    print(f"  ScriptVault OCR — Build ({tool}, mode {cfg.mode})")
    print(f"  Plateforme : {PLATFORM_LABEL} | Python {sys.version.split()[0]}")
    print(f"  Entrée     : {cfg.entry}")
    print(f"  Modèles    : {len(cfg.models)} dossier(s) détecté(s)")
    for model_dir in cfg.models:
        print(f"    - {model_dir}")
    print(f"  Icône      : {cfg.icon or '(aucune)'}")
    print(f"  Sortie     : {cfg.out_dir.resolve()}")
    print("=" * 68)

    _prepare_output(cfg)
    code = _run(command, cfg.dry_run)
    if code != 0:
        return code

    if not cfg.dry_run:
        artifact = _locate_artifact(cfg)
        print(f"\n[OK] Artefact produit : {artifact}")
        if cfg.zip_release:
            archive = _zip_release(cfg, artifact)
            print(f"[OK] Archive release  : {archive}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> BuildConfig:
    """Analyse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Compile ScriptVault OCR en exécutable autonome "
        "(Nuitka recommandé, PyInstaller en alternative).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tool",
        choices=("nuitka", "pyinstaller", "auto"),
        default="auto",
        help="Compilateur à utiliser (auto = nuitka si disponible).",
    )
    parser.add_argument(
        "--mode",
        choices=("onefile", "onedir"),
        default="onefile",
        help="onefile = exécutable unique ; onedir = dossier distribué.",
    )
    parser.add_argument("--name", default=DEFAULT_NAME, help="Nom de l'exécutable.")
    parser.add_argument("--entry", default=DEFAULT_ENTRY, help="Point d'entrée Python.")
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        help="Dossier de poids de modèles (répétable). Découverte auto sinon.",
    )
    parser.add_argument(
        "--icon", default=None, help="Icône de l'application (découverte auto sinon)."
    )
    parser.add_argument("--out-dir", default="dist", help="Dossier de sortie.")
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Ne pas embarquer les poids des modèles OCR.",
    )
    parser.add_argument("--zip", action="store_true", help="Créer l'archive release.")
    parser.add_argument("--no-clean", action="store_true", help="Ne pas purger dist/.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Afficher la commande sans l'exécuter."
    )
    args = parser.parse_args(argv)

    cfg = BuildConfig(
        tool=args.tool,
        mode=args.mode,
        name=args.name,
        entry=Path(args.entry),
        icon=Path(args.icon) if args.icon else None,
        out_dir=Path(args.out_dir),
        clean=not args.no_clean,
        zip_release=args.zip,
        dry_run=args.dry_run,
    )
    if not args.skip_models:
        if args.models:
            cfg.models = [Path(p).resolve() for p in args.models]
            for model_dir in cfg.models:
                if not model_dir.is_dir():
                    raise BuildError(f"Dossier de modèles introuvable: {model_dir}")
        else:
            cfg.models = discover_model_dirs(Path.cwd())
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée CLI: retourne le code de sortie du build."""
    try:
        cfg = parse_args(argv)
        return build(cfg)
    except BuildError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nBuild interrompu.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
