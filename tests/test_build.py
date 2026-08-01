"""Tests unitaires de l'automatisation de build (Nuitka / PyInstaller)."""

from pathlib import Path

import pytest

import build_installer as bi


def test_version():
    assert bi.__version__ == "1.0.0"


def test_discover_model_dirs_consolidates_det_rec(tmp_path: Path):
    root = tmp_path / "models"
    (root / "det").mkdir(parents=True)
    (root / "rec").mkdir()
    (root / "det" / "inference.pdmodel").write_bytes(b"\x00")
    (root / "rec" / "inference.pdmodel").write_bytes(b"\x00")

    found = bi.discover_model_dirs(tmp_path)

    assert len(found) == 1
    assert found[0].name == "models"  # consolidation det/rec/cls -> racine


def test_discover_model_dirs_ignores_empty(tmp_path: Path):
    (tmp_path / "models" / "det").mkdir(parents=True)
    assert bi.discover_model_dirs(tmp_path) == []


def test_discover_model_dirs_respects_max_depth(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "weights"
    deep.mkdir(parents=True)
    (deep / "model.onnx").write_bytes(b"\x00")
    found = bi.discover_model_dirs(tmp_path, max_depth=2)
    assert found == []


def test_find_icon_returns_none_without_assets():
    assert bi.find_icon() is None


def test_nuitka_command_onefile():
    cfg = bi.BuildConfig(tool="nuitka", mode="onefile", dry_run=True)
    tool, cmd = bi._build_command(cfg)
    assert tool == "nuitka"
    assert cmd[0] == "python" or Path(cmd[0]).name in ("python", "python.exe")
    assert "--onefile" in cmd
    assert "--enable-plugin=pyside6" in cmd
    assert "--include-package=paddle" in cmd
    assert cmd[-1] == "gui_app.py"


def test_nuitka_command_onedir_uses_standalone():
    cfg = bi.BuildConfig(tool="nuitka", mode="onedir", dry_run=True)
    _, cmd = bi._build_command(cfg)
    assert "--standalone" in cmd


def test_pyinstaller_command_collects_packages():
    cfg = bi.BuildConfig(tool="pyinstaller", mode="onefile", dry_run=True)
    tool, cmd = bi._build_command(cfg)
    assert tool == "pyinstaller"
    assert "--onefile" in cmd
    assert "--collect-all" in cmd
    assert cmd[-1] == "gui_app.py"


def test_parse_args_defaults():
    cfg = bi.parse_args([])
    assert cfg.tool == "auto"
    assert cfg.mode == "onefile"
    assert cfg.name == "ScriptVaultOCR"
    assert not cfg.dry_run


def test_parse_args_full(tmp_path: Path):
    weights = tmp_path / "models" / "det"
    weights.mkdir(parents=True)
    (weights / "inference.pdmodel").write_bytes(b"\x00")

    cfg = bi.parse_args(
        [
            "--tool",
            "nuitka",
            "--mode",
            "onedir",
            "--name",
            "SVOCR",
            "--models",
            str(weights),
            "--zip",
            "--dry-run",
        ]
    )
    assert cfg.tool == "nuitka"
    assert cfg.mode == "onedir"
    assert cfg.name == "SVOCR"
    assert cfg.models == [weights.resolve()]
    assert cfg.zip_release
    assert cfg.dry_run


def test_parse_args_missing_models_dir_raises(tmp_path: Path):
    with pytest.raises(bi.BuildError):
        bi.parse_args(["--models", str(tmp_path / "absent")])
