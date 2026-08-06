"""Application desktop OCR "ScriptVault OCR" (PySide6 + core_ocr + worker_thread).

Interface commerciale moderne (thème sombre/clair), entièrement locale,
avec :

* Zone de drag & drop de fichiers (PNG, JPG, TIFF, WebP, PDF).
* Vue scindée : à gauche, visualiseur d'image avec superposition des boîtes
  englobantes (couleur = confiance) ; à droite, éditeur de texte enrichi pour
  corriger le résultat OCR.
* Barre de progression du lot, indicateur de confiance global (jauge
  circulaire) et temps d'exécution en millisecondes.
* Export des textes corrigés en ``.txt``, ``.docx`` (python-docx) et ``.pdf``
  avec couche texte sélectionnable (ReportLab — texte vectoriel).
* OCR exécuté en arrière-plan via :class:`worker_thread.BatchWorker`
  (QThread + signaux) : l'interface ne se bloque jamais.
* Les fichiers PDF sont rastérisés page par page (PyMuPDF) puis traités comme
  des images, chaque page apparaissant comme une tâche distincte.

Lancement::

    python gui_app.py [--lang fr] [--threads 4] [--model-dir D:\\models]

Dépendances: PySide6, core_ocr (paddleocr + paddlepaddle), worker_thread,
python-docx, reportlab, PyMuPDF (pour les PDF), numpy, opencv-python.
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from scriptvault.core_ocr import ImagePreprocessor, LocalOCREngine
from worker_thread import BatchWorker

__version__ = "1.0.0"

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"})
_PDF_EXTS = frozenset({".pdf"})
_SUPPORTED_EXTS = _IMAGE_EXTS | _PDF_EXTS


# --------------------------------------------------------------------------- #
# Thème (sombre / clair)
# --------------------------------------------------------------------------- #
class Theme:
    """Palettes de couleurs et stylesheet QSS pour les deux thèmes."""

    DARK = {
        "window": "#141519",
        "surface": "#1D1F26",
        "surface_alt": "#252833",
        "border": "#33363F",
        "text": "#E8EAF0",
        "muted": "#9BA1B0",
        "accent": "#4C7DFF",
        "accent_hover": "#6590FF",
        "accent_pressed": "#3B66D6",
        "danger": "#E5484D",
        "success": "#30A46C",
        "warning": "#F5A524",
        "input_bg": "#181A21",
        "selection": "#2A4370",
        "shadow": "#000000",
    }
    LIGHT = {
        "window": "#F2F4F8",
        "surface": "#FFFFFF",
        "surface_alt": "#F4F6FA",
        "border": "#DCE1EA",
        "text": "#1A1D27",
        "muted": "#667085",
        "accent": "#4C7DFF",
        "accent_hover": "#3B6AE8",
        "accent_pressed": "#2F58C4",
        "danger": "#D92D20",
        "success": "#12B76A",
        "warning": "#DC6803",
        "input_bg": "#FFFFFF",
        "selection": "#D6E4FF",
        "shadow": "#00000040",
    }

    @staticmethod
    def stylesheet(colors: dict[str, str]) -> str:
        """Construit le stylesheet QSS global à partir d'une palette."""
        c = colors
        return f"""
        QMainWindow, QWidget#root {{ background: {c["window"]}; color: {c["text"]}; }}
        QLabel {{ color: {c["text"]}; background: transparent; }}
        QLabel#muted {{ color: {c["muted"]}; }}
        QLabel#title {{ font-size: 15px; font-weight: 700; }}
        QFrame#card {{ background: {c["surface"]}; border: 1px solid {c["border"]};
                      border-radius: 12px; }}
        QPushButton {{ background: {c["surface_alt"]}; color: {c["text"]};
                       border: 1px solid {c["border"]}; border-radius: 8px;
                       padding: 7px 16px; font-size: 13px; }}
        QPushButton:hover {{ background: {c["border"]}; }}
        QPushButton:pressed {{ background: {c["surface_alt"]}; }}
        QPushButton:disabled {{ color: {c["muted"]}; background: {c["surface_alt"]};
                               border-color: {c["border"]}; }}
        QPushButton#primary {{ background: {c["accent"]}; color: #FFFFFF;
                              border: none; font-weight: 600; }}
        QPushButton#primary:hover {{ background: {c["accent_hover"]}; }}
        QPushButton#primary:pressed {{ background: {c["accent_pressed"]}; }}
        QPushButton#danger {{ color: {c["danger"]}; }}
        QToolButton {{ background: transparent; color: {c["text"]};
                       border: 1px solid transparent; border-radius: 6px;
                       padding: 4px 10px; font-weight: 600; }}
        QToolButton:hover {{ background: {c["surface_alt"]}; }}
        QToolButton:checked {{ background: {c["selection"]}; color: {c["accent"]};
                              border-color: {c["accent"]}; }}
        QComboBox {{ background: {c["surface_alt"]}; color: {c["text"]};
                     border: 1px solid {c["border"]}; border-radius: 8px;
                     padding: 4px 10px; }}
        QComboBox QAbstractItemView {{ background: {c["surface"]}; color: {c["text"]};
                                       selection-background-color: {c["selection"]};
                                       border: 1px solid {c["border"]}; }}
        QListWidget {{ background: {c["surface"]}; border: 1px solid {c["border"]};
                       border-radius: 10px; padding: 4px; outline: none; }}
        QListWidget::item {{ padding: 7px 10px; border-radius: 6px; color: {c["text"]}; }}
        QListWidget::item:hover {{ background: {c["surface_alt"]}; }}
        QListWidget::item:selected {{ background: {c["selection"]}; }}
        QTextEdit {{ background: {c["surface"]}; border: 1px solid {c["border"]};
                     border-radius: 10px; padding: 10px; color: {c["text"]};
                     selection-background-color: {c["accent"]};
                     selection-color: #FFFFFF; font-size: 14px; }}
        QProgressBar {{ background: {c["surface_alt"]}; border: none;
                        border-radius: 5px; height: 10px; }}
        QProgressBar::chunk {{ background: {c["accent"]}; border-radius: 5px; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {c["border"]};
                                       border-radius: 5px; min-height: 30px; }}
        QScrollBar::handle:vertical:hover {{ background: {c["muted"]}; }}
        QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {c["border"]};
                                        border-radius: 5px; min-width: 30px; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QSplitter::handle {{ background: transparent; }}
        QSplitter::handle:horizontal {{ width: 2px; }}
        QSplitter::handle:vertical {{ height: 2px; }}
        QMessageBox {{ background: {c["surface"]}; }}
        QToolTip {{ background: {c["surface_alt"]}; color: {c["text"]};
                    border: 1px solid {c["border"]}; padding: 4px; }}
        """

    @staticmethod
    def box_color(confidence: float) -> str:
        """Couleur de boîte selon la confiance (vert/ambre/rouge)."""
        if confidence >= 0.85:
            return "#2ECC71"
        if confidence >= 0.60:
            return "#F5A524"
        return "#E5484D"


# --------------------------------------------------------------------------- #
# Zone de drag & drop
# --------------------------------------------------------------------------- #
class DropZone(QFrame):
    """Zone cliquable acceptant le glisser-déposer de fichiers."""

    files_dropped = Signal(list)

    def __init__(
        self, colors: dict[str, str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._colors = colors
        self._hover = False
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(220)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)

        icon = QLabel("＋")
        icon.setStyleSheet(
            f"color: {colors['muted']}; font-size: 42px; font-weight: 200;"
        )
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Glissez-déposez vos fichiers ici")
        title.setStyleSheet(
            f"color: {colors['text']}; font-size: 15px; font-weight: 600;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("PNG · JPG · TIFF · WebP · PDF  —  ou cliquez pour parcourir")
        hint.setStyleSheet(f"color: {colors['muted']}; font-size: 12px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch()

    def mousePressEvent(self, event: Any) -> None:
        self.browse()

    def browse(self) -> None:
        """Ouvre la boîte de dialogue de sélection de fichiers."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Ajouter des fichiers",
            "",
            "Documents et images (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.bmp *.pdf)",
        )
        if paths:
            self.files_dropped.emit(paths)

    def dragEnterEvent(self, event: Any) -> None:
        if self._has_supported_urls(event):
            self._hover = True
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: Any) -> None:
        self._hover = False
        self.update()

    def dropEvent(self, event: Any) -> None:
        self._hover = False
        self.update()
        paths = [
            url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()
        ]
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()

    @staticmethod
    def _has_supported_urls(event: Any) -> bool:
        if not event.mimeData().hasUrls():
            return False
        for url in event.mimeData().urls():
            if url.isLocalFile():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in _SUPPORTED_EXTS:
                    return True
        return False

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self._colors["accent"] if self._hover else self._colors["border"]
        pen = QPen(QColor(color), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(6, 6, -6, -6), 14, 14)


# --------------------------------------------------------------------------- #
# Visualiseur d'image avec overlay des boîtes OCR
# --------------------------------------------------------------------------- #
class ImageCanvas(QWidget):
    """Visualiseur d'image avec zoom, déplacement et overlay de boîtes."""

    def __init__(
        self, colors: dict[str, str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._colors = colors
        self._image: Optional[QImage] = None
        self._overlay: list[tuple[list[list[float]], str, float]] = []
        self._scale = 1.0
        self._fit_scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._pan_start: Optional[QPointF] = None
        self._pan_offset0 = QPointF(0.0, 0.0)
        self._message = "Sélectionnez un fichier pour afficher le document"
        self.setMinimumSize(240, 240)

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #
    def set_image(self, bgr: np.ndarray) -> None:
        """Affiche une image BGR (numpy) et réinitialise la vue."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self._image = qimage
        self._overlay = []
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._message = ""
        self.update()

    def set_overlay(
        self, boxes: Sequence[tuple[list[list[float]], str, float]]
    ) -> None:
        """Définit les boîtes à superposer (points, texte, confiance)."""
        self._overlay = list(boxes)
        self.update()

    def clear_overlay(self) -> None:
        self._overlay = []
        self.update()

    def set_message(self, message: str) -> None:
        self._message = message
        self.update()

    # ------------------------------------------------------------------ #
    # Interactions
    # ------------------------------------------------------------------ #
    def wheelEvent(self, event: Any) -> None:
        if self._image is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 ** (delta / 120.0)
        old_px = self._fit_scale * self._scale
        self._scale = min(10.0, max(0.25, self._scale * factor))
        new_px = self._fit_scale * self._scale
        ratio = new_px / old_px if old_px > 0 else 1.0
        pos = event.position()
        self._offset = QPointF(
            pos.x() - (pos.x() - self._offset.x()) * ratio,
            pos.y() - (pos.y() - self._offset.y()) * ratio,
        )
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        if self._image is not None and event.button() == Qt.MouseButton.LeftButton:
            self._pan_start = event.position()
            self._pan_offset0 = self._offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._pan_start is not None:
            self._offset = self._pan_offset0 + (event.position() - self._pan_start)
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        self._pan_start = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------ #
    # Rendu
    # ------------------------------------------------------------------ #
    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self._colors["window"]))

        if self._image is None:
            painter.setPen(QColor(self._colors["muted"]))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._message,
            )
            return

        margin = 8
        avail_w = max(1, self.width() - 2 * margin)
        avail_h = max(1, self.height() - 2 * margin)
        img_w = self._image.width()
        img_h = self._image.height()
        self._fit_scale = min(avail_w / img_w, avail_h / img_h)
        scale = self._fit_scale * self._scale
        draw_w = img_w * scale
        draw_h = img_h * scale
        x0 = (self.width() - draw_w) / 2.0 + self._offset.x()
        y0 = (self.height() - draw_h) / 2.0 + self._offset.y()
        target = QRectF(x0, y0, draw_w, draw_h)
        painter.drawImage(target, self._image)

        for box, text, confidence in self._overlay:
            if len(box) < 3:
                continue
            color = QColor(Theme.box_color(confidence))
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            polygon = QPolygonF(
                [QPointF(x0 + px * scale, y0 + py * scale) for px, py in box]
            )
            painter.drawPolygon(polygon)

            if text and self._scale >= 0.5:
                font = QFont("Segoe UI", 8)
                painter.setFont(font)
                metrics = QFontMetrics(font)
                text_w = metrics.horizontalAdvance(text)
                text_h = metrics.height()
                bx, by = box[0][0] * scale + x0, box[0][1] * scale + y0
                label_rect = QRectF(bx, by - text_h - 6, text_w + 10, text_h + 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 150))
                painter.drawRoundedRect(label_rect, 3, 3)
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(
                    QPointF(bx + 5, by - 5),
                    text,
                )


# --------------------------------------------------------------------------- #
# Jauge de confiance circulaire
# --------------------------------------------------------------------------- #
class ConfidenceGauge(QWidget):
    """Jauge circulaire affichant la confiance globale (%)."""

    def __init__(
        self, colors: dict[str, str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._colors = colors
        self._value: Optional[float] = None
        self.setFixedSize(84, 84)

    def set_value(self, value: Optional[float]) -> None:
        self._value = value
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(6, 6, 72, 72)
        track = QPen(QColor(self._colors["surface_alt"]), 7)
        track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track)
        painter.drawArc(rect, 135 * 16, 270 * 16)

        if self._value is not None:
            color = (
                self._colors["success"]
                if self._value >= 70
                else self._colors["warning"]
                if self._value >= 50
                else self._colors["danger"]
            )
            arc = QPen(QColor(color), 7)
            arc.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(arc)
            painter.drawArc(rect, 135 * 16, int(270 * 16 * self._value / 100.0))

        painter.setPen(QColor(self._colors["text"]))
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        text = "—" if self._value is None else f"{self._value:.0f}%"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


# --------------------------------------------------------------------------- #
# Éditeur de texte enrichi
# --------------------------------------------------------------------------- #
class EditorPanel(QWidget):
    """Éditeur de texte enrichi (B/I/U, taille de police, compteur)."""

    def __init__(
        self, colors: dict[str, str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._colors = colors
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        self._btn_bold = QToolButton()
        self._btn_bold.setText("B")
        self._btn_bold.setCheckable(True)
        self._btn_bold.setToolTip("Gras (Ctrl+B)")
        self._btn_italic = QToolButton()
        self._btn_italic.setText("I")
        self._btn_italic.setCheckable(True)
        self._btn_italic.setToolTip("Italique (Ctrl+I)")
        self._btn_underline = QToolButton()
        self._btn_underline.setText("U")
        self._btn_underline.setCheckable(True)
        self._btn_underline.setToolTip("Souligné (Ctrl+U)")
        self._font_size = QComboBox()
        self._font_size.addItems([str(s) for s in (10, 11, 12, 14, 16, 18, 20, 24, 28)])
        self._font_size.setCurrentText("12")
        self._font_size.setFixedWidth(64)
        self._count_label = QLabel("0 caractères")
        self._count_label.setObjectName("muted")
        self._clear_btn = QToolButton()
        self._clear_btn.setText("Effacer")
        toolbar.addWidget(self._btn_bold)
        toolbar.addWidget(self._btn_italic)
        toolbar.addWidget(self._btn_underline)
        toolbar.addWidget(self._font_size)
        toolbar.addSpacing(12)
        toolbar.addWidget(self._count_label)
        toolbar.addStretch()
        toolbar.addWidget(self._clear_btn)
        layout.addLayout(toolbar)

        self._editor = QTextEdit()
        self._editor.setPlaceholderText(
            "Le texte reconnu s'affichera ici. Vous pouvez le corriger avant "
            "l'exportation."
        )
        layout.addWidget(self._editor, 1)

        self._btn_bold.clicked.connect(self._toggle_bold)
        self._btn_italic.clicked.connect(self._toggle_italic)
        self._btn_underline.clicked.connect(self._toggle_underline)
        self._font_size.currentTextChanged.connect(self._apply_font_size)
        self._clear_btn.clicked.connect(self._editor.clear)
        self._editor.textChanged.connect(self._update_count)
        self._editor.currentCharFormatChanged.connect(self._sync_format_buttons)

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def set_text(self, text: str) -> None:
        self._editor.setPlainText(text)
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self._editor.setTextCursor(cursor)
        self._update_count()

    def text(self) -> str:
        return self._editor.toPlainText()

    def widget(self) -> QTextEdit:
        return self._editor

    # ------------------------------------------------------------------ #
    def _toggle_bold(self, checked: bool) -> None:
        fmt = QTextCharFormat()
        fmt.setFontWeight(QFont.Weight.Bold if checked else QFont.Weight.Normal)
        self._editor.mergeCurrentCharFormat(fmt)

    def _toggle_italic(self, checked: bool) -> None:
        fmt = QTextCharFormat()
        fmt.setFontItalic(checked)
        self._editor.mergeCurrentCharFormat(fmt)

    def _toggle_underline(self, checked: bool) -> None:
        fmt = QTextCharFormat()
        fmt.setFontUnderline(checked)
        self._editor.mergeCurrentCharFormat(fmt)

    def _apply_font_size(self, size_text: str) -> None:
        try:
            size = float(size_text)
        except ValueError:
            return
        fmt = QTextCharFormat()
        fmt.setFontPointSize(size)
        self._editor.mergeCurrentCharFormat(fmt)

    def _sync_format_buttons(self, fmt: QTextCharFormat) -> None:
        for btn, test in (
            (self._btn_bold, lambda f: f.fontWeight() == QFont.Weight.Bold),
            (self._btn_italic, lambda f: f.fontItalic()),
            (self._btn_underline, lambda f: f.fontUnderline()),
        ):
            btn.blockSignals(True)
            btn.setChecked(bool(test(fmt)))
            btn.blockSignals(False)

    def _update_count(self) -> None:
        n = len(self._editor.toPlainText())
        self._count_label.setText(f"{n} caractère{'s' if n != 1 else ''}")


# --------------------------------------------------------------------------- #
# Chargement asynchrone du moteur OCR
# --------------------------------------------------------------------------- #
class _EngineLoader(QThread):
    """Construit le moteur OCR dans un thread dédié (chargement des poids)."""

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(
        self, factory: Callable[[], Any], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._factory = factory

    def run(self) -> None:
        try:
            engine = self._factory()
        except Exception as exc:
            self.failed.emit(
                f"Initialisation du moteur OCR impossible: {type(exc).__name__}: {exc}"
            )
        else:
            self.loaded.emit(engine)


# --------------------------------------------------------------------------- #
# Fenêtre principale
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    """Fenêtre principale de l'application ScriptVault OCR."""

    def __init__(
        self,
        engine_factory: Optional[Callable[[], Any]] = None,
        lang: str = "en",
        model_dir: Optional[str] = None,
        cpu_threads: int = 0,
        batch_size: int = 2,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._engine_factory = engine_factory or (
            lambda: LocalOCREngine(
                lang=lang, model_dir=model_dir, cpu_threads=cpu_threads
            )
        )
        self._batch_size = max(1, int(batch_size))
        self._theme = "dark"
        self._colors = Theme.DARK

        self._engine: Any = None
        self._engine_loader: Optional[_EngineLoader] = None
        self._worker: Optional[BatchWorker] = None
        self._start_pending = False

        self._tasks: list[dict[str, Any]] = []
        self._results: dict[str, dict[str, Any]] = {}
        self._current_path: Optional[str] = None
        self._total_ms = 0.0
        self._confidences: list[float] = []
        self._tempdir = tempfile.TemporaryDirectory(prefix="scriptvault_")

        self._preprocessor = ImagePreprocessor()

        self.setWindowTitle("ScriptVault OCR")
        self.resize(1280, 820)
        self.setMinimumSize(1024, 640)

        self._build_ui()
        self._apply_theme()

        QShortcut(QKeySequence.StandardKey.Open, self, self._browse_files)

    # ------------------------------------------------------------------ #
    # Construction de l'interface
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_top_bar())
        root_layout.addWidget(self._build_central(), 1)
        root_layout.addWidget(self._build_bottom_bar())

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(58)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(8)

        title = QLabel("ScriptVault OCR")
        title.setObjectName("title")

        self._btn_add = QPushButton("Ajouter des fichiers")
        self._btn_start = QPushButton("Démarrer l'OCR")
        self._btn_start.setObjectName("primary")
        self._btn_cancel = QPushButton("Annuler")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False)

        self._btn_export_txt = QPushButton("Exporter TXT")
        self._btn_export_docx = QPushButton("Exporter DOCX")
        self._btn_export_pdf = QPushButton("Exporter PDF")
        for btn in (self._btn_export_txt, self._btn_export_docx, self._btn_export_pdf):
            btn.setEnabled(False)

        self._btn_theme = QPushButton("Thème clair")

        self._btn_add.clicked.connect(self._browse_files)
        self._btn_start.clicked.connect(self._start_ocr)
        self._btn_cancel.clicked.connect(self._cancel_ocr)
        self._btn_export_txt.clicked.connect(self._export_txt)
        self._btn_export_docx.clicked.connect(self._export_docx)
        self._btn_export_pdf.clicked.connect(self._export_pdf)
        self._btn_theme.clicked.connect(self._toggle_theme)

        layout.addWidget(title)
        layout.addSpacing(16)
        layout.addWidget(self._btn_add)
        layout.addWidget(self._btn_start)
        layout.addWidget(self._btn_cancel)
        layout.addStretch()
        layout.addWidget(self._btn_export_txt)
        layout.addWidget(self._btn_export_docx)
        layout.addWidget(self._btn_export_pdf)
        layout.addSpacing(8)
        layout.addWidget(self._btn_theme)
        return bar

    def _build_central(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 500])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addWidget(splitter)
        return central

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._dropzone = DropZone(self._colors)
        self._canvas = ImageCanvas(self._colors)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._dropzone)
        self._stack.addWidget(self._canvas)
        self._dropzone.files_dropped.connect(self.add_files)

        self._file_header = QLabel("Fichiers  (0)")
        self._file_header.setObjectName("muted")
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(170)
        self._file_list.currentRowChanged.connect(self._on_file_selected)

        layout.addWidget(self._stack, 1)
        layout.addWidget(self._file_header)
        layout.addWidget(self._file_list)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("Éditeur de texte — corrigez le résultat de l'OCR")
        header.setObjectName("muted")
        layout.addWidget(header)

        self._editor = EditorPanel(self._colors)
        layout.addWidget(self._editor, 1)
        return panel

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(66)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 8, 18, 10)
        layout.setSpacing(12)

        self._status_label = QLabel("En attente de fichiers…")
        self._status_label.setObjectName("muted")
        self._time_label = QLabel("Temps : —")
        self._time_label.setObjectName("muted")
        self._count_label = QLabel("0 / 0")
        self._count_label.setObjectName("muted")
        self._progress = QProgressBar()
        self._progress.setFixedWidth(230)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._gauge = ConfidenceGauge(self._colors)

        layout.addWidget(self._status_label, 1)
        layout.addWidget(self._time_label)
        layout.addWidget(self._count_label)
        layout.addWidget(self._progress)
        layout.addWidget(self._gauge)
        return bar

    # ------------------------------------------------------------------ #
    # Thème
    # ------------------------------------------------------------------ #
    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._colors = Theme.LIGHT if self._theme == "light" else Theme.DARK
        self._btn_theme.setText(
            "Thème sombre" if self._theme == "light" else "Thème clair"
        )
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(Theme.stylesheet(self._colors))
        self._dropzone._colors = self._colors
        self._canvas._colors = self._colors
        self._gauge._colors = self._colors
        self._dropzone.update()
        self._canvas.update()
        self._gauge.update()

    # ------------------------------------------------------------------ #
    # Ajout de fichiers (images + PDF)
    # ------------------------------------------------------------------ #
    def _browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Ajouter des fichiers",
            "",
            "Documents et images (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.bmp *.pdf)",
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths: Sequence[str]) -> int:
        """Ajoute des fichiers à la file de traitement.

        Les PDF sont rastérisés page par page (PyMuPDF) dans un dossier
        temporaire ; chaque page devient une tâche distincte.

        Args:
            paths: Chemins des fichiers à ajouter (images ou PDF).

        Returns:
            Le nombre de tâches ajoutées (pages incluses).
        """
        added = 0
        for raw_path in paths:
            path = str(raw_path)
            ext = Path(path).suffix.lower()
            if ext in _IMAGE_EXTS:
                self._append_task(path, source=path, page=None)
                added += 1
            elif ext in _PDF_EXTS:
                pages = self._expand_pdf(path)
                if pages is None:
                    continue
                self._tasks.extend(pages)
                added += len(pages)
            else:
                self._set_status(f"Ignoré (format non supporté): {path}")

        if added:
            if self._stack.currentIndex() == 0:
                self._stack.setCurrentIndex(1)
            self._file_header.setText(f"Fichiers  ({len(self._tasks)})")
            self._rebuild_file_list()
            self._show_task(self._tasks[0]["path"])
            self._set_status(f"{added} tâche(s) ajoutée(s) — cliquez sur Démarrer")
        return added

    def _append_task(self, path: str, source: str, page: Optional[int]) -> None:
        label = f"{Path(source).name}" + (f"  ·  p.{page}" if page else "")
        self._tasks.append(
            {"path": path, "source": source, "page": page, "label": label}
        )

    def _expand_pdf(self, path: str) -> Optional[list[dict[str, Any]]]:
        """Rastérise chaque page d'un PDF en PNG temporaire."""
        try:
            import fitz
        except ImportError:
            QMessageBox.critical(
                self,
                "PyMuPDF manquant",
                "Le support des PDF nécessite PyMuPDF.\n"
                "Installez-le avec: pip install PyMuPDF",
            )
            self._set_status("PDF ignoré: PyMuPDF non installé")
            return None
        try:
            doc = fitz.open(path)
        except Exception as exc:
            QMessageBox.critical(self, "PDF illisible", f"{path}\n{exc}")
            self._set_status("PDF illisible")
            return None

        stem = Path(path).stem.replace(" ", "_")
        tasks: list[dict[str, Any]] = []
        try:
            for index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=160)
                samples = bytes(pix.samples)
                if pix.n == 2:
                    pix = pix.copy(alpha=False)
                    samples = bytes(pix.samples)
                if pix.n == 4:
                    qimage = QImage(
                        samples,
                        pix.width,
                        pix.height,
                        pix.stride,
                        QImage.Format.Format_RGBA8888,
                    )
                elif pix.n == 1:
                    qimage = QImage(
                        samples,
                        pix.width,
                        pix.height,
                        pix.stride,
                        QImage.Format.Format_Grayscale8,
                    )
                else:
                    qimage = QImage(
                        samples,
                        pix.width,
                        pix.height,
                        pix.stride,
                        QImage.Format.Format_RGB888,
                    )
                tmp_path = os.path.join(self._tempdir.name, f"{stem}_p{index:03d}.png")
                if not qimage.copy().save(tmp_path, "PNG"):
                    raise OSError(f"Rendu de la page {index} impossible")
                tasks.append(
                    {"path": tmp_path, "source": path, "page": index, "label": None}
                )
        except Exception as exc:
            QMessageBox.critical(self, "Rendu PDF échoué", f"{path}\n{exc}")
            self._set_status("Rendu PDF échoué")
            return None
        finally:
            doc.close()
        return tasks

    def _rebuild_file_list(self) -> None:
        self._file_list.clear()
        for index, task in enumerate(self._tasks):
            item = QListWidgetItem(task["label"])
            item.setData(Qt.ItemDataRole.UserRole, task["path"])
            self._file_list.addItem(item)

    # ------------------------------------------------------------------ #
    # Affichage d'une tâche
    # ------------------------------------------------------------------ #
    def _on_file_selected(self, row: int) -> None:
        if 0 <= row < len(self._tasks):
            self._show_task(self._tasks[row]["path"])

    def _show_task(self, path: str) -> None:
        self._current_path = path
        self._stack.setCurrentIndex(1)
        image = self._load_display_image(path)
        self._canvas.set_image(image)
        record = self._results.get(path)
        if record is not None and record.get("status") == "ok":
            items = record.get("results", [])
            self._canvas.set_overlay(
                [
                    (
                        list(item.get("box", [])),
                        str(item.get("text", "")),
                        float(item.get("confidence", 0.0)),
                    )
                    for item in items
                ]
            )
            text = "\n".join(str(item.get("text", "")) for item in items)
            self._editor.set_text(text or "(aucun texte détecté)")
            self._time_label.setText(f"Temps : {record['elapsed_ms']:.0f} ms")
            self._enable_exports(bool(text))
        else:
            self._canvas.clear_overlay()
            self._editor.set_text("")
            self._editor.widget().setPlaceholderText(
                "Ce fichier n'a pas encore été analysé."
            )
            self._time_label.setText("Temps : —")
            self._enable_exports(False)

    def _load_display_image(self, path: str) -> np.ndarray:
        """Charge l'image en appliquant le même prétraitement que l'OCR
        (les boîtes détectées sont alignées avec l'image affichée)."""
        try:
            image = self._preprocessor.read_image(path)
        except Exception:
            blank: np.ndarray = np.full((400, 600, 3), 245, dtype=np.uint8)
            return blank
        if self._engine is not None:
            try:
                options = getattr(self._engine, "preprocess_options", {})
                image = self._preprocessor.preprocess(image, **options)
            except Exception:
                pass
        return image

    # ------------------------------------------------------------------ #
    # Cycle de vie OCR
    # ------------------------------------------------------------------ #
    def _start_ocr(self) -> None:
        if not self._tasks:
            self._set_status("Ajoutez des fichiers avant de démarrer")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._reset_run_state()
        if self._engine is None:
            self._start_pending = True
            self._set_status("Initialisation du moteur OCR (chargement des modèles)…")
            self._load_engine()
            return
        self._launch_worker()

    def _load_engine(self) -> None:
        if self._engine_loader is not None and self._engine_loader.isRunning():
            return
        loader = _EngineLoader(self._engine_factory, self)
        self._engine_loader = loader
        loader.loaded.connect(self._on_engine_loaded)
        loader.failed.connect(self._on_engine_failed)
        loader.start()

    def _on_engine_loaded(self, engine: Any) -> None:
        self._engine = engine
        self._set_status("Moteur OCR prêt")
        if self._start_pending:
            self._start_pending = False
            self._launch_worker()

    def _on_engine_failed(self, message: str) -> None:
        self._start_pending = False
        self._set_status("Échec de l'initialisation du moteur")
        QMessageBox.critical(self, "Erreur d'initialisation", message)

    def _launch_worker(self) -> None:
        assert self._engine is not None
        paths = [task["path"] for task in self._tasks]
        worker = BatchWorker(self._engine, batch_size=self._batch_size, parent=self)
        self._worker = worker
        worker.progress_updated.connect(self._on_progress)
        worker.result_ready.connect(self._on_result)
        worker.error_occurred.connect(self._on_error)
        worker.state_changed.connect(self._on_state)
        worker.batch_finished.connect(self._on_finished)
        worker.start()
        try:
            worker.submit(paths)
        except RuntimeError as exc:
            self._set_status(f"Erreur: {exc}")
            QMessageBox.critical(self, "Erreur", str(exc))
            return
        self._set_busy(True)
        self._set_status(f"Analyse en cours… (0/{len(paths)})")

    def _cancel_ocr(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._set_status("Annulation demandée…")

    # ------------------------------------------------------------------ #
    # Slots du worker
    # ------------------------------------------------------------------ #
    def _on_progress(self, processed: int) -> None:
        total = len(self._tasks)
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(processed)
        self._count_label.setText(f"{processed} / {total}")

    def _on_result(self, record: dict[str, Any]) -> None:
        path = record.get("path", "")
        self._results[path] = record
        elapsed = float(record.get("elapsed_ms", 0.0))
        self._total_ms += elapsed
        items = record.get("results", [])
        if record.get("status") == "ok" and items:
            scores = [float(i.get("confidence", 0.0)) for i in items]
            self._confidences.append(sum(scores) / len(scores))
            self._update_gauge()
        self._update_file_marker(path, record.get("status") == "ok")
        if path == self._current_path:
            self._show_task(path)

    def _on_error(self, message: str) -> None:
        self._set_status(f"Erreur: {message}")

    def _on_state(self, state: str) -> None:
        if state == "cancelled":
            self._set_status("Traitement annulé")
        elif state == "failed":
            self._set_status("Échec du traitement")

    def _on_finished(self, count: int) -> None:
        self._set_busy(False)
        if self._worker is not None and self._worker.state == "cancelled":
            self._set_status(
                f"Annulé — {count} fichier(s) analysé(s) avant interruption"
            )
        else:
            failed = self._worker.failed if self._worker is not None else 0
            suffix = f", {failed} en erreur" if failed else ""
            self._set_status(
                f"Terminé — {count} fichier(s) analysé(s) en "
                f"{self._total_ms:.0f} ms au total{suffix}"
            )
        self._update_gauge()

    # ------------------------------------------------------------------ #
    # Indicateurs
    # ------------------------------------------------------------------ #
    def _update_gauge(self) -> None:
        if not self._confidences:
            self._gauge.set_value(None)
            return
        mean = sum(self._confidences) / len(self._confidences)
        self._gauge.set_value(min(100.0, max(0.0, mean * 100.0)))

    def _update_file_marker(self, path: str, success: bool) -> None:
        for row in range(self._file_list.count()):
            item = self._file_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                name = Path(os.fspath(path)).name
                item.setText(("✓  " if success else "✗  ") + name)
                break

    def _set_busy(self, busy: bool) -> None:
        self._btn_start.setEnabled(not busy)
        self._btn_cancel.setEnabled(busy)
        self._btn_add.setEnabled(not busy)

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)

    def _reset_run_state(self) -> None:
        self._results.clear()
        self._confidences.clear()
        self._total_ms = 0.0
        self._progress.setValue(0)
        self._count_label.setText(f"0 / {len(self._tasks)}")
        self._gauge.set_value(None)
        self._time_label.setText("Temps : —")
        self._enable_exports(False)
        self._rebuild_file_list()

    def _enable_exports(self, enabled: bool) -> None:
        for btn in (self._btn_export_txt, self._btn_export_docx, self._btn_export_pdf):
            btn.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # Exportations
    # ------------------------------------------------------------------ #
    def _current_text(self) -> str:
        return self._editor.text().strip()

    def _default_export_name(self, extension: str) -> str:
        source = "document"
        if self._current_path:
            task = self._find_task(self._current_path)
            if task is not None:
                source = Path(task["source"]).stem
        return f"{source}.{extension}"

    def _find_task(self, path: str) -> Optional[dict[str, Any]]:
        for task in self._tasks:
            if task["path"] == path:
                return task
        return None

    def _export_txt(self) -> None:
        text = self._current_text()
        if not text:
            self._warn_no_text()
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter en TXT", self._default_export_name("txt"), "Texte (*.txt)"
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Export échoué", str(exc))
            return
        self._set_status(f"Export TXT terminé: {path}")

    def _export_docx(self) -> None:
        text = self._current_text()
        if not text:
            self._warn_no_text()
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter en DOCX", self._default_export_name("docx"), "Word (*.docx)"
        )
        if not path:
            return
        try:
            from docx import Document
        except ImportError:
            QMessageBox.critical(
                self,
                "python-docx manquant",
                "Installez-le avec: pip install python-docx",
            )
            return
        try:
            document = Document()
            for line in text.splitlines():
                document.add_paragraph(line)
            document.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export échoué", str(exc))
            return
        self._set_status(f"Export DOCX terminé: {path}")

    def _export_pdf(self) -> None:
        text = self._current_text()
        if not text:
            self._warn_no_text()
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter en PDF", self._default_export_name("pdf"), "PDF (*.pdf)"
        )
        if not path:
            return
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate
        except ImportError:
            QMessageBox.critical(
                self, "ReportLab manquant", "Installez-le avec: pip install reportlab"
            )
            return
        try:
            from xml.sax.saxutils import escape

            styles = getSampleStyleSheet()
            body = styles["BodyText"]
            body.fontSize = 11
            story = [
                Paragraph(escape(line) or "&nbsp;", body) for line in text.splitlines()
            ]
            SimpleDocTemplate(path, pagesize=A4, title="ScriptVault OCR").build(story)
        except Exception as exc:
            QMessageBox.critical(self, "Export échoué", str(exc))
            return
        self._set_status(f"Export PDF terminé (couche texte sélectionnable): {path}")

    def _warn_no_text(self) -> None:
        QMessageBox.information(
            self, "Aucun texte", "L'éditeur est vide : rien à exporter."
        )

    # ------------------------------------------------------------------ #
    # Fermeture
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: Any) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.shutdown(3000)
            self._worker = None
        if self._engine_loader is not None and self._engine_loader.isRunning():
            self._engine_loader.wait(3000)
            self._engine_loader = None
        engine = self._engine
        self._engine = None
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        try:
            self._tempdir.cleanup()
        except Exception:
            pass
        gc.collect()
        event.accept()


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """Lance l'application desktop.

    Args:
        argv: Arguments de ligne de commande (ou ``None`` pour ``sys.argv``).

    Returns:
        Code de sortie de l'application Qt.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ScriptVault OCR — interface desktop.")
    parser.add_argument(
        "--lang", default="en", help="Langue des modèles OCR (en, fr, ch...)."
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="Threads CPU (0 = auto)."
    )
    parser.add_argument(
        "--model-dir", default=None, help="Dossier local des modèles OCR."
    )
    parser.add_argument("--batch", type=int, default=2, help="Taille de lot du worker.")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("ScriptVault OCR")
    app.setStyle("Fusion")

    window = MainWindow(
        lang=args.lang,
        model_dir=args.model_dir,
        cpu_threads=args.threads,
        batch_size=args.batch,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
