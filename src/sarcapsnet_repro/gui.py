from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .predict import SARCapsPredictor, find_latest_checkpoint


class MainWindow(QMainWindow):
    def __init__(self, ckpt_path: Path | None = None, device: str = "auto") -> None:
        super().__init__()
        self.setWindowTitle("SARCapsNet Inference")
        self.resize(1120, 680)

        self.device_name = device
        self.predictor: SARCapsPredictor | None = None
        self.image_path: Path | None = None
        self.original_base_pixmap: QPixmap | None = None
        self.heatmap_base_pixmap: QPixmap | None = None

        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        top_bar = QHBoxLayout()
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("Path to checkpoint, e.g. runs/.../best.pt")
        top_bar.addWidget(QLabel("Checkpoint"))
        top_bar.addWidget(self.ckpt_edit, 1)

        self.ckpt_button = QPushButton("Browse")
        self.ckpt_button.clicked.connect(self.choose_checkpoint)
        top_bar.addWidget(self.ckpt_button)

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.load_predictor_from_ui)
        top_bar.addWidget(self.reload_button)
        root.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([540, 500])
        root.addWidget(splitter, 1)

        self.status_label = QLabel("Load a checkpoint and select an image.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        initial_ckpt = ckpt_path or find_latest_checkpoint()
        if initial_ckpt is not None:
            self.ckpt_edit.setText(str(initial_ckpt))
            self.load_predictor(initial_ckpt)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        group = QGroupBox("Image")
        group_layout = QVBoxLayout(group)
        preview_row = QHBoxLayout()
        preview_row.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Original"))
        self.original_preview_label = QLabel("Select an image")
        self.original_preview_label.setAlignment(Qt.AlignCenter)
        self.original_preview_label.setMinimumSize(240, 240)
        self.original_preview_label.setFrameShape(QFrame.StyledPanel)
        self.original_preview_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        left_col.addWidget(self.original_preview_label, 1)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Heatmap Overlay"))
        self.heatmap_preview_label = QLabel("Run prediction to show heatmap")
        self.heatmap_preview_label.setAlignment(Qt.AlignCenter)
        self.heatmap_preview_label.setMinimumSize(240, 240)
        self.heatmap_preview_label.setFrameShape(QFrame.StyledPanel)
        self.heatmap_preview_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        right_col.addWidget(self.heatmap_preview_label, 1)

        preview_row.addLayout(left_col, 1)
        preview_row.addLayout(right_col, 1)
        group_layout.addLayout(preview_row, 1)

        self.image_path_label = QLabel("Path: -")
        self.image_path_label.setWordWrap(True)
        group_layout.addWidget(self.image_path_label)

        self.image_button = QPushButton("Choose Image")
        self.image_button.clicked.connect(self.choose_image)
        group_layout.addWidget(self.image_button)

        layout.addWidget(group)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        summary_group = QGroupBox("Prediction Summary")
        summary_layout = QGridLayout(summary_group)
        summary_layout.addWidget(QLabel("Class"), 0, 0)
        self.pred_class_label = QLabel("-")
        self.pred_class_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(self.pred_class_label, 0, 1)

        summary_layout.addWidget(QLabel("Confidence"), 1, 0)
        self.confidence_label = QLabel("-")
        summary_layout.addWidget(self.confidence_label, 1, 1)

        summary_layout.addWidget(QLabel("Device"), 2, 0)
        self.device_label = QLabel("-")
        summary_layout.addWidget(self.device_label, 2, 1)

        summary_layout.addWidget(QLabel("Model Input"), 3, 0)
        self.input_label = QLabel("-")
        summary_layout.addWidget(self.input_label, 3, 1)

        summary_layout.addWidget(QLabel("Heatmap peak"), 4, 0)
        self.attention_peak_label = QLabel("-")
        summary_layout.addWidget(self.attention_peak_label, 4, 1)

        layout.addWidget(summary_group)

        details_group = QGroupBox("Top Predictions")
        details_layout = QVBoxLayout(details_group)
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        details_layout.addWidget(self.result_text)
        layout.addWidget(details_group, 1)

        return panel

    def choose_checkpoint(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint",
            str(Path.cwd()),
            "PyTorch Checkpoint (*.pt)",
        )
        if not filename:
            return
        self.ckpt_edit.setText(filename)
        self.load_predictor(Path(filename))

    def load_predictor_from_ui(self) -> None:
        ckpt_text = self.ckpt_edit.text().strip()
        if not ckpt_text:
            self.show_error("Please provide a checkpoint path.")
            return
        self.load_predictor(Path(ckpt_text))

    def load_predictor(self, ckpt_path: Path) -> None:
        try:
            self.predictor = SARCapsPredictor(ckpt_path, device=self.device_name)
        except Exception as exc:  # noqa: BLE001
            self.predictor = None
            self.show_error(f"Failed to load checkpoint:\n{exc}")
            self.status_label.setText("Failed to load checkpoint.")
            return

        self.device_label.setText(self.predictor.device.type)
        self.input_label.setText(
            f"{self.predictor.input_size} x {self.predictor.input_size} ({self.predictor.resize_mode})"
        )
        self.status_label.setText(f"Checkpoint loaded: {ckpt_path}")

        if self.image_path is not None:
            self.run_prediction()

    def choose_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not filename:
            return

        self.image_path = Path(filename)
        self.image_path_label.setText(f"Path: {self.image_path}")
        self.update_preview(self.image_path)
        self.heatmap_base_pixmap = None
        self.refresh_preview_pixmaps()
        self.run_prediction()

    def update_preview(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.original_base_pixmap = None
            self.refresh_preview_pixmaps()
            return
        self.set_original_pixmap(pixmap)

    def set_original_pixmap(self, pixmap: QPixmap) -> None:
        self.original_base_pixmap = pixmap
        self.refresh_preview_pixmaps()

    def set_heatmap_pixmap(self, pixmap: QPixmap) -> None:
        self.heatmap_base_pixmap = pixmap
        self.refresh_preview_pixmaps()

    @staticmethod
    def _refresh_preview_label(
        label: QLabel,
        base_pixmap: QPixmap | None,
        empty_text: str,
    ) -> None:
        if base_pixmap is None:
            label.clear()
            label.setText(empty_text)
            return

        scaled = base_pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.clear()
        label.setPixmap(scaled)

    def refresh_preview_pixmaps(self) -> None:
        self._refresh_preview_label(
            self.original_preview_label,
            self.original_base_pixmap,
            "Select an image",
        )
        self._refresh_preview_label(
            self.heatmap_preview_label,
            self.heatmap_base_pixmap,
            "Run prediction to show heatmap",
        )

    @staticmethod
    def _array_to_qpixmap(image_rgb: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(image_rgb.astype(np.uint8))
        height, width = arr.shape[:2]
        qimg = QImage(arr.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _jet_colormap(values: np.ndarray) -> np.ndarray:
        v = np.clip(values.astype(np.float32), 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1) * 255.0

    @classmethod
    def render_attention_overlay(
        cls,
        model_input_image: np.ndarray,
        attention_map: np.ndarray,
        peak: tuple[int, int],
    ) -> QPixmap:
        base = np.clip(model_input_image.astype(np.float32), 0.0, 1.0)
        attn = np.clip(attention_map.astype(np.float32), 0.0, 1.0)
        if base.shape != attn.shape:
            raise ValueError(
                f"input and heatmap shapes must match, got {base.shape} vs {attn.shape}"
            )

        base_rgb = np.repeat((base * 255.0).astype(np.float32)[..., None], 3, axis=2)
        heat_rgb = cls._jet_colormap(attn).astype(np.float32)
        alpha = (0.2 + 0.55 * attn)[..., None]
        overlay = np.clip((1.0 - alpha) * base_rgb + alpha * heat_rgb, 0.0, 255.0)

        h, w = attn.shape
        peak_y = int(np.clip(peak[0], 0, h - 1))
        peak_x = int(np.clip(peak[1], 0, w - 1))
        radius = max(2, min(h, w) // 12)

        overlay[
            max(peak_y - 1, 0) : min(peak_y + 2, h),
            max(peak_x - radius, 0) : min(peak_x + radius + 1, w),
        ] = [255.0, 255.0, 255.0]
        overlay[
            max(peak_y - radius, 0) : min(peak_y + radius + 1, h),
            max(peak_x - 1, 0) : min(peak_x + 2, w),
        ] = [255.0, 255.0, 255.0]

        return cls._array_to_qpixmap(overlay.astype(np.uint8))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.refresh_preview_pixmaps()

    def run_prediction(self) -> None:
        if self.predictor is None:
            self.status_label.setText("Please load a checkpoint first.")
            return
        if self.image_path is None:
            self.status_label.setText("Please select an image first.")
            return

        try:
            result = self.predictor.predict_image(self.image_path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(f"Prediction failed:\n{exc}")
            self.status_label.setText("Prediction failed.")
            return

        self.pred_class_label.setText(
            f"{result.predicted_class} (index {result.predicted_index})"
        )
        self.confidence_label.setText(f"{result.confidence:.2%}")
        self.device_label.setText(result.device)

        peak_y, peak_x = result.attention_peak
        self.attention_peak_label.setText(f"(x={peak_x}, y={peak_y})")

        try:
            overlay = self.render_attention_overlay(
                result.model_input_image,
                result.attention_map,
                result.attention_peak,
            )
        except Exception:
            # Keep UI responsive even if heatmap rendering fails unexpectedly.
            self.heatmap_base_pixmap = None
            self.refresh_preview_pixmaps()
        else:
            self.set_heatmap_pixmap(overlay)

        lines = []
        for row in result.probabilities:
            class_name = str(row["class_name"])
            class_index = int(row["index"])
            prob = float(row["probability"])
            lines.append(f"{class_name:<12} idx={class_index}  prob={prob:.4f}")
        self.result_text.setPlainText("\n".join(lines))
        self.status_label.setText(
            f"Prediction done: {result.image_path.name} -> {result.predicted_class}"
        )

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    window = MainWindow(ckpt_path=args.ckpt, device=args.device)
    window.show()
    return app.exec()
