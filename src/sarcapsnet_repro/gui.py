from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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
        self.setWindowTitle("SARCapsNet Image Predictor")
        self.resize(1120, 680)

        self.device_name = device
        self.predictor: SARCapsPredictor | None = None
        self.image_path: Path | None = None

        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        top_bar = QHBoxLayout()
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("Select a trained checkpoint: best.pt")
        top_bar.addWidget(QLabel("Checkpoint"))
        top_bar.addWidget(self.ckpt_edit, 1)

        self.ckpt_button = QPushButton("Select Model")
        self.ckpt_button.clicked.connect(self.choose_checkpoint)
        top_bar.addWidget(self.ckpt_button)

        self.reload_button = QPushButton("Load Model")
        self.reload_button.clicked.connect(self.load_predictor_from_ui)
        top_bar.addWidget(self.reload_button)
        root.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([540, 500])
        root.addWidget(splitter, 1)

        self.status_label = QLabel("Select a model and an image to start inference.")
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

        self.preview_label = QLabel("No image selected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 480)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        group_layout.addWidget(self.preview_label, 1)

        self.image_path_label = QLabel("Path: -")
        self.image_path_label.setWordWrap(True)
        group_layout.addWidget(self.image_path_label)

        self.image_button = QPushButton("Select Image")
        self.image_button.clicked.connect(self.choose_image)
        group_layout.addWidget(self.image_button)

        layout.addWidget(group)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        summary_group = QGroupBox("Prediction")
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

        summary_layout.addWidget(QLabel("Input"), 3, 0)
        self.input_label = QLabel("-")
        summary_layout.addWidget(self.input_label, 3, 1)

        layout.addWidget(summary_group)

        details_group = QGroupBox("Top Results")
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
            self.show_error("Please select a checkpoint file first.")
            return
        self.load_predictor(Path(ckpt_text))

    def load_predictor(self, ckpt_path: Path) -> None:
        try:
            self.predictor = SARCapsPredictor(ckpt_path, device=self.device_name)
        except Exception as exc:  # noqa: BLE001
            self.predictor = None
            self.show_error(f"Failed to load model:\n{exc}")
            self.status_label.setText("Model load failed.")
            return

        self.device_label.setText(self.predictor.device.type)
        self.input_label.setText(
            f"{self.predictor.input_size} x {self.predictor.input_size} ({self.predictor.resize_mode})"
        )
        self.status_label.setText(f"Model loaded: {ckpt_path}")

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
        self.run_prediction()

    def update_preview(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.preview_label.setText("Unable to preview this image")
            return
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.image_path is not None:
            self.update_preview(self.image_path)

    def run_prediction(self) -> None:
        if self.predictor is None:
            self.status_label.setText("Select and load a model first.")
            return
        if self.image_path is None:
            self.status_label.setText("Select an image first.")
            return

        try:
            result = self.predictor.predict_image(self.image_path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(f"Inference failed:\n{exc}")
            self.status_label.setText("Inference failed.")
            return

        self.pred_class_label.setText(
            f"{result.predicted_class} (index {result.predicted_index})"
        )
        self.confidence_label.setText(f"{result.confidence:.2%}")
        self.device_label.setText(result.device)

        lines = []
        for row in result.probabilities:
            class_name = str(row["class_name"])
            class_index = int(row["index"])
            prob = float(row["probability"])
            lines.append(f"{class_name:<12} idx={class_index}  prob={prob:.4f}")
        self.result_text.setPlainText("\n".join(lines))
        self.status_label.setText(f"Predicted {result.predicted_class} for {result.image_path.name}")

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
