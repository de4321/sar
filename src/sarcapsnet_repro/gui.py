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
        self.setWindowTitle("SARCapsNet 图像预测器")
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
        self.ckpt_edit.setPlaceholderText("选择已训练的 checkpoint: best.pt")
        top_bar.addWidget(QLabel("Checkpoint 检查点"))
        top_bar.addWidget(self.ckpt_edit, 1)

        self.ckpt_button = QPushButton("选择模型")
        self.ckpt_button.clicked.connect(self.choose_checkpoint)
        top_bar.addWidget(self.ckpt_button)

        self.reload_button = QPushButton("加载模型")
        self.reload_button.clicked.connect(self.load_predictor_from_ui)
        top_bar.addWidget(self.reload_button)
        root.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([540, 500])
        root.addWidget(splitter, 1)

        self.status_label = QLabel("请选择模型和图像以开始推理。")
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

        group = QGroupBox("图像")
        group_layout = QVBoxLayout(group)

        self.preview_label = QLabel("未选择图像")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 480)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        group_layout.addWidget(self.preview_label, 1)

        self.image_path_label = QLabel("路径: -")
        self.image_path_label.setWordWrap(True)
        group_layout.addWidget(self.image_path_label)

        self.image_button = QPushButton("选择图像")
        self.image_button.clicked.connect(self.choose_image)
        group_layout.addWidget(self.image_button)

        layout.addWidget(group)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        summary_group = QGroupBox("预测结果")
        summary_layout = QGridLayout(summary_group)
        summary_layout.addWidget(QLabel("类别"), 0, 0)
        self.pred_class_label = QLabel("-")
        self.pred_class_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(self.pred_class_label, 0, 1)

        summary_layout.addWidget(QLabel("置信度"), 1, 0)
        self.confidence_label = QLabel("-")
        summary_layout.addWidget(self.confidence_label, 1, 1)

        summary_layout.addWidget(QLabel("设备"), 2, 0)
        self.device_label = QLabel("-")
        summary_layout.addWidget(self.device_label, 2, 1)

        summary_layout.addWidget(QLabel("输入"), 3, 0)
        self.input_label = QLabel("-")
        summary_layout.addWidget(self.input_label, 3, 1)

        layout.addWidget(summary_group)

        details_group = QGroupBox("Top 结果")
        details_layout = QVBoxLayout(details_group)
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        details_layout.addWidget(self.result_text)
        layout.addWidget(details_group, 1)

        return panel

    def choose_checkpoint(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Checkpoint 文件",
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
            self.show_error("请先选择一个 checkpoint 文件。")
            return
        self.load_predictor(Path(ckpt_text))

    def load_predictor(self, ckpt_path: Path) -> None:
        try:
            self.predictor = SARCapsPredictor(ckpt_path, device=self.device_name)
        except Exception as exc:  # noqa: BLE001
            self.predictor = None
            self.show_error(f"模型加载失败：\n{exc}")
            self.status_label.setText("模型加载失败。")
            return

        self.device_label.setText(self.predictor.device.type)
        self.input_label.setText(
            f"{self.predictor.input_size} x {self.predictor.input_size} ({self.predictor.resize_mode})"
        )
        self.status_label.setText(f"模型已加载: {ckpt_path}")

        if self.image_path is not None:
            self.run_prediction()

    def choose_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not filename:
            return

        self.image_path = Path(filename)
        self.image_path_label.setText(f"路径: {self.image_path}")
        self.update_preview(self.image_path)
        self.run_prediction()

    def update_preview(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.preview_label.setText("无法预览该图像")
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
            self.status_label.setText("请先选择并加载模型。")
            return
        if self.image_path is None:
            self.status_label.setText("请先选择图像。")
            return

        try:
            result = self.predictor.predict_image(self.image_path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(f"推理失败：\n{exc}")
            self.status_label.setText("推理失败。")
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
        self.status_label.setText(f"预测结果：{result.image_path.name} 为 {result.predicted_class}")

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "错误", message)


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
