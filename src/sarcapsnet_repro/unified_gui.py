from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .eval_gui import EvalWidget
from .gui import MainWindow as InferenceMainWindow
from .splits_gui import MakeSplitsWidget
from .train_gui import TrainMainWindow, build_arg_parser as build_train_arg_parser


class UnifiedMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SARCapsNet Workbench")
        self.resize(1360, 860)

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.inference_window = InferenceMainWindow(ckpt_path=None, device="auto")
        self.train_window = TrainMainWindow(build_train_arg_parser().parse_args([]))
        self.eval_widget = EvalWidget(self)
        self.splits_widget = MakeSplitsWidget(self)

        self.tabs.addTab(self._embed_main_window(self.inference_window), "Inference")
        self.tabs.addTab(self._embed_main_window(self.train_window), "Train")
        self.tabs.addTab(self.eval_widget, "Eval")
        self.tabs.addTab(self.splits_widget, "Make Splits")

    def _embed_main_window(self, window: QMainWindow) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        window.setWindowFlags(Qt.Widget)
        window.setParent(container)
        layout.addWidget(window)
        window.show()
        return container


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def main(argv: list[str] | None = None) -> int:
    _ = build_arg_parser().parse_args(argv)
    app = QApplication(sys.argv)
    window = UnifiedMainWindow()
    window.show()
    return app.exec()
