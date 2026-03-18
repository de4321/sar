from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SPLIT_SCRIPT = PROJECT_ROOT / "src" / "run_make_splits.py"


class MakeSplitsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = PROJECT_ROOT
        self.process: QProcess | None = None
        self._log_stream_buffer = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        group = QGroupBox("Split Parameters")
        form = QFormLayout(group)

        data_row, self.data_root_edit = self._build_path_row("", mode="dir")
        self.data_root_edit.setPlaceholderText("dataset/SAR-ACD")
        out_row, self.out_edit = self._build_path_row(
            "splits/sar_acd_seed0.json",
            mode="save",
            file_filter="JSON (*.json)",
        )
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000_000)
        self.seed_spin.setValue(0)

        form.addRow("data_root", data_row)
        form.addRow("seed", self.seed_spin)
        form.addRow("out", out_row)
        root.addWidget(group)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Generate Split")
        self.start_button.clicked.connect(self.start_split)
        button_row.addWidget(self.start_button)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        button_row.addWidget(self.clear_log_button)
        button_row.addStretch(1)
        root.addLayout(button_row)

        self.status_label = QLabel("Ready")
        root.addWidget(self.status_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        root.addWidget(self.log_view, 1)

        self._set_running(False)

    def _build_path_row(
        self,
        initial_text: str,
        mode: str,
        file_filter: str = "All files (*)",
    ) -> tuple[QWidget, QLineEdit]:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        edit = QLineEdit(initial_text)
        layout.addWidget(edit, 1)

        browse = QPushButton("Browse")
        browse.clicked.connect(
            lambda: self._browse_into(edit=edit, mode=mode, file_filter=file_filter)
        )
        layout.addWidget(browse)
        return row, edit

    def _resolve_path(self, text: str) -> Path:
        p = Path(text).expanduser()
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def _suggest_dialog_dir(self, text: str) -> str:
        if not text:
            return str(self.project_root)
        p = self._resolve_path(text)
        if p.is_file():
            return str(p.parent)
        if p.exists():
            return str(p)
        if p.parent.exists():
            return str(p.parent)
        return str(self.project_root)

    def _browse_into(self, edit: QLineEdit, mode: str, file_filter: str) -> None:
        current = self._suggest_dialog_dir(edit.text().strip())
        if mode == "dir":
            selected = QFileDialog.getExistingDirectory(self, "Select folder", current)
            if selected:
                edit.setText(selected)
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Select output file",
            current,
            file_filter,
        )
        if selected:
            edit.setText(selected)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)

    def _append_log(self, text: str) -> None:
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.End)

    def clear_log(self) -> None:
        self.log_view.clear()

    def _build_split_args(self) -> list[str]:
        data_root = self.data_root_edit.text().strip() or "dataset/SAR-ACD"
        out = self.out_edit.text().strip() or "splits/sar_acd_seed0.json"
        return [
            str(RUN_SPLIT_SCRIPT),
            "--cli",
            "--data-root",
            data_root,
            "--seed",
            str(self.seed_spin.value()),
            "--out",
            out,
        ]

    def start_split(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return
        if not RUN_SPLIT_SCRIPT.exists():
            self.show_error(f"run_make_splits.py not found:\n{RUN_SPLIT_SCRIPT}")
            return

        data_root_path = self._resolve_path(
            self.data_root_edit.text().strip() or "dataset/SAR-ACD"
        )
        if not data_root_path.exists():
            self.show_error(f"data_root not found:\n{data_root_path}")
            return

        args = self._build_split_args()
        full_cmd = [sys.executable, *args]
        self._log_stream_buffer = ""

        self._append_log("\n" + "=" * 78 + "\n")
        self._append_log("Launching:\n")
        self._append_log(" ".join(shlex.quote(part) for part in full_cmd) + "\n\n")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(str(self.project_root))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._drain_output)
        process.started.connect(self._on_started)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)

        self.process = process
        self._set_running(True)
        self.status_label.setText("Generating split...")
        process.start()

    def _drain_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput())
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self._log_stream_buffer += text
        lines = self._log_stream_buffer.split("\n")
        self._log_stream_buffer = lines.pop()
        for line in lines:
            self._append_log(line + "\n")

    def _flush_log_buffer(self) -> None:
        if not self._log_stream_buffer:
            return
        self._append_log(self._log_stream_buffer + "\n")
        self._log_stream_buffer = ""

    def _on_started(self) -> None:
        self.status_label.setText("Split generation running...")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._drain_output()
        self._flush_log_buffer()
        if exit_status == QProcess.NormalExit and exit_code == 0:
            self.status_label.setText("Split generated successfully.")
        else:
            self.status_label.setText(
                f"Split generation stopped (exit_code={exit_code}, exit_status={int(exit_status)})."
            )
        self._set_running(False)
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

    def _on_error(self, error: QProcess.ProcessError) -> None:
        self._append_log(f"\n[GUI] Process error: {int(error)}\n")
        self.status_label.setText("Split process error.")

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


class MakeSplitsMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SARCapsNet Make Splits")
        self.resize(860, 600)
        self.setCentralWidget(MakeSplitsWidget(self))


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def main(argv: list[str] | None = None) -> int:
    _ = build_arg_parser().parse_args(argv)
    app = QApplication(sys.argv)
    window = MakeSplitsMainWindow()
    window.show()
    return app.exec()
