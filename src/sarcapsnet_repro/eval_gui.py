from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer
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
    QComboBox,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_EVAL_SCRIPT = PROJECT_ROOT / "src" / "run_eval.py"


class EvalWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = PROJECT_ROOT
        self.process: QProcess | None = None
        self._log_stream_buffer = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        paths_group = QGroupBox("Paths")
        paths_form = QFormLayout(paths_group)
        ckpt_row, self.ckpt_edit = self._build_path_row(
            "", mode="file", file_filter="Checkpoint (*.pt)"
        )
        self.ckpt_edit.setPlaceholderText("required")
        data_row, self.data_root_edit = self._build_path_row("", mode="dir")
        self.data_root_edit.setPlaceholderText("optional, default from ckpt")
        split_row, self.split_edit = self._build_path_row(
            "", mode="file", file_filter="JSON (*.json)"
        )
        self.split_edit.setPlaceholderText("optional, auto-detect split_used.json")
        out_row, self.out_cm_edit = self._build_path_row(
            "", mode="save", file_filter="CSV (*.csv)"
        )
        self.out_cm_edit.setPlaceholderText("optional")
        paths_form.addRow("ckpt", ckpt_row)
        paths_form.addRow("data_root", data_row)
        paths_form.addRow("split", split_row)
        paths_form.addRow("out_cm", out_row)
        root.addWidget(paths_group)

        params_group = QGroupBox("Eval Parameters")
        params_form = QFormLayout(params_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        params_form.addRow("device", self.device_combo)

        self.input_size_edit = QLineEdit("")
        self.input_size_edit.setPlaceholderText("optional, e.g. 28")
        params_form.addRow("input_size", self.input_size_edit)

        self.resize_mode_combo = QComboBox()
        self.resize_mode_combo.addItems(["auto", "letterbox", "stretch"])
        params_form.addRow("resize_mode", self.resize_mode_combo)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 4096)
        self.batch_size_spin.setValue(16)
        params_form.addRow("batch_size", self.batch_size_spin)
        root.addWidget(params_group)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start Eval")
        self.start_button.clicked.connect(self.start_eval)
        button_row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_eval)
        button_row.addWidget(self.stop_button)

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
        if mode == "save":
            selected, _ = QFileDialog.getSaveFileName(
                self,
                "Select output file",
                current,
                file_filter,
            )
            if selected:
                edit.setText(selected)
            return
        selected, _ = QFileDialog.getOpenFileName(self, "Select file", current, file_filter)
        if selected:
            edit.setText(selected)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _append_log(self, text: str) -> None:
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.End)

    def clear_log(self) -> None:
        self.log_view.clear()

    def _build_eval_args(self, ckpt_path: Path) -> list[str]:
        args = [
            str(RUN_EVAL_SCRIPT),
            "--cli",
            "--ckpt",
            str(ckpt_path),
            "--device",
            self.device_combo.currentText(),
            "--batch-size",
            str(self.batch_size_spin.value()),
        ]

        data_root = self.data_root_edit.text().strip()
        if data_root:
            args.extend(["--data-root", data_root])
        split = self.split_edit.text().strip()
        if split:
            args.extend(["--split", split])
        out_cm = self.out_cm_edit.text().strip()
        if out_cm:
            args.extend(["--out-cm", out_cm])
        input_size = self.input_size_edit.text().strip()
        if input_size:
            args.extend(["--input-size", input_size])

        resize_mode = self.resize_mode_combo.currentText()
        if resize_mode != "auto":
            args.extend(["--resize-mode", resize_mode])
        return args

    def start_eval(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return
        if not RUN_EVAL_SCRIPT.exists():
            self.show_error(f"run_eval.py not found:\n{RUN_EVAL_SCRIPT}")
            return

        ckpt_text = self.ckpt_edit.text().strip()
        if not ckpt_text:
            self.show_error("ckpt is required.")
            return
        ckpt_path = self._resolve_path(ckpt_text)
        if not ckpt_path.exists():
            self.show_error(f"checkpoint not found:\n{ckpt_path}")
            return

        args = self._build_eval_args(ckpt_path=ckpt_path)
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
        self.status_label.setText("Starting eval process...")
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
        self.status_label.setText("Eval running...")

    def stop_eval(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self._append_log("\n[GUI] Stop requested. Sending terminate...\n")
        self.status_label.setText("Stopping...")
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self._append_log("[GUI] Process still running. Sending kill.\n")
        self.process.kill()

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._drain_output()
        self._flush_log_buffer()
        if exit_status == QProcess.NormalExit and exit_code == 0:
            self.status_label.setText("Eval completed successfully.")
        else:
            self.status_label.setText(
                f"Eval stopped (exit_code={exit_code}, exit_status={int(exit_status)})."
            )
        self._set_running(False)
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

    def _on_error(self, error: QProcess.ProcessError) -> None:
        self._append_log(f"\n[GUI] Process error: {int(error)}\n")
        self.status_label.setText("Eval process error.")

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


class EvalMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SARCapsNet Eval")
        self.resize(920, 700)
        self.setCentralWidget(EvalWidget(self))


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def main(argv: list[str] | None = None) -> int:
    _ = build_arg_parser().parse_args(argv)
    app = QApplication(sys.argv)
    window = EvalMainWindow()
    window.show()
    return app.exec()
