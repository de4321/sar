from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtGui import QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train.py")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCH_RE = re.compile(r"\[epoch\s+(\d+)\]")
TRAIN_PROGRESS_RE = re.compile(r"train:.*?(\d+)\s*/\s*(\d+)\s*\[")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dataset/SAR-ACD"))
    parser.add_argument("--split", type=Path, default=Path("splits/sar_acd_seed0.json"))
    parser.add_argument("--limited-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="letterbox",
        choices=["stretch", "letterbox"],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-gamma", type=float, default=0.98)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--debug-finite", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default="")
    return parser


class TrainMainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.setWindowTitle("SARCapsNet Training")
        self.resize(1020, 760)
        self.project_root = PROJECT_ROOT
        self._initial_args = argparse.Namespace(**vars(args))
        self.process: QProcess | None = None
        self._pending_ui_reset = False
        self._log_stream_buffer = ""
        self._total_epochs = max(1, int(args.epochs))

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self.paths_group = QGroupBox("Paths")
        paths_form = QFormLayout(self.paths_group)
        data_row, self.data_root_edit = self._build_path_row(
            str(args.data_root), choose_dir=True
        )
        split_row, self.split_edit = self._build_path_row(
            str(args.split),
            choose_dir=False,
            file_filter="JSON (*.json)",
        )
        out_row, self.out_dir_edit = self._build_path_row(str(args.out_dir), choose_dir=True)
        paths_form.addRow("data_root", data_row)
        paths_form.addRow("split", split_row)
        paths_form.addRow("out_dir", out_row)
        root.addWidget(self.paths_group)

        self.params_group = QGroupBox("Training Parameters")
        params_form = QFormLayout(self.params_group)

        self.limited_rate_spin = QDoubleSpinBox()
        self.limited_rate_spin.setDecimals(4)
        self.limited_rate_spin.setRange(0.0001, 1.0)
        self.limited_rate_spin.setSingleStep(0.1)
        self.limited_rate_spin.setValue(float(args.limited_rate))
        params_form.addRow("limited_rate", self.limited_rate_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000_000)
        self.seed_spin.setValue(int(args.seed))
        params_form.addRow("seed", self.seed_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(str(args.device))
        params_form.addRow("device", self.device_combo)

        self.input_size_spin = QSpinBox()
        self.input_size_spin.setRange(8, 1024)
        self.input_size_spin.setValue(int(args.input_size))
        params_form.addRow("input_size", self.input_size_spin)

        self.resize_mode_combo = QComboBox()
        self.resize_mode_combo.addItems(["letterbox", "stretch"])
        self.resize_mode_combo.setCurrentText(str(args.resize_mode))
        params_form.addRow("resize_mode", self.resize_mode_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1_000_000)
        self.epochs_spin.setValue(int(args.epochs))
        params_form.addRow("epochs", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 4096)
        self.batch_size_spin.setValue(int(args.batch_size))
        params_form.addRow("batch_size", self.batch_size_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(8)
        self.lr_spin.setRange(1e-8, 10.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(float(args.lr))
        params_form.addRow("lr", self.lr_spin)

        self.lr_gamma_spin = QDoubleSpinBox()
        self.lr_gamma_spin.setDecimals(6)
        self.lr_gamma_spin.setRange(1e-6, 1.0)
        self.lr_gamma_spin.setSingleStep(0.01)
        self.lr_gamma_spin.setValue(float(args.lr_gamma))
        params_form.addRow("lr_gamma", self.lr_gamma_spin)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 128)
        self.num_workers_spin.setValue(int(args.num_workers))
        params_form.addRow("num_workers", self.num_workers_spin)

        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setDecimals(4)
        self.grad_clip_spin.setRange(0.0, 1_000_000.0)
        self.grad_clip_spin.setSingleStep(0.5)
        self.grad_clip_spin.setValue(float(args.grad_clip))
        params_form.addRow("grad_clip", self.grad_clip_spin)

        self.run_name_edit = QLineEdit(str(args.run_name))
        self.run_name_edit.setPlaceholderText("optional")
        params_form.addRow("run_name", self.run_name_edit)
        root.addWidget(self.params_group)

        self.flags_group = QGroupBox("Flags")
        flags_layout = QHBoxLayout(self.flags_group)
        self.amp_check = QCheckBox("--amp")
        self.amp_check.setChecked(bool(args.amp))
        flags_layout.addWidget(self.amp_check)
        self.detect_anomaly_check = QCheckBox("--detect-anomaly")
        self.detect_anomaly_check.setChecked(bool(args.detect_anomaly))
        flags_layout.addWidget(self.detect_anomaly_check)
        self.debug_finite_check = QCheckBox("--debug-finite")
        self.debug_finite_check.setChecked(bool(args.debug_finite))
        flags_layout.addWidget(self.debug_finite_check)
        flags_layout.addStretch(1)
        root.addWidget(self.flags_group)

        self.progress_group = QGroupBox("Progress")
        progress_form = QFormLayout(self.progress_group)
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setTextVisible(True)
        self.epoch_progress.setFormat("%v / %m epochs")
        progress_form.addRow("epoch", self.epoch_progress)

        self.batch_progress = QProgressBar()
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setFormat("%v / %m batches")
        progress_form.addRow("batch", self.batch_progress)

        self.progress_detail_label = QLabel("Idle")
        progress_form.addRow("detail", self.progress_detail_label)
        root.addWidget(self.progress_group)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        button_row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_training)
        button_row.addWidget(self.stop_button)

        self.reset_button = QPushButton("Reset UI")
        self.reset_button.clicked.connect(self.reset_ui)
        button_row.addWidget(self.reset_button)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        button_row.addWidget(self.clear_log_button)

        button_row.addStretch(1)
        root.addLayout(button_row)

        self.status_label = QLabel("Ready")
        root.addWidget(self.status_label)

        self.cwd_label = QLabel(f"Working directory: {self.project_root}")
        root.addWidget(self.cwd_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        root.addWidget(self.log_view, 1)

        self._reset_progress(max(1, int(args.epochs)))
        self._set_running(False)

    def _build_path_row(
        self,
        initial_text: str,
        choose_dir: bool,
        file_filter: str = "",
    ) -> tuple[QWidget, QLineEdit]:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        edit = QLineEdit(initial_text)
        layout.addWidget(edit, 1)

        browse = QPushButton("Browse")
        if choose_dir:
            browse.clicked.connect(lambda: self._choose_dir(edit))
        else:
            browse.clicked.connect(lambda: self._choose_file(edit, file_filter))
        layout.addWidget(browse)

        return row, edit

    def _resolve_path(self, text: str) -> Path:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _suggest_dialog_dir(self, text: str) -> str:
        if not text:
            return str(self.project_root)
        p = self._resolve_path(text)
        if p.is_file():
            return str(p.parent)
        if p.exists():
            return str(p)
        parent = p.parent
        if parent.exists():
            return str(parent)
        return str(self.project_root)

    def _choose_dir(self, edit: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            self._suggest_dialog_dir(edit.text().strip()),
        )
        if selected:
            edit.setText(selected)

    def _choose_file(self, edit: QLineEdit, file_filter: str) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            self._suggest_dialog_dir(edit.text().strip()),
            file_filter or "All files (*)",
        )
        if selected:
            edit.setText(selected)

    def _set_running(self, running: bool) -> None:
        self.paths_group.setEnabled(not running)
        self.params_group.setEnabled(not running)
        self.flags_group.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.reset_button.setEnabled(True)

    def _cleanup_finished_process(self) -> None:
        if self.process is None:
            return
        if self.process.state() != QProcess.NotRunning:
            return
        self.process.deleteLater()
        self.process = None

    def _apply_args_to_form(self, args: argparse.Namespace) -> None:
        self.data_root_edit.setText(str(args.data_root))
        self.split_edit.setText(str(args.split))
        self.out_dir_edit.setText(str(args.out_dir))

        self.limited_rate_spin.setValue(float(args.limited_rate))
        self.seed_spin.setValue(int(args.seed))
        self.device_combo.setCurrentText(str(args.device))
        self.input_size_spin.setValue(int(args.input_size))
        self.resize_mode_combo.setCurrentText(str(args.resize_mode))
        self.epochs_spin.setValue(int(args.epochs))
        self.batch_size_spin.setValue(int(args.batch_size))
        self.lr_spin.setValue(float(args.lr))
        self.lr_gamma_spin.setValue(float(args.lr_gamma))
        self.num_workers_spin.setValue(int(args.num_workers))
        self.grad_clip_spin.setValue(float(args.grad_clip))
        self.run_name_edit.setText(str(args.run_name))

        self.amp_check.setChecked(bool(args.amp))
        self.detect_anomaly_check.setChecked(bool(args.detect_anomaly))
        self.debug_finite_check.setChecked(bool(args.debug_finite))

    def _perform_ui_reset(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return

        self._pending_ui_reset = False
        self._cleanup_finished_process()
        self._set_running(False)
        self._apply_args_to_form(self._initial_args)
        self._log_stream_buffer = ""
        self._reset_progress(max(1, int(self.epochs_spin.value())))
        self.status_label.setText("UI reset. Ready to start training.")
        self._append_log("[GUI] UI reset to initial options.\n")

    def reset_ui(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self._pending_ui_reset = True
            self._append_log("[GUI] UI reset requested. Stopping training first...\n")
            self.status_label.setText("Reset requested. Stopping training...")
            self.stop_training()
            return

        self._perform_ui_reset()

    def _append_log(self, text: str) -> None:
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.End)

    def clear_log(self) -> None:
        self.log_view.clear()

    def _reset_progress(self, total_epochs: int) -> None:
        self._total_epochs = max(1, int(total_epochs))
        self.epoch_progress.setRange(0, self._total_epochs)
        self.epoch_progress.setValue(0)
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.progress_detail_label.setText(f"Epoch 0/{self._total_epochs} | Batch 0/0")

    def _parse_progress_line(self, line: str) -> bool:
        """Parse known training progress lines and update progress bars."""
        matched = False
        epoch_m = EPOCH_RE.search(line)
        if epoch_m:
            current_epoch = min(int(epoch_m.group(1)), self._total_epochs)
            self.epoch_progress.setValue(current_epoch)
            if current_epoch >= self._total_epochs:
                self.batch_progress.setValue(self.batch_progress.maximum())
            self.progress_detail_label.setText(
                f"Epoch {current_epoch}/{self._total_epochs} | Batch {self.batch_progress.value()}/{self.batch_progress.maximum()}"
            )
            matched = True

        batch_m = TRAIN_PROGRESS_RE.search(line)
        if batch_m:
            batch_done = int(batch_m.group(1))
            batch_total = max(1, int(batch_m.group(2)))
            self.batch_progress.setRange(0, batch_total)
            self.batch_progress.setValue(min(batch_done, batch_total))
            current_epoch = self.epoch_progress.value()
            self.progress_detail_label.setText(
                f"Epoch {current_epoch}/{self._total_epochs} | Batch {batch_done}/{batch_total}"
            )
            matched = True
        return matched

    @staticmethod
    def _is_tqdm_line(line: str) -> bool:
        stripped = line.lstrip()
        return stripped.startswith("train:") and "/" in stripped and "[" in stripped

    def _handle_output_line(self, raw_line: str) -> None:
        clean_line = ANSI_ESCAPE_RE.sub("", raw_line)
        clean_line = clean_line.rstrip("\n")
        stripped = clean_line.strip()
        if stripped:
            self._parse_progress_line(stripped)
        if stripped and self._is_tqdm_line(stripped):
            # Hide noisy per-batch terminal redraw lines in GUI log.
            return
        self._append_log(clean_line + "\n")

    def _flush_log_buffer(self) -> None:
        if not self._log_stream_buffer:
            return
        self._handle_output_line(self._log_stream_buffer)
        self._log_stream_buffer = ""

    def _build_train_args(self) -> list[str]:
        args = [
            str(TRAIN_SCRIPT),
            "--data-root",
            self.data_root_edit.text().strip(),
            "--split",
            self.split_edit.text().strip(),
            "--limited-rate",
            f"{self.limited_rate_spin.value():.10g}",
            "--seed",
            str(self.seed_spin.value()),
            "--device",
            self.device_combo.currentText(),
            "--input-size",
            str(self.input_size_spin.value()),
            "--resize-mode",
            self.resize_mode_combo.currentText(),
            "--epochs",
            str(self.epochs_spin.value()),
            "--batch-size",
            str(self.batch_size_spin.value()),
            "--lr",
            f"{self.lr_spin.value():.10g}",
            "--lr-gamma",
            f"{self.lr_gamma_spin.value():.10g}",
            "--num-workers",
            str(self.num_workers_spin.value()),
            "--grad-clip",
            f"{self.grad_clip_spin.value():.10g}",
            "--out-dir",
            self.out_dir_edit.text().strip(),
        ]
        run_name = self.run_name_edit.text().strip()
        if run_name:
            args.extend(["--run-name", run_name])
        if self.amp_check.isChecked():
            args.append("--amp")
        if self.detect_anomaly_check.isChecked():
            args.append("--detect-anomaly")
        if self.debug_finite_check.isChecked():
            args.append("--debug-finite")
        return args

    def start_training(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return
        self._pending_ui_reset = False

        if not TRAIN_SCRIPT.exists():
            self.show_error(f"Training script not found:\n{TRAIN_SCRIPT}")
            return

        data_root_text = self.data_root_edit.text().strip()
        split_text = self.split_edit.text().strip()
        out_dir_text = self.out_dir_edit.text().strip()
        if not data_root_text or not split_text or not out_dir_text:
            self.show_error("data_root, split, and out_dir are required.")
            return

        split_path = self._resolve_path(split_text)
        if not split_path.exists():
            self.show_error(f"split file not found:\n{split_path}")
            return
        data_root_path = self._resolve_path(data_root_text)
        if not data_root_path.exists():
            self.show_error(f"data_root not found:\n{data_root_path}")
            return

        args = self._build_train_args()
        full_cmd = [sys.executable, *args]
        self._log_stream_buffer = ""
        self._reset_progress(total_epochs=self.epochs_spin.value())

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
        self.status_label.setText("Starting training process...")
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
            self._handle_output_line(line)

    def _on_started(self) -> None:
        self.status_label.setText("Training running...")
        self.progress_detail_label.setText(f"Epoch 0/{self._total_epochs} | Batch 0/0")

    def stop_training(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self._append_log("\n[GUI] Stop requested. Sending terminate...\n")
        self.status_label.setText("Stopping...")
        self.process.terminate()
        QTimer.singleShot(4000, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self._append_log("[GUI] Process still running. Sending kill.\n")
        self.process.kill()

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._drain_output()
        self._flush_log_buffer()
        normal_exit = exit_status == QProcess.NormalExit and exit_code == 0
        if normal_exit:
            self.epoch_progress.setValue(self._total_epochs)
            if self.batch_progress.maximum() > 0:
                self.batch_progress.setValue(self.batch_progress.maximum())
            self.progress_detail_label.setText(
                f"Epoch {self._total_epochs}/{self._total_epochs} | Completed"
            )
            self.status_label.setText("Training completed successfully.")
            self._append_log("\n[GUI] Training completed successfully.\n")
        else:
            self.status_label.setText(
                f"Training stopped (exit_code={exit_code}, exit_status={int(exit_status)})."
            )
            self._append_log(
                f"\n[GUI] Training stopped (exit_code={exit_code}, exit_status={int(exit_status)}).\n"
            )
        self._set_running(False)
        if self._pending_ui_reset:
            self._pending_ui_reset = False
            self._perform_ui_reset()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

    def _on_error(self, error: QProcess.ProcessError) -> None:
        self._append_log(f"\n[GUI] Process error: {int(error)}\n")
        self.status_label.setText("Training process error.")
        # Some startup/IO failures can leave controls disabled without a finish callback.
        self._cleanup_finished_process()
        if self.process is None:
            self._set_running(False)
            if self._pending_ui_reset:
                self._perform_ui_reset()

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()
                self.process.waitForFinished(1000)
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    window = TrainMainWindow(args)
    window.show()
    return app.exec()
