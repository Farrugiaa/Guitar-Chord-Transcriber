"""
Main application window for the Guitar Chord Extractor desktop app.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QScrollArea, QFrame, QProgressBar,
    QSizePolicy, QGridLayout, QSplitter, QTextEdit, QComboBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    _HAS_MEDIA = True
except ImportError:
    _HAS_MEDIA = False

from src.ui.chord_diagram_widget import ChordDiagramPanel
from src.ui.chord_token_widget import ChordLineWidget
from src.ui.strum_pattern_widget import StrumPatternWidget

# ── Palette ───────────────────────────────────────────────────────────────────
DARK   = "#12111a"
PANEL  = "#1c1b27"
CARD   = "#26243a"
BORDER = "#3b3855"
ACCENT = "#c084fc"
TEXT   = "#f1f5f9"
DIM    = "#94a3b8"
GREEN  = "#4ade80"
RED    = "#f87171"

APP_STYLE = f"""
* {{
    font-family: 'Segoe UI', sans-serif;
    color: {TEXT};
}}
QMainWindow, QWidget {{
    background-color: {DARK};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {PANEL};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {PANEL};
    height: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QSplitter::handle {{
    background: {BORDER};
    width: 1px;
}}
"""

BTN_PRIMARY = f"""
QPushButton {{
    background: {ACCENT};
    color: #0f0f1a;
    border: none;
    border-radius: 8px;
    padding: 10px 0;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton:hover  {{ background: #d8b4fe; }}
QPushButton:pressed {{ background: #a855f7; }}
QPushButton:disabled {{ background: {BORDER}; color: {DIM}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 0;
    font-size: 12px;
}}
QPushButton:hover  {{ border-color: {ACCENT}; background: #312f4a; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {DIM}; }}
"""

BTN_PLAY = f"""
QPushButton {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 16px;
    font-size: 14px;
    min-width: 32px; max-width: 32px;
    min-height: 32px; max-height: 32px;
}}
QPushButton:hover  {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {DIM}; }}
"""

BTN_PAUSE = f"""
QPushButton {{
    background: {CARD};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 6px;
    padding: 6px 0;
    font-size: 12px;
    font-weight: bold;
}}
QPushButton:hover  {{ background: #312f4a; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {DIM}; border-color: {BORDER}; }}
"""

CARD_STYLE = f"""
QFrame {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
"""

PROGRESS_STYLE = f"""
QProgressBar {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 4px;
}}
"""

SECTION_LABEL_STYLE = f"color: {ACCENT}; font-size: 12px; font-weight: bold;"
CHORD_LINE_STYLE    = f"""
    color: {TEXT};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
"""


# ── Background workers ────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    """Runs the pipeline off the main thread so the UI stays responsive."""

    status   = pyqtSignal(str)
    finished = pyqtSignal(object)   # PipelineResult
    failed   = pyqtSignal(str)

    def __init__(self, audio_path: str, vocab_level: str = "extended",
                 device: str = "cpu", tuning_key: str = "auto"):
        super().__init__()
        self.audio_path  = audio_path
        self.vocab_level = vocab_level
        self.device      = device
        self.tuning_key  = tuning_key   # "auto" or a key from TUNINGS dict

    def run(self):
        try:
            import torch
            from src.pipeline.pipeline import GuitarChordPipeline
            from src.theory.tuning import TUNINGS

            device = "cuda" if torch.cuda.is_available() else "cpu"

            forced_tuning = None
            if self.tuning_key != "auto":
                forced_tuning = TUNINGS.get(self.tuning_key)

            self.status.emit(f"Building pipeline… ({device.upper()})")
            repo_root = Path(__file__).resolve().parents[2]
            recogniser_ckpt = repo_root / "guitar_only" / "checkpoints" / "chord_recogniser_guitar_weights.pt"
            pipeline = GuitarChordPipeline.from_checkpoints(
                recogniser_path=str(recogniser_ckpt) if recogniser_ckpt.exists() else None,
                vocab_level=self.vocab_level,
                device=device,
                tuning=forced_tuning,
            )

            self.status.emit("Running analysis…")
            pre_separated = Path(self.audio_path).stem.lower().endswith("_guitar")
            if pre_separated:
                self.status.emit("Running analysis… (pre-separated guitar stem detected)")
            result = pipeline.process(self.audio_path, pre_separated=pre_separated)

            self.status.emit("Saving extracted guitar audio…")
            stem        = Path(self.audio_path).stem
            guitar_path = str(Path(self.audio_path).parent / f"{stem}_guitar.wav")
            result.save_guitar_audio(guitar_path)
            result._guitar_audio_path = guitar_path

            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.failed.emit(traceback.format_exc())




# ── Helper widgets ────────────────────────────────────────────────────────────

def _card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setStyleSheet(CARD_STYLE)
    return f


def _label(text: str, style: str = "", parent=None) -> QLabel:
    lbl = QLabel(text, parent)
    if style:
        lbl.setStyleSheet(style)
    return lbl


class InfoCard(QFrame):
    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setStyleSheet(CARD_STYLE)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        self._title = QLabel(title)
        self._title.setStyleSheet(f"color: {DIM}; font-size: 11px;")

        self._value = QLabel(value)
        self._value.setStyleSheet(
            f"color: {TEXT}; font-size: 18px; font-weight: bold;"
        )
        self._value.setWordWrap(True)

        lay.addWidget(self._title)
        lay.addWidget(self._value)

    def set_value(self, v: str):
        self._value.setText(v)


class SectionBlock(QFrame):
    """One [Section] heading + chord lines."""

    def __init__(
        self,
        label: str,
        lines: list[str],
        confidence_lookup: dict | None = None,
        tuning=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-left: 3px solid {ACCENT};
                border-radius: 8px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(4)

        lbl = QLabel(f"[{label}]")
        lbl.setStyleSheet(SECTION_LABEL_STYLE + " border: none;")
        lay.addWidget(lbl)

        for line in lines:
            lay.addWidget(ChordLineWidget(line, confidence_lookup=confidence_lookup, tuning=tuning))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Chord Extractor")
        self.resize(1100, 720)
        self.setMinimumSize(800, 560)

        self._worker:       AnalysisWorker | None = None
        self._result        = None
        self._orig_path:    str = ""
        self._guitar_path:  str = ""

        # Media player (instance-owned so state signals work correctly)
        self._player:      QMediaPlayer | None = None
        self._audio_out:   QAudioOutput | None = None
        if _HAS_MEDIA:
            self._player    = QMediaPlayer()
            self._audio_out = QAudioOutput()
            self._player.setAudioOutput(self._audio_out)
            self._audio_out.setVolume(1.0)
            self._player.playbackStateChanged.connect(self._on_playback_state)

        self.setStyleSheet(APP_STYLE)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_content())
        splitter.setSizes([240, 860])
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet(f"QWidget#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}")

        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(16, 20, 16, 20)
        lay.setSpacing(16)

        # Title
        title = QLabel("🎸  Chord Extractor")
        title.setStyleSheet(
            f"color: {ACCENT}; font-size: 15px; font-weight: bold;"
        )
        lay.addWidget(title)

        lay.addWidget(self._h_rule())

        # File section
        lay.addWidget(_label("Audio File", f"color: {DIM}; font-size: 11px;"))

        self._file_label = QLabel("No file selected")
        self._file_label.setStyleSheet(
            f"color: {TEXT}; font-size: 12px; "
            f"background: {CARD}; border: 1px solid {BORDER}; "
            f"border-radius: 6px; padding: 6px 8px;"
        )
        self._file_label.setWordWrap(True)
        self._file_label.setMaximumHeight(54)
        lay.addWidget(self._file_label)

        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setStyleSheet(BTN_SECONDARY)
        self._browse_btn.clicked.connect(self._browse)
        lay.addWidget(self._browse_btn)

        # Analyse button
        self._analyse_btn = QPushButton("Analyse")
        self._analyse_btn.setStyleSheet(BTN_PRIMARY)
        self._analyse_btn.setEnabled(False)
        self._analyse_btn.clicked.connect(self._analyse)
        lay.addWidget(self._analyse_btn)

        # Progress
        self._progress = QProgressBar()
        self._progress.setStyleSheet(PROGRESS_STYLE)
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(6)
        self._progress.hide()
        lay.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)

        lay.addWidget(self._h_rule())

        # Playback
        lay.addWidget(_label("Playback", f"color: {DIM}; font-size: 11px;"))

        self._play_orig_btn   = self._play_row("Original",  self._play_original)
        self._play_guitar_btn = self._play_row("Extracted guitar", self._play_guitar)
        lay.addWidget(self._play_orig_btn[0])
        lay.addWidget(self._play_guitar_btn[0])

        self._play_orig_btn[1].setEnabled(False)
        self._play_guitar_btn[1].setEnabled(False)

        # Pause / Resume / Stop controls
        pb_row = QWidget()
        pb_row.setStyleSheet("background: transparent;")
        pb_h = QHBoxLayout(pb_row)
        pb_h.setContentsMargins(0, 0, 0, 0)
        pb_h.setSpacing(6)

        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.setStyleSheet(BTN_PAUSE)
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        pb_h.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹")
        self._stop_btn.setStyleSheet(BTN_PLAY)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip("Stop")
        self._stop_btn.clicked.connect(self._stop_playback)
        pb_h.addWidget(self._stop_btn)

        lay.addWidget(pb_row)

        lay.addWidget(self._h_rule())

        # Export
        self._export_btn = QPushButton("Export chord sheet…")
        self._export_btn.setStyleSheet(BTN_SECONDARY)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        lay.addWidget(self._export_btn)

        self._midi_btn = QPushButton("Export MIDI…")
        self._midi_btn.setStyleSheet(BTN_SECONDARY)
        self._midi_btn.setEnabled(False)
        self._midi_btn.clicked.connect(self._export_midi)
        lay.addWidget(self._midi_btn)

        self._tab_btn = QPushButton("Export Tab (.txt)…")
        self._tab_btn.setStyleSheet(BTN_SECONDARY)
        self._tab_btn.setEnabled(False)
        self._tab_btn.clicked.connect(self._export_tab)
        lay.addWidget(self._tab_btn)

        lay.addStretch()

        return sidebar

    def _play_row(self, label: str, slot) -> tuple[QWidget, QPushButton]:
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        h   = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT}; font-size: 12px;")

        btn = QPushButton("▶")
        btn.setStyleSheet(BTN_PLAY)
        btn.clicked.connect(slot)

        h.addWidget(lbl)
        h.addStretch()
        h.addWidget(btn)

        return row, btn

    def _h_rule(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    # ── Main content ──────────────────────────────────────────────────────────

    def _build_content(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setStyleSheet(f"background: {DARK};")
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content.setStyleSheet(f"background: {DARK};")
        self._content_lay = QVBoxLayout(content)
        self._content_lay.setContentsMargins(24, 24, 24, 24)
        self._content_lay.setSpacing(20)

        self._build_placeholder()

        scroll.setWidget(content)
        outer.addWidget(scroll)

        return wrapper

    def _build_placeholder(self):
        ph = QLabel("Open an audio file and click Analyse to extract chords.")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet(f"color: {DIM}; font-size: 14px;")
        ph.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._content_lay.addWidget(ph)
        self._content_lay.addStretch()

    def _clear_content(self):
        while self._content_lay.count():
            item = self._content_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.aiff *.m4a);;All Files (*)",
        )
        if path:
            self._orig_path = path
            self._file_label.setText(Path(path).name)
            self._analyse_btn.setEnabled(True)
            self._play_orig_btn[1].setEnabled(True)

    def _analyse(self):
        if not self._orig_path:
            return

        self._analyse_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._midi_btn.setEnabled(False)
        self._tab_btn.setEnabled(False)
        self._progress.show()
        self._clear_content()

        self._worker = AnalysisWorker(self._orig_path)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_status(self, msg: str):
        self._status_lbl.setText(msg)

    def _on_finished(self, result):
        self._result     = result
        self._guitar_path = getattr(result, "_guitar_audio_path", "")

        self._progress.hide()
        self._status_lbl.setText("Done")
        self._analyse_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._midi_btn.setEnabled(True)
        self._tab_btn.setEnabled(True)
        self._play_guitar_btn[1].setEnabled(bool(self._guitar_path))

        self._render_result(result)

    def _on_failed(self, tb: str):
        self._progress.hide()
        self._status_lbl.setText("Error — see details below")
        self._analyse_btn.setEnabled(True)
        self._clear_content()

        err = QLabel(f"Analysis failed:\n\n{tb}")
        err.setStyleSheet(f"color: {RED}; font-family: Consolas; font-size: 11px;")
        err.setWordWrap(True)
        self._content_lay.addWidget(err)
        self._content_lay.addStretch()

    def _play_original(self):
        if self._orig_path:
            self._play_audio(self._orig_path)

    def _play_guitar(self):
        if self._guitar_path:
            self._play_audio(self._guitar_path)

    def _play_audio(self, path: str):
        """Load and play an audio file via QMediaPlayer."""
        if self._player is not None:
            self._player.setSource(QUrl.fromLocalFile(str(Path(path).absolute())))
            self._player.play()
        else:
            # Fallback: open with system default player (no pause control)
            import os
            os.startfile(str(Path(path).absolute()))

    def _toggle_pause(self):
        """Pause if playing; resume if paused."""
        if self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _stop_playback(self):
        """Stop playback and reset position to start."""
        if self._player is not None:
            self._player.stop()

    def _on_playback_state(self, state):
        """Update pause/stop button labels when the player state changes."""
        if self._player is None:
            return
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        paused  = state == QMediaPlayer.PlaybackState.PausedState

        self._pause_btn.setEnabled(playing or paused)
        self._stop_btn.setEnabled(playing or paused)
        self._pause_btn.setText("⏸  Pause" if playing else "▶  Resume")

    def _export_midi(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI", "", "MIDI Files (*.mid);;All Files (*)"
        )
        if path:
            if not path.lower().endswith(".mid"):
                path += ".mid"
            try:
                self._result.export_midi(path)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "MIDI Export Failed", str(e))

    def _export_tab(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Tab", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            if not path.lower().endswith(".txt"):
                path += ".txt"
            try:
                self._result.export_tab(path)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Tab Export Failed", str(e))

    def _export(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chord Sheet", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._result.summary())

    # ── Result rendering ──────────────────────────────────────────────────────

    def _render_result(self, result):
        self._clear_content()
        a = result.analysis

        # ── Info cards row ────────────────────────────────────────────────────
        cards_row = QWidget()
        cards_row.setStyleSheet("background: transparent;")
        grid = QGridLayout(cards_row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        tuning     = result.tuning
        confidence = result.tuning_confidence
        conf_str   = f" ({confidence:.0%})" if confidence > 0 else ""
        tuning_str = f"{tuning.name}{conf_str}"

        self._card_key    = InfoCard("Key",    f"{a.key} {a.scale_type}")
        self._card_bpm    = InfoCard("BPM",    f"{a.bpm:.0f}")
        self._card_time   = InfoCard("Time",   f"{a.time_signature[0]}/{a.time_signature[1]}")
        self._card_chords = InfoCard("Chords", str(len(result.chord_events)))
        self._card_tuning = InfoCard("Tuning", tuning_str)

        grid.addWidget(self._card_key,    0, 0)
        grid.addWidget(self._card_bpm,    0, 1)
        grid.addWidget(self._card_time,   0, 2)
        grid.addWidget(self._card_chords, 0, 3)

        grid.addWidget(self._card_tuning, 1, 0, 1, 4)

        self._content_lay.addWidget(cards_row)

        # ── Section divider ───────────────────────────────────────────────────
        self._content_lay.addWidget(self._section_heading("Chord Progression"))

        # ── Sections ──────────────────────────────────────────────────────────
        self._render_progression(result)

        # ── Strumming pattern ─────────────────────────────────────────────────
        if getattr(result, "strum_pattern", None) is not None:
            self._content_lay.addWidget(self._section_heading("Strumming Pattern"))
            sp_widget = StrumPatternWidget()
            sp_widget.set_pattern(result.strum_pattern)
            self._content_lay.addWidget(sp_widget)

        # ── Chord diagrams ────────────────────────────────────────────────────
        self._content_lay.addWidget(self._section_heading("Chord Diagrams"))
        self._render_diagrams(result)

        self._content_lay.addStretch()

    def _section_heading(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {DIM}; font-size: 11px; font-weight: bold; "
            f"text-transform: uppercase; letter-spacing: 1px;"
        )
        return lbl

    def _render_progression(self, result):
        """Parse formatted_output into SectionBlock widgets."""
        raw = result.formatted_output
        lines = raw.splitlines()

        # Build chord-symbol → note_confidences lookup from chord events.
        confidence_lookup: dict = {}
        for event in result.chord_events:
            if event.symbol not in confidence_lookup and event.note_confidences:
                confidence_lookup[event.symbol] = event.note_confidences

        # Skip the header lines (time sig / BPM) already shown in cards
        chord_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("4/", "3/", "6/", "2/")):
                continue
            chord_lines.append(stripped)

        # Group by section headers [Intro], [Verse], etc.
        current_label = None
        current_lines: list[str] = []
        has_sections = False

        tuning = getattr(result, "tuning", None)

        for line in chord_lines:
            if line.startswith("[") and line.endswith("]"):
                if current_label is not None:
                    self._content_lay.addWidget(
                        SectionBlock(current_label, current_lines, confidence_lookup, tuning=tuning)
                    )
                current_label = line[1:-1]
                current_lines = []
                has_sections  = True
            elif line.startswith("──") or line.startswith("---"):
                # Stop before diagram section
                break
            elif line:
                current_lines.append(line)

        # Flush last section / unsectioned content
        if current_label is not None and current_lines:
            self._content_lay.addWidget(
                SectionBlock(current_label, current_lines, confidence_lookup, tuning=tuning)
            )
        elif not has_sections and current_lines:
            # No section labels — show as one plain block
            block = SectionBlock("Full song", current_lines, confidence_lookup, tuning=tuning)
            self._content_lay.addWidget(block)

    def _render_diagrams(self, result):
        """Add a ChordDiagramPanel with the unique chords detected."""
        unique_chords = list(dict.fromkeys(
            e.symbol for e in result.chord_events if e.symbol != "N.C."
        ))

        panel = ChordDiagramPanel()
        panel.set_chords(unique_chords, tuning=getattr(result, "tuning", None))
        self._content_lay.addWidget(panel)

