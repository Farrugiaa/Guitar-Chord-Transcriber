"""
QPainter-based guitar chord diagram widget.

Each ChordDiagramWidget renders a single chord as a fretboard grid with:
  - Mute (×) / open (○) markers above the nut
  - Finger dots on fret positions
  - Barre indicator (thick rounded bar)
  - Fret position number for high-position chords
  - String names (E A D G B e) below
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QScrollArea, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont

# ── Colours ──────────────────────────────────────────────────────────────────
C_BG      = QColor("#26243a")
C_BORDER  = QColor("#3b3855")
C_STRING  = QColor("#5c5880")
C_NUT     = QColor("#e2e8f0")
C_DOT     = QColor("#c084fc")
C_BARRE   = QColor("#9333ea")
C_OPEN    = QColor("#4ade80")
C_MUTE    = QColor("#f87171")
C_TEXT    = QColor("#f1f5f9")
C_DIM     = QColor("#94a3b8")


class ChordDiagramWidget(QWidget):
    """Draws one guitar chord diagram."""

    # Fixed layout constants (pixels)
    W  = 118   # total widget width
    H  = 172   # total widget height
    ML = 24    # left margin  (room for fret number)
    MT = 44    # top margin   (title + nut markers)
    MR = 12    # right margin
    MB = 18    # bottom margin (string names)
    NF = 4     # number of fret rows shown

    def __init__(
        self,
        chord_symbol: str,
        voicing: list[int],
        string_names: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.chord_symbol = chord_symbol
        self.voicing      = voicing          # 6 ints: -1=mute, 0=open, N=fret
        self.string_names = string_names or ["E", "A", "D", "G", "B", "e"]
        self.setFixedSize(self.W, self.H)
        self.setToolTip(chord_symbol)

    # ── geometry helpers ─────────────────────────────────────────────────────

    @property
    def _gw(self) -> float:
        return self.W - self.ML - self.MR

    @property
    def _gh(self) -> float:
        return self.H - self.MT - self.MB

    @property
    def _ss(self) -> float:           # string spacing
        return self._gw / 5

    @property
    def _fs(self) -> float:           # fret spacing
        return self._gh / self.NF

    def _sx(self, s: int) -> int:     # x of string s  (0=low E … 5=high e)
        return int(self.ML + s * self._ss)

    def _fy(self, row: int) -> int:   # y of TOP of fret row (1-based)
        return int(self.MT + (row - 1) * self._fs)

    # ── paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_BG)

        fretted   = [f for f in self.voicing if f > 0]
        has_open  = any(f == 0 for f in self.voicing if f >= 0)
        min_fret  = min(fretted) if fretted else 1
        # Start at fret 1 only when open strings exist AND the lowest fretted
        # note is within the 4-fret display window.  If fretted notes are all
        # above fret 4 (barre position), start there so dots are visible.
        start     = 1 if (has_open and min_fret <= self.NF) else (min_fret if fretted else 1)

        self._title(p)
        self._nut_markers(p)
        self._grid(p, start)
        self._barre(p, start)
        self._dots(p, start)
        self._string_names(p)
        if start > 1:
            self._fret_label(p, start)

    def _title(self, p: QPainter):
        f = QFont("Segoe UI", 10, QFont.Weight.Bold)
        p.setFont(f)
        p.setPen(C_TEXT)
        p.drawText(QRect(0, 4, self.W, 20),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                   self.chord_symbol)

    def _nut_markers(self, p: QPainter):
        yc = self.MT - 12
        r  = 5
        p.setFont(QFont("Segoe UI", 8))
        for s, fret in enumerate(self.voicing):
            x = self._sx(s)
            if fret == -1:
                p.setPen(QPen(C_MUTE, 1.8))
                p.drawLine(x - r, yc - r, x + r, yc + r)
                p.drawLine(x + r, yc - r, x - r, yc + r)
            elif fret == 0:
                p.setPen(QPen(C_OPEN, 1.8))
                p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                p.drawEllipse(x - r, yc - r, r * 2, r * 2)

    def _grid(self, p: QPainter, start: int):
        # Nut line
        pen = QPen(C_NUT, 3) if start == 1 else QPen(C_BORDER, 1)
        p.setPen(pen)
        p.drawLine(self.ML, self.MT, int(self.ML + self._gw), self.MT)

        # Fret lines
        p.setPen(QPen(C_BORDER, 1))
        for row in range(1, self.NF + 1):
            y = self._fy(row + 1) if row < self.NF else int(self.MT + self._gh)
            y = int(self.MT + row * self._fs)
            p.drawLine(self.ML, y, int(self.ML + self._gw), y)

        # String lines
        p.setPen(QPen(C_STRING, 1))
        for s in range(6):
            x = self._sx(s)
            p.drawLine(x, self.MT, x, int(self.MT + self._gh))

    def _barre(self, p: QPainter, start: int):
        from collections import Counter
        counts = Counter(f for f in self.voicing if f > 0)
        for fret, n in counts.items():
            if n >= 4 and fret == start:
                row     = fret - start + 1
                yc      = int(self._fy(row) + self._fs / 2)
                r       = int(self._fs * 0.35)
                strings = [s for s, f in enumerate(self.voicing) if f == fret]
                x1      = self._sx(min(strings))
                x2      = self._sx(max(strings))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(C_BARRE))
                p.drawRoundedRect(x1 - r // 2, yc - r, x2 - x1 + r, r * 2, r, r)
                break

    def _dots(self, p: QPainter, start: int):
        from collections import Counter
        counts    = Counter(f for f in self.voicing if f > 0)
        barre_fret = next(
            (f for f, n in counts.items() if n >= 4 and f == start), None
        )
        r = int(self._fs * 0.32)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_DOT))
        for s, fret in enumerate(self.voicing):
            if fret <= 0:
                continue
            if barre_fret and fret == barre_fret:
                continue
            row = fret - start + 1
            if not (1 <= row <= self.NF):
                continue
            x  = self._sx(s)
            yc = int(self._fy(row) + self._fs / 2)
            p.drawEllipse(x - r, yc - r, r * 2, r * 2)

    def _string_names(self, p: QPainter):
        p.setFont(QFont("Segoe UI", 7))
        p.setPen(C_DIM)
        yb = int(self.MT + self._gh + 13)
        for s, name in enumerate(self.string_names):
            x = self._sx(s)
            p.drawText(QRect(x - 8, yb - 8, 16, 16),
                       Qt.AlignmentFlag.AlignCenter, name)

    def _fret_label(self, p: QPainter, start: int):
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(C_DIM)
        y = int(self.MT + self._fs / 2 - 7)
        p.drawText(QRect(0, y, self.ML - 4, 14),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{start}fr")


# ── Scrollable row of diagrams ────────────────────────────────────────────────

class ChordDiagramPanel(QScrollArea):
    """Horizontal scrollable panel of ChordDiagramWidgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedHeight(ChordDiagramWidget.H + 20)

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(16)
        self._layout.addStretch()
        self.setWidget(self._container)

    def set_chords(self, chord_symbols: list[str], tuning=None):
        """Replace all diagrams with the given chord list."""
        from src.theory.chord_voicings import ChordVoicingEngine
        engine = ChordVoicingEngine(tuning=tuning)
        str_names = engine._str_names

        # Clear existing
        while self._layout.count() > 1:      # keep the trailing stretch
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for symbol in chord_symbols:
            voicing = engine.get_voicing(symbol)
            if voicing is None:
                continue
            if not any(f > 0 for f in voicing):   # skip all-open (no fretted notes)
                continue
            widget = ChordDiagramWidget(symbol, voicing, string_names=str_names)
            self._layout.insertWidget(self._layout.count() - 1, widget)
