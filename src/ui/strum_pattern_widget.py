"""
Strumming pattern visualisation widget.

Draws a one-bar strumming diagram similar to Ultimate Guitar / Chordify:

    ↑        ↑  ↑
    |        |  |
────|──|──|──|──|──|──|──
       ↓     ↓     ↓  ↓
    1  &  2  &  3  &  4  &
    ⌣  ⌣  ⌣  ⌣  ⌣  ⌣  ⌣  ⌣

Down-strokes hang BELOW the string line.
Up-strokes rise ABOVE the string line.
Silent positions show only a short tick on the string line.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QFrame, QVBoxLayout, QLabel
from PyQt6.QtCore    import Qt, QRect, QPoint
from PyQt6.QtGui     import (
    QPainter, QPen, QColor, QFont, QFontMetrics, QPolygon, QPainterPath,
)

# Palette — mirrors main_window.py
DARK   = "#12111a"
CARD   = "#26243a"
BORDER = "#3b3855"
ACCENT = "#c084fc"
TEXT   = "#f1f5f9"
DIM    = "#94a3b8"


class _PatternCanvas(QWidget):
    """Inner widget that paints the strum diagram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pattern = None
        self.setMinimumHeight(150)
        self.setMaximumHeight(150)

    def set_pattern(self, pattern):
        self._pattern = pattern
        self.update()

    def paintEvent(self, event):
        if self._pattern is None:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        beats = self._pattern.beats
        n     = len(beats)
        if n == 0:
            p.end()
            return

        w = self.width()
        h = self.height()

        # ── Geometry ──────────────────────────────────────────────────────────
        pad_h       = 28          # horizontal padding each side
        usable_w    = w - 2 * pad_h
        col_w       = usable_w / n

        string_y    = int(h * 0.50)   # the horizontal string/beat line
        down_len    = int(h * 0.28)   # length of a downstroke arrow
        up_len      = int(h * 0.22)   # length of an upstroke arrow (shorter)
        head_px     = 6               # arrowhead size
        label_gap   = 10              # pixels below string line to first label

        # ── Colours ───────────────────────────────────────────────────────────
        c_text   = QColor(TEXT)
        c_dim    = QColor(DIM)
        c_line   = QColor(BORDER)

        # ── String line ───────────────────────────────────────────────────────
        p.setPen(QPen(c_line, 1))
        p.drawLine(pad_h - 4, string_y, w - pad_h + 4, string_y)

        # ── Font for labels ───────────────────────────────────────────────────
        font_beat = QFont("Segoe UI", 9, QFont.Weight.Bold)
        font_sub  = QFont("Segoe UI", 9)
        fm_beat   = QFontMetrics(font_beat)
        fm_sub    = QFontMetrics(font_sub)

        for i, beat in enumerate(beats):
            cx       = int(pad_h + (i + 0.5) * col_w)
            is_main  = beat.label != "&"

            # Tick on the string line
            p.setPen(QPen(c_dim, 1))
            p.drawLine(cx, string_y - 4, cx, string_y + 4)

            # ── Arrow ─────────────────────────────────────────────────────────
            if beat.direction == "D":
                # Downstroke: arrow hangs BELOW the string line
                y_tip  = string_y + down_len
                y_stem = string_y + 1

                p.setPen(QPen(c_text, 2))
                p.drawLine(cx, y_stem, cx, y_tip - head_px)

                pts = QPolygon([
                    QPoint(cx,            y_tip),
                    QPoint(cx - head_px,  y_tip - head_px),
                    QPoint(cx + head_px,  y_tip - head_px),
                ])
                p.setBrush(c_text)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(pts)
                p.setBrush(Qt.BrushStyle.NoBrush)

            elif beat.direction == "U":
                # Upstroke: arrow rises ABOVE the string line
                y_tip  = string_y - up_len
                y_stem = string_y - 1

                p.setPen(QPen(c_text, 2))
                p.drawLine(cx, y_stem, cx, y_tip + head_px)

                pts = QPolygon([
                    QPoint(cx,            y_tip),
                    QPoint(cx - head_px,  y_tip + head_px),
                    QPoint(cx + head_px,  y_tip + head_px),
                ])
                p.setBrush(c_text)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(pts)
                p.setBrush(Qt.BrushStyle.NoBrush)

            # ── Beat label ────────────────────────────────────────────────────
            if is_main:
                p.setFont(font_beat)
                fm  = fm_beat
                col = c_text
            else:
                p.setFont(font_sub)
                fm  = fm_sub
                col = c_dim

            lbl_w = fm.horizontalAdvance(beat.label)
            lbl_x = cx - lbl_w // 2
            lbl_y = string_y + label_gap + fm.ascent()

            p.setPen(QPen(col, 1))
            p.drawText(lbl_x, lbl_y, beat.label)

            # ── Small arch below label ─────────────────────────────────────
            arch_y  = string_y + label_gap + fm.height() + 6
            arch_r  = int(col_w * 0.32)
            arch_rect = QRect(cx - arch_r, arch_y - arch_r, 2 * arch_r, 2 * arch_r)
            p.setPen(QPen(c_dim, 1))
            # Draw bottom semicircle (0° to 180° going clockwise, i.e. negative span)
            p.drawArc(arch_rect, 0 * 16, -180 * 16)

        p.end()


class StrumPatternWidget(QFrame):
    """
    Card widget that shows the detected strumming pattern.

    Usage::

        widget = StrumPatternWidget()
        widget.set_pattern(pipeline_result.strum_pattern)
        layout.addWidget(widget)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(4)

        # ── Header ────────────────────────────────────────────────────────────
        self._title = QLabel("Strumming Pattern")
        self._title.setStyleSheet(
            f"color: {TEXT}; font-size: 13px; font-weight: bold; border: none;"
        )
        lay.addWidget(self._title)

        self._subtitle = QLabel("")
        self._subtitle.setStyleSheet(
            f"color: {DIM}; font-size: 11px; border: none;"
        )
        lay.addWidget(self._subtitle)

        # ── Canvas ────────────────────────────────────────────────────────────
        self._canvas = _PatternCanvas()
        lay.addWidget(self._canvas)

    def set_pattern(self, pattern):
        """Load and display a StrumPattern."""
        if pattern is None:
            self._subtitle.setText("No pattern detected")
            return

        ts = pattern.time_signature
        self._subtitle.setText(
            f"● Whole song  {pattern.bpm:.0f} bpm  ·  {ts[0]}/{ts[1]}"
        )
        self._canvas.set_pattern(pattern)
