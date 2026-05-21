"""
Chord token widgets — inline hyperlink-style labels that show a
chord diagram popup on hover.

ChordToken       — a single clickable chord label (e.g. "Em7")
ChordLineWidget  — parses one line of chord notation and builds a row of
                   ChordTokens mixed with plain notation labels
ChordPopup       — frameless floating window containing a ChordDiagramWidget
"""

from __future__ import annotations

import re

from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout, QVBoxLayout, QFrame
from PyQt6.QtCore import Qt, QPoint, QTimer, QRect
from PyQt6.QtGui import QFont, QCursor, QEnterEvent

from src.ui.chord_diagram_widget import ChordDiagramWidget

# ── Palette (mirrors main_window.py) ─────────────────────────────────────────
DARK   = "#12111a"
CARD   = "#26243a"
BORDER = "#3b3855"
ACCENT = "#c084fc"
TEXT   = "#f1f5f9"
DIM    = "#94a3b8"

# Regex that matches chord symbols: root + optional accidental + suffix
# e.g.  Em7  F#m  Dsus4  A7sus4  Cmaj7  Bb  G#dim  N.C.
_CHORD_RE = re.compile(
    r"""
    \b
    (
        [A-G][#b]?          # root
        (?:
            maj7?|m7?b5|m7?|dim7?|aug7?|sus[24]|7sus[24]|
            add\d+|maj\d+|\d+|ø|°
        )?
        (?:[#b]\d+)*        # alterations like #11, b9
    )
    \b
    |
    (N\.C\.)
    """,
    re.VERBOSE,
)

# Tokens that are pure notation (not chords)
_NOTATION_TOKENS = frozenset({
    "|:", ":|", "||", "|", "::", "—", "-",
})
_REPEAT_RE = re.compile(r"^x\d+$")


def _is_chord(token: str) -> bool:
    """Return True if the token looks like a chord symbol."""
    token = token.strip()
    if not token:
        return False
    if token in _NOTATION_TOKENS:
        return False
    if _REPEAT_RE.match(token):
        return False
    # Must start with A-G
    return bool(re.match(r"^[A-G]", token))


def _tokenise(line: str) -> list[tuple[str, bool]]:
    """
    Split a chord line into (text, is_chord) pairs preserving notation.

    e.g. "|: Em7 | G :| x2"
    → [('|:', False), ('Em7', True), ('|', False), ('G', True),
       (':|', False), ('x2', False)]
    """
    tokens: list[tuple[str, bool]] = []

    # Split on whitespace but keep track of each piece
    raw_parts = line.split()

    for part in raw_parts:
        # Each part may itself be a notation token or chord
        if part in _NOTATION_TOKENS:
            tokens.append((part, False))
        elif _REPEAT_RE.match(part):
            tokens.append((part, False))
        elif _is_chord(part):
            tokens.append((part, True))
        else:
            tokens.append((part, False))

    return tokens


# ── Popup window ──────────────────────────────────────────────────────────────

class ChordPopup(QWidget):
    """Frameless floating window with a chord diagram and note confidences."""

    def __init__(
        self,
        chord_symbol: str,
        voicing: list[int],
        note_confidences: dict | None = None,
        parent=None,
    ):
        super().__init__(
            parent,
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(f"""
            QWidget {{
                background: {CARD};
                border: 1px solid {ACCENT};
                border-radius: 10px;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        diagram = ChordDiagramWidget(chord_symbol, voicing)
        lay.addWidget(diagram)

        if note_confidences:
            conf_widget = self._build_confidence_widget(note_confidences)
            lay.addWidget(conf_widget)

        self.adjustSize()

    @staticmethod
    def _build_confidence_widget(note_confidences: dict) -> QWidget:
        """Build a small widget showing per-note chroma confidence bars."""
        container = QWidget()
        container.setStyleSheet("background: transparent; border: none;")
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(4, 0, 4, 4)
        vlay.setSpacing(2)

        header = QLabel("Notes detected")
        header.setStyleSheet(f"color: {DIM}; font-size: 10px; border: none;")
        vlay.addWidget(header)

        sorted_notes = sorted(note_confidences.items(), key=lambda x: -x[1])
        for note, conf in sorted_notes:
            row = QWidget()
            row.setStyleSheet("background: transparent; border: none;")
            hlay = QHBoxLayout(row)
            hlay.setContentsMargins(0, 0, 0, 0)
            hlay.setSpacing(4)

            note_lbl = QLabel(note)
            note_lbl.setFixedWidth(22)
            note_lbl.setStyleSheet(
                f"color: {ACCENT}; font-size: 11px; font-weight: bold; border: none;"
            )

            # Mini bar: filled blocks proportional to confidence
            filled = round(conf * 10)
            bar_text = "█" * filled + "░" * (10 - filled)
            bar_lbl = QLabel(bar_text)
            bar_lbl.setStyleSheet(
                f"color: {ACCENT}; font-size: 9px; letter-spacing: 1px; border: none;"
            )

            pct_lbl = QLabel(f"{conf:.0%}")
            pct_lbl.setFixedWidth(32)
            pct_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px; border: none;")

            hlay.addWidget(note_lbl)
            hlay.addWidget(bar_lbl)
            hlay.addWidget(pct_lbl)
            vlay.addWidget(row)

        return container

    def show_near(self, global_pos: QPoint):
        """Position the popup above the cursor and show it."""
        x = global_pos.x() - self.width() // 2
        y = global_pos.y() - self.height() - 12
        self.move(x, y)
        self.show()
        self.raise_()


# ── Chord token label ─────────────────────────────────────────────────────────

class ChordToken(QLabel):
    """
    A single chord label styled as a hyperlink.
    Shows a floating chord-diagram popup on hover.
    """

    # Shared popup instance — only one visible at a time
    _popup: ChordPopup | None = None

    def __init__(
        self,
        chord_symbol: str,
        note_confidences: dict | None = None,
        tuning=None,
        parent=None,
    ):
        super().__init__(chord_symbol, parent)

        self._chord_symbol    = chord_symbol
        self._note_confidences = note_confidences or {}
        self._tuning          = tuning
        self._voicing: list[int] | None = None
        self._hide_timer  = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_popup)

        self.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                text-decoration: underline;
                padding: 0 2px;
            }}
            QLabel:hover {{
                color: #e9d5ff;
            }}
        """)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMouseTracking(True)

    def _get_voicing(self) -> list[int] | None:
        if self._voicing is not None:
            return self._voicing
        try:
            from src.theory.chord_voicings import ChordVoicingEngine
            engine = ChordVoicingEngine(tuning=self._tuning)
            self._voicing = engine.get_voicing(self._chord_symbol)
        except Exception:
            self._voicing = None
        return self._voicing

    def enterEvent(self, event: QEnterEvent):
        self._hide_timer.stop()
        voicing = self._get_voicing()

        # No popup if no playable voicing exists
        if voicing is None:
            return
        from src.theory.chord_voicings import ChordVoicingEngine
        if not ChordVoicingEngine._is_playable(voicing):
            return

        # Destroy old popup if for a different chord
        if ChordToken._popup is not None:
            ChordToken._popup.close()
            ChordToken._popup = None

        ChordToken._popup = ChordPopup(
            self._chord_symbol, voicing,
            note_confidences=self._note_confidences or None,
        )
        ChordToken._popup.show_near(QCursor.pos())

    def leaveEvent(self, event):
        # Short delay so the user can move onto the popup without it vanishing
        self._hide_timer.start(200)

    @staticmethod
    def _hide_popup():
        if ChordToken._popup is not None:
            ChordToken._popup.close()
            ChordToken._popup = None


# ── One line of chord notation ────────────────────────────────────────────────

class ChordLineWidget(QWidget):
    """
    Renders a single line of chord notation as a horizontal row of widgets:
      • ChordToken  for chord symbols  (hoverable, underlined)
      • QLabel      for notation glyphs (|:  :|  ||  x3 …)
    """

    def __init__(
        self,
        line: str,
        confidence_lookup: dict | None = None,
        tuning=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        tokens = _tokenise(line)
        lookup = confidence_lookup or {}

        for text, is_chord in tokens:
            if is_chord:
                lay.addWidget(ChordToken(text, note_confidences=lookup.get(text), tuning=tuning))
            else:
                lbl = QLabel(text)
                lbl.setStyleSheet(
                    f"color: {DIM}; "
                    f"font-family: 'Consolas', 'Courier New', monospace; "
                    f"font-size: 14px;"
                )
                lay.addWidget(lbl)

        lay.addStretch()
