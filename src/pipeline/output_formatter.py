"""
Output formatter for chord progressions.

Converts frame-level chord predictions and music analysis results
into human-readable text notation:
    4/4 C Em | G B7 ||

Supports bar lines, time signatures, repeat markers, and
section labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.features.structure_segmenter import SongSection


@dataclass
class ChordEvent:
    """A single chord with its timing."""
    symbol: str
    start_time: float
    end_time: float
    beat_position: Optional[float] = None
    bar_number: Optional[int] = None
    note_confidences: dict = field(default_factory=dict)  # {note_name: 0-1}


class OutputFormatter:
    """Format detected chords into text notation."""

    def __init__(
        self,
        beats_per_bar: int = 4,
        beat_value: int = 4,
        simplify_repeats: bool = True,
    ):
        self.beats_per_bar = beats_per_bar
        self.beat_value = beat_value
        self.simplify_repeats = simplify_repeats

    def format_progression(
        self,
        chord_events: list[ChordEvent],
        bpm: float,
        time_signature: tuple[int, int] = (4, 4),
        key: Optional[str] = None,
        sections: Optional[list[SongSection]] = None,
    ) -> str:
        """
        Format a list of chord events into text notation.

        Args:
            chord_events: List of ChordEvent with timing
            bpm: Tempo in BPM
            time_signature: (beats_per_bar, beat_value)
            key: Detected key (optional, shown in header)
            sections: Detected song sections (Intro, Verse, etc.)

        Returns:
            Formatted string like:
                [Intro]
                4/4 C Em | G B7 ||
        """
        if not chord_events:
            return "N.C."

        self.beats_per_bar = time_signature[0]
        self.beat_value = time_signature[1]

        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * self.beats_per_bar

        # Build output string
        lines = []

        # Header
        header_parts = [f"{time_signature[0]}/{time_signature[1]}"]
        if key:
            header_parts.append(f"Key: {key}")
        header_parts.append(f"BPM: {bpm:.0f}")
        lines.append("  ".join(header_parts))
        lines.append("")

        if sections:
            # Merge consecutive sections that share the same label so that
            # e.g. five separate [Chorus] blocks become one.
            merged_sections = [sections[0]]
            for sec in sections[1:]:
                if sec.label == merged_sections[-1].label:
                    prev = merged_sections[-1]
                    from src.features.structure_segmenter import SongSection
                    merged_sections[-1] = SongSection(
                        label=prev.label,
                        start_time=prev.start_time,
                        end_time=sec.end_time,
                        cluster_id=prev.cluster_id,
                    )
                else:
                    merged_sections.append(sec)

            # Format chords grouped by song section
            for section in merged_sections:
                # Get chord events that fall within this section,
                # excluding N.C. — gaps where nothing is played are silent
                section_chords = [
                    e for e in chord_events
                    if e.start_time >= section.start_time - 0.05
                    and e.start_time < section.end_time - 0.05
                    and e.symbol != "N.C."
                ]
                if not section_chords:
                    continue

                lines.append(f"[{section.label}]")
                section_lines = self._format_section_bars(
                    section_chords, bar_duration
                )
                lines.extend(section_lines)
                lines.append("")
        else:
            # No sections detected — flat format (drop N.C. events)
            non_nc = [e for e in chord_events if e.symbol != "N.C."]
            bars = self._assign_to_bars(non_nc, bar_duration)
            bar_strings = [s for s in (self._format_bar(bar) for bar in bars) if s]
            section_lines = self._render_bars(bar_strings)
            lines.extend(section_lines)

        # Clean up trailing blank lines
        while lines and lines[-1] == "":
            lines.pop()

        # Ensure the last line ends with ||
        if lines and not lines[-1].endswith("||"):
            if lines[-1].endswith("|"):
                lines[-1] = lines[-1][:-1] + "|"
            else:
                lines[-1] += " ||"

        return "\n".join(lines)

    def _format_section_bars(
        self,
        chord_events: list[ChordEvent],
        bar_duration: float,
    ) -> list[str]:
        """Format chord events within a section into bar lines."""
        bars = self._assign_to_bars(chord_events, bar_duration)
        bar_strings = [s for s in (self._format_bar(bar) for bar in bars) if s]
        return self._render_bars(bar_strings)

    def _render_bars(self, bar_strings: list[str]) -> list[str]:
        """Render a list of bar strings with repeat collapsing."""
        lines = []

        if self.simplify_repeats:
            collapsed = self._collapse_repeats(bar_strings)
        else:
            collapsed = [("plain", bar_strings, 1)]

        for section_type, section_bars, count in collapsed:
            if section_type == "repeat":
                inner = " | ".join(section_bars)
                lines.append(f"|: {inner} :| x{count}")
            else:
                bars_per_line = 4
                for i in range(0, len(section_bars), bars_per_line):
                    chunk = section_bars[i : i + bars_per_line]
                    lines.append(" | ".join(chunk) + " |")

        # Fix last line ending
        if lines and lines[-1].endswith(" |"):
            lines[-1] = lines[-1][:-2] + " ||"

        return lines

    def _assign_to_bars(
        self,
        chord_events: list[ChordEvent],
        bar_duration: float,
    ) -> list[list[ChordEvent]]:
        """Assign chord events to bars based on timing."""
        if not chord_events:
            return []

        total_duration = chord_events[-1].end_time
        n_bars = max(1, int(np.ceil(total_duration / bar_duration)))

        bars: list[list[ChordEvent]] = [[] for _ in range(n_bars)]

        for event in chord_events:
            bar_idx = int(event.start_time / bar_duration)
            bar_idx = min(bar_idx, n_bars - 1)
            event.bar_number = bar_idx
            event.beat_position = (
                (event.start_time % bar_duration)
                / (bar_duration / self.beats_per_bar)
            )
            bars[bar_idx].append(event)

        return bars

    def _collapse_repeats(
        self, bar_strings: list[str]
    ) -> list[tuple[str, list[str], int]]:
        """
        Collapse consecutive repeated bar patterns into sections.

        Returns a list of (section_type, bars, repeat_count) tuples:
          - ("repeat", ["C Em", "G B7"], 3)  -> |: C Em | G B7 :| x3
          - ("plain",  ["Am F"],          1)  -> Am F |

        Example:
            ["C Em", "G B7", "C Em", "G B7", "C Em", "G B7", "Am F"]
            -> [("repeat", ["C Em", "G B7"], 3), ("plain", ["Am F"], 1)]
        """
        sections: list[tuple[str, list[str], int]] = []
        plain_buf: list[str] = []
        i = 0
        n = len(bar_strings)

        while i < n:
            best_len = 0
            best_count = 1

            # Try pattern lengths from longest (4) to shortest (1)
            for pat_len in range(min(4, (n - i) // 2), 0, -1):
                pattern = bar_strings[i : i + pat_len]
                count = 1
                j = i + pat_len
                while j + pat_len <= n:
                    if bar_strings[j : j + pat_len] == pattern:
                        count += 1
                        j += pat_len
                    else:
                        break
                if count >= 2 and pat_len * count > best_len * best_count:
                    best_len = pat_len
                    best_count = count

            if best_count >= 2:
                # Flush any buffered plain bars first
                if plain_buf:
                    sections.append(("plain", plain_buf, 1))
                    plain_buf = []
                pattern = bar_strings[i : i + best_len]
                sections.append(("repeat", pattern, best_count))
                i += best_len * best_count
            else:
                plain_buf.append(bar_strings[i])
                i += 1

        # Flush remaining plain bars
        if plain_buf:
            sections.append(("plain", plain_buf, 1))

        return sections

    def _format_bar(self, bar_chords: list[ChordEvent]) -> str:
        """Format chords within a single bar. Returns empty string for silent bars."""
        if not bar_chords:
            return ""

        # Deduplicate consecutive identical chords
        unique = []
        for chord in bar_chords:
            if not unique or unique[-1].symbol != chord.symbol:
                unique.append(chord)

        symbols = [c.symbol for c in unique]
        return " ".join(symbols)

    def frames_to_chord_events(
        self,
        frame_labels: list[str],
        hop_length: int = 512,
        sample_rate: int = 44100,
    ) -> list[ChordEvent]:
        """
        Convert frame-level chord labels into ChordEvent list.
        Merges consecutive identical labels.
        """
        if not frame_labels:
            return []

        frame_duration = hop_length / sample_rate
        events = []
        current_symbol = frame_labels[0]
        start_frame = 0

        for i in range(1, len(frame_labels)):
            if frame_labels[i] != current_symbol:
                events.append(ChordEvent(
                    symbol=current_symbol,
                    start_time=start_frame * frame_duration,
                    end_time=i * frame_duration,
                ))
                current_symbol = frame_labels[i]
                start_frame = i

        # Final chord
        events.append(ChordEvent(
            symbol=current_symbol,
            start_time=start_frame * frame_duration,
            end_time=len(frame_labels) * frame_duration,
        ))

        # ── Minimum duration gate ─────────────────────────────────────────
        # A chord must last at least 0.5 s to be counted.  This allows
        # beat-level chord changes at any tempo down to ~120 BPM (0.5 s/beat)
        # while still filtering sub-beat strum-attack noise (< 0.1 s).
        # 1.0 s was too aggressive: at 68 BPM one beat = 0.88 s < 1.0 s,
        # causing every beat-level chord change to be absorbed.
        min_dur = 0.5
        merged: list[ChordEvent] = []
        for ev in events:
            dur = ev.end_time - ev.start_time
            if dur < min_dur and merged:
                # Absorb into the previous event
                merged[-1] = ChordEvent(
                    symbol    = merged[-1].symbol,
                    start_time= merged[-1].start_time,
                    end_time  = ev.end_time,
                )
            else:
                merged.append(ev)

        # Second pass: drop any remaining short events at the start
        events = [
            e for e in merged
            if (e.end_time - e.start_time) >= min_dur or e == merged[-1]
        ]

        return events

    def format_simple(self, chord_symbols: list[str]) -> str:
        """
        Simple formatting: just group chords into bars.

        Args:
            chord_symbols: List of chord symbols, one per beat.

        Returns:
            Formatted string.
        """
        bars = []
        for i in range(0, len(chord_symbols), self.beats_per_bar):
            bar = chord_symbols[i : i + self.beats_per_bar]
            # Deduplicate consecutive
            deduped = [bar[0]]
            for c in bar[1:]:
                if c != deduped[-1]:
                    deduped.append(c)
            bars.append(" ".join(deduped))

        return " | ".join(bars) + " ||"
