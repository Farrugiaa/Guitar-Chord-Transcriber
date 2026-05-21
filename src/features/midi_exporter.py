"""
MIDI export for detected chord events.

Converts the chord progression into a standard MIDI file (.mid) so the
output can be imported into any DAW (Logic, Reaper, GarageBand, etc.) for
verification — checking for skipped chords, wrong labels, or timing drift.

Each chord event becomes simultaneous notes based on the voicing (fret
positions on each string), so the MIDI reflects exactly which pitches the
chord algorithm chose, not just abstract chord symbols.

Requires: pretty_midi  (already in requirements.txt)
"""

from __future__ import annotations

import pretty_midi

# Standard tuning: E2 A2 D3 G3 B3 E4
_STANDARD_OPEN_MIDI = [40, 45, 50, 55, 59, 64]

# MIDI program 25 = Acoustic Guitar (steel strings), 0-indexed
_GUITAR_PROGRAM = 25


class MidiExporter:
    """
    Convert ChordEvent objects into a pretty_midi.PrettyMIDI file.

    Uses the chord voicing engine to resolve each chord symbol to fret
    positions, then maps those to MIDI pitches using the detected tuning's
    open-string MIDI values.
    """

    def __init__(self, voicing_engine=None, tuning=None):
        """
        Args:
            voicing_engine: ChordVoicingEngine instance (from src.theory.chord_voicings).
            tuning:         Tuning dataclass (from src.theory.tuning).
                            None → standard EADGBe tuning.
        """
        self._voicing_engine = voicing_engine
        if tuning is not None and hasattr(tuning, "midi_notes"):
            self._open_midi = list(tuning.midi_notes)
        else:
            self._open_midi = list(_STANDARD_OPEN_MIDI)

    def export(
        self,
        chord_events: list,
        bpm: float,
        time_signature: tuple[int, int] = (4, 4),
        velocity: int = 80,
    ) -> pretty_midi.PrettyMIDI:
        """
        Build and return a PrettyMIDI object from a list of chord events.

        Args:
            chord_events:   list of ChordEvent (symbol, start_time, end_time)
            bpm:            tempo in beats-per-minute
            time_signature: (numerator, denominator) — e.g. (4, 4)
            velocity:       MIDI velocity for all notes (0–127)

        Returns:
            pretty_midi.PrettyMIDI — call `.write(path)` to save as .mid
        """
        pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        pm.time_signature_changes = [
            pretty_midi.TimeSignature(time_signature[0], time_signature[1], 0.0)
        ]

        guitar = pretty_midi.Instrument(program=_GUITAR_PROGRAM, name="Guitar")

        for event in chord_events:
            if event.symbol in ("N.C.", ""):
                continue
            notes = self._chord_to_notes(
                event.symbol, event.start_time, event.end_time, velocity
            )
            guitar.notes.extend(notes)

        pm.instruments.append(guitar)
        return pm

    def _chord_to_notes(
        self,
        symbol: str,
        start: float,
        end: float,
        velocity: int,
    ) -> list[pretty_midi.Note]:
        """Resolve a chord symbol to MIDI notes via the voicing engine."""
        if self._voicing_engine is None:
            return []

        voicing = self._voicing_engine.get_voicing(symbol)
        if voicing is None:
            return []

        notes = []
        for string_idx, fret in enumerate(voicing):
            if fret < 0:                    # muted string
                continue
            pitch = self._open_midi[string_idx] + fret
            if 0 <= pitch <= 127:
                notes.append(pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=float(start),
                    end=float(max(end, start + 0.05)),   # 50 ms minimum
                ))
        return notes
