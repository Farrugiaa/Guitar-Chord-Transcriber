"""MIDI export for detected chord events — converts chord symbols to .mid files."""

from __future__ import annotations

import pretty_midi
import copy

# Standard tuning: E2 A2 D3 G3 B3 E4
_STANDARD_OPEN_MIDI = [40, 45, 50, 55, 59, 64]

# MIDI program 25 = Acoustic Guitar (steel strings), 0-indexed
_GUITAR_PROGRAM = 25


class MidiExporter:
    """Converts ChordEvent objects into a PrettyMIDI file using fret voicings."""

    def __init__(self, voicing_engine=None, tuning=None):
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
        """Build a PrettyMIDI object from a list of chord events."""
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

    @staticmethod
    def filter_pitch(
        pm: pretty_midi.PrettyMIDI,
        max_pitch: int = 84,
    ) -> pretty_midi.PrettyMIDI:
        """Return a copy of pm with notes above max_pitch (default C6) removed."""
        pm_f = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())
        pm_f.time_signature_changes = copy.deepcopy(pm.time_signature_changes)
        for inst in pm.instruments:
            inst_f = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name,
            )
            inst_f.notes = [
                copy.copy(n) for n in inst.notes if n.pitch <= max_pitch
            ]
            pm_f.instruments.append(inst_f)
        return pm_f

    @staticmethod
    def quantise_midi(
        pm: pretty_midi.PrettyMIDI,
        bpm: float,
        grid_beats: float = 0.25,
    ) -> pretty_midi.PrettyMIDI:
        """Return a quantised copy of pm — snaps start times to the nearest grid (default 16th note)."""
        beat_sec = 60.0 / max(bpm, 1.0)
        grid_sec = beat_sec * grid_beats

        pm_q = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        pm_q.time_signature_changes = copy.deepcopy(pm.time_signature_changes)

        for inst in pm.instruments:
            inst_q = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name,
            )
            for note in inst.notes:
                dur = note.end - note.start
                start_q = round(note.start / grid_sec) * grid_sec
                inst_q.notes.append(pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=float(start_q),
                    end=float(start_q + dur),
                ))
            inst_q.notes.sort(key=lambda n: n.start)
            pm_q.instruments.append(inst_q)

        return pm_q

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
