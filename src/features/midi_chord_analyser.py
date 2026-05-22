"""Detects chords from a polyphonic MIDI file using onset clustering."""

from __future__ import annotations

from dataclasses import dataclass, field

# Standard EADGBe open-string MIDI values
_STANDARD_OPEN_MIDI = [40, 45, 50, 55, 59, 64]
_MAX_FRET = 22


@dataclass
class MidiChordEvent:
    start_time: float
    end_time: float
    symbol: str
    note_count: int
    pitch_classes: list[int] = field(default_factory=list)


class MidiChordAnalyser:
    """Groups MIDI notes into chords using onset timing.

    A strum fires 2-6 notes within ~80ms. Single notes (riffs, melody)
    are thrown away. When a tuning is given, notes are mapped to actual
    guitar fret positions so unplayable notes get filtered out.
    """

    _DEFAULT_MAX_PITCH = 72  # C5 - drops basic-pitch overtone artefacts

    def __init__(
        self,
        cluster_window_sec: float = 0.08,
        max_pitch: int | None = _DEFAULT_MAX_PITCH,
        tuning=None,
    ):
        self.cluster_window_sec = cluster_window_sec
        self.max_pitch = max_pitch
        # Use detected tuning, fall back to standard EADGBe
        if tuning is not None and hasattr(tuning, "midi_notes"):
            self._open_midi = list(tuning.midi_notes)
        else:
            self._open_midi = list(_STANDARD_OPEN_MIDI)

    def _fret_filter(self, cluster: list) -> tuple[list[int], int | None]:
        """Map cluster notes to guitar strings, drop notes that can't be fretted.

        Returns (pitch_classes, root_pc). Root comes from the lowest guitar
        string with a note assigned — that's the bass string, so it's the root.
        """
        # Find the lowest fret on any string for each note.
        # One note per string max — you can't fret two notes on the same string.
        string_assignments: dict[int, tuple[int, int]] = {}  # string -> (fret, pitch)

        for note in cluster:
            best_string: int | None = None
            best_fret: int = _MAX_FRET + 1
            for s, open_note in enumerate(self._open_midi):
                fret = note.pitch - open_note
                if 0 <= fret <= _MAX_FRET and fret < best_fret:
                    best_fret = fret
                    best_string = s
            if best_string is not None:
                existing = string_assignments.get(best_string)
                if existing is None or best_fret < existing[0]:
                    string_assignments[best_string] = (best_fret, note.pitch)

        if not string_assignments:
            return [], None

        # Build pitch classes from lowest string upward (bass string = root)
        seen: set[int] = set()
        pitch_classes: list[int] = []
        onset_root_pc: int | None = None

        for s in sorted(string_assignments):
            _, pitch = string_assignments[s]
            pc = int(pitch % 12)
            if onset_root_pc is None:
                onset_root_pc = pc  # first = lowest string = root
            if pc not in seen:
                seen.add(pc)
                pitch_classes.append(pc)

        return pitch_classes, onset_root_pc

    def analyse(
        self,
        pm,
        bpm: float,
        key: str | None = None,
        scale_type: str | None = None,
    ) -> list[MidiChordEvent]:
        """Extract chord events from a PrettyMIDI object."""
        from src.theory.chord_builder import ChordBuilder
        from src.theory.scales import ScaleAnalyser

        # Collect non-drum notes, drop anything above the pitch ceiling
        all_notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)
        if self.max_pitch is not None:
            all_notes = [n for n in all_notes if n.pitch <= self.max_pitch]
        if not all_notes:
            return []
        all_notes.sort(key=lambda n: n.start)

        # Greedy clustering — new cluster whenever gap > 80ms
        clusters: list[list] = []
        current = [all_notes[0]]
        cluster_start = all_notes[0].start

        for note in all_notes[1:]:
            if note.start - cluster_start <= self.cluster_window_sec:
                current.append(note)
            else:
                clusters.append(current)
                current = [note]
                cluster_start = note.start
        clusters.append(current)

        # Discard clusters with fewer than 2 unique pitch classes (single notes / octaves)
        chord_clusters = [
            c for c in clusters
            if len({n.pitch % 12 for n in c}) >= 2
        ]
        if not chord_clusters:
            return []

        chord_builder = ChordBuilder()
        scale_pcs = None
        if key and scale_type:
            scale_pcs = set(ScaleAnalyser().get_scale_pitch_classes(key, scale_type))

        events: list[MidiChordEvent] = []
        for i, cluster in enumerate(chord_clusters):
            start = cluster[0].start
            max_end = max(n.end for n in cluster)

            # End at the next chord's start if it comes before the notes finish
            if i + 1 < len(chord_clusters):
                end = min(max_end, chord_clusters[i + 1][0].start)
            else:
                end = max_end

            # Map notes to fret positions — filters unplayable notes, sets root
            pitch_classes, onset_root_pc = self._fret_filter(cluster)

            if len(set(pitch_classes)) < 2:
                continue  # not enough distinct notes after fret filtering

            symbol = chord_builder.identify_chord_from_pitches(
                pitch_classes,
                key_context=key,
                scale_pcs=scale_pcs,
                onset_root_pc=onset_root_pc,
            ) or "N.C."

            events.append(MidiChordEvent(
                start_time=start,
                end_time=end,
                symbol=symbol,
                note_count=len(cluster),
                pitch_classes=pitch_classes,
            ))

        return events
