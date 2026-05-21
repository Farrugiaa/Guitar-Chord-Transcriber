"""
DadaGP dataset loader.

Loads GuitarPro tablature files and extracts guitar chord voicings
for training the chord recognition model on guitar-specific data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DadaGPDataset(Dataset):
    """
    Dataset for DadaGP (guitar tablature dataset).

    Parses GuitarPro files to extract guitar chord voicings,
    timing, and metadata (tempo, time signature, key).

    Expected directory structure:
        dadagp_root/
            tokens/          # pre-tokenised files (if available)
            gp_files/        # raw GuitarPro files
                song1.gp5
                song2.gpx
                ...
    """

    def __init__(
        self,
        root_dir: str,
        use_tokens: bool = True,
        chord_vocabulary=None,
        max_sequence_length: int = 512,
    ):
        self.root_dir = Path(root_dir)
        self.use_tokens = use_tokens
        self.chord_vocabulary = chord_vocabulary
        self.max_sequence_length = max_sequence_length

        if use_tokens:
            self.files = sorted(
                (self.root_dir / "tokens").glob("*.txt")
            ) if (self.root_dir / "tokens").exists() else []
        else:
            self.files = sorted(self._find_gp_files())

    def _find_gp_files(self) -> list[Path]:
        """Find all GuitarPro files."""
        extensions = ["*.gp3", "*.gp4", "*.gp5", "*.gpx", "*.gp"]
        files = []
        gp_dir = self.root_dir / "gp_files"
        if gp_dir.exists():
            for ext in extensions:
                files.extend(gp_dir.glob(ext))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        filepath = self.files[idx]

        if self.use_tokens:
            return self._load_token_file(filepath)
        else:
            return self._load_gp_file(filepath)

    def _load_token_file(self, path: Path) -> dict:
        """Load pre-tokenised DadaGP file."""
        with open(path, "r") as f:
            tokens = f.read().strip().split("\n")

        chords = self._extract_chords_from_tokens(tokens)
        return {
            "chords": chords,
            "filepath": str(path),
        }

    def _load_gp_file(self, path: Path) -> dict:
        """Load and parse a GuitarPro file."""
        try:
            import guitarpro
            song = guitarpro.parse(str(path))
        except Exception as e:
            return {"chords": [], "metadata": {}, "filepath": str(path), "error": str(e)}

        metadata = {
            "title": song.title,
            "artist": song.artist,
            "tempo": song.tempo.value if hasattr(song.tempo, 'value') else song.tempo,
        }

        chords = self._extract_chords_from_gp(song)

        return {
            "chords": chords,
            "metadata": metadata,
            "filepath": str(path),
        }

    def _extract_chords_from_gp(self, song) -> list[dict]:
        """
        Extract chord information from a GuitarPro song object.
        Finds guitar tracks and extracts simultaneous notes as chords.
        """
        import guitarpro

        chords = []

        for track in song.tracks:
            # Focus on guitar tracks
            if not self._is_guitar_track(track):
                continue

            current_time = 0.0
            tempo = song.tempo if isinstance(song.tempo, (int, float)) else 120

            for measure in track.measures:
                for voice in measure.voices:
                    for beat in voice.beats:
                        if isinstance(beat.notes, list) and len(beat.notes) > 0:
                            # Extract pitch classes from the beat's notes
                            pitches = []
                            string_frets = []
                            for note in beat.notes:
                                if hasattr(note, 'value') and hasattr(note, 'string'):
                                    midi_pitch = self._string_fret_to_midi(
                                        note.string, note.value, track
                                    )
                                    pitches.append(midi_pitch % 12)
                                    string_frets.append(
                                        (note.string, note.value)
                                    )

                            if pitches:
                                beat_duration = self._beat_duration_seconds(
                                    beat, tempo
                                )
                                chords.append({
                                    "time": current_time,
                                    "duration": beat_duration,
                                    "pitch_classes": pitches,
                                    "string_frets": string_frets,
                                })
                                current_time += beat_duration

        return chords

    def _is_guitar_track(self, track) -> bool:
        """Check if a track is a guitar track based on channel/name."""
        guitar_keywords = ["guitar", "gtr", "guit", "acoustic", "electric"]
        name_lower = track.name.lower() if track.name else ""
        for kw in guitar_keywords:
            if kw in name_lower:
                return True
        # Check MIDI program (25-32 are guitar programs)
        if hasattr(track, 'channel') and hasattr(track.channel, 'instrument'):
            prog = track.channel.instrument
            if 24 <= prog <= 31:
                return True
        return False

    def _string_fret_to_midi(self, string_num: int, fret: int, track) -> int:
        """Convert string number + fret to MIDI pitch using track tuning."""
        if hasattr(track, 'strings') and string_num <= len(track.strings):
            open_pitch = track.strings[string_num - 1].value
            return open_pitch + fret
        # Standard tuning fallback: E2 B2 G3 D3 A2 E4
        standard = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
        base = standard.get(string_num, 40)
        return base + fret

    @staticmethod
    def _beat_duration_seconds(beat, tempo: float) -> float:
        """Calculate beat duration in seconds."""
        # GuitarPro beat.duration.value: 1=whole, 2=half, 4=quarter, etc.
        if hasattr(beat, 'duration') and hasattr(beat.duration, 'value'):
            quarter_duration = 60.0 / tempo
            return quarter_duration * (4.0 / beat.duration.value)
        return 60.0 / tempo  # default to quarter note

    def _extract_chords_from_tokens(self, tokens: list[str]) -> list[dict]:
        """Parse DadaGP token format into chord events."""
        chords = []
        current_notes = []
        current_time = 0.0

        for token in tokens:
            token = token.strip()
            if token.startswith("note:"):
                parts = token.split(":")
                if len(parts) >= 3:
                    string_num = int(parts[1])
                    fret = int(parts[2])
                    # Standard tuning MIDI pitches for open strings
                    standard = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
                    midi = standard.get(string_num, 40) + fret
                    current_notes.append(midi % 12)
            elif token.startswith("wait:"):
                if current_notes:
                    chords.append({
                        "time": current_time,
                        "pitch_classes": list(set(current_notes)),
                    })
                    current_notes = []
                try:
                    wait_val = float(token.split(":")[1])
                    current_time += wait_val
                except (ValueError, IndexError):
                    pass

        # Flush remaining
        if current_notes:
            chords.append({
                "time": current_time,
                "pitch_classes": list(set(current_notes)),
            })

        return chords
