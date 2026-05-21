"""
MAPS dataset loader.

Loads piano audio with aligned note annotations for chord recognition
pre-training. Particularly useful for the UCHO (usual chords) subset.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MAPSDataset(Dataset):
    """
    Dataset for MAPS (MIDI Aligned Piano Sounds).

    Loads audio segments with note-level annotations.
    Subsets: ISOL (isolated notes), RAND (random chords),
    UCHO (usual chords), MUS (full pieces).

    Expected directory structure:
        maps_root/
            ENSTDkAm/
                UCHO/
                    MAPS_UCHO-*.wav
                    MAPS_UCHO-*.txt
                MUS/
                    ...
            SptkBGCl/
                ...
    """

    def __init__(
        self,
        root_dir: str,
        subset: str = "UCHO",
        piano_types: list[str] | None = None,
        segment_duration: float = 10.0,
        hop_length: int = 512,
        sample_rate: int = 44100,
        feature_extractor=None,
        chord_vocabulary=None,
    ):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.segment_duration = segment_duration
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.chord_vocabulary = chord_vocabulary

        if piano_types is None:
            piano_types = ["ENSTDkAm", "ENSTDkCl"]

        self.file_pairs = self._find_files(piano_types)

    def _find_files(self, piano_types: list[str]) -> list[tuple[Path, Path]]:
        """Find all (audio, annotation) pairs for the given subset."""
        pairs = []
        for ptype in piano_types:
            subset_dir = self.root_dir / ptype / self.subset
            if not subset_dir.exists():
                continue
            for wav_file in sorted(subset_dir.glob("*.wav")):
                txt_file = wav_file.with_suffix(".txt")
                if txt_file.exists():
                    pairs.append((wav_file, txt_file))
        return pairs

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> dict:
        audio_path, annot_path = self.file_pairs[idx]

        # Load audio features
        if self.feature_extractor:
            waveform, _ = self.feature_extractor.load_audio(str(audio_path))
            cqt = self.feature_extractor.compute_cqt(waveform)
        else:
            cqt = torch.zeros(1, 84, 100)

        # Parse annotations -> frame-level labels
        notes = self._parse_annotations(str(annot_path))
        chord_labels = self._notes_to_chord_labels(notes, n_frames=cqt.shape[-1])

        return {
            "cqt": cqt,
            "chord_labels": chord_labels,
            "audio_path": str(audio_path),
        }

    def _parse_annotations(self, txt_path: str) -> list[dict]:
        """
        Parse MAPS annotation file.
        Format: onset_time\toffset_time\tmidi_pitch
        """
        notes = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    notes.append({
                        "onset": float(parts[0]),
                        "offset": float(parts[1]),
                        "pitch": int(parts[2]),
                    })
        return notes

    def _notes_to_chord_labels(
        self, notes: list[dict], n_frames: int
    ) -> torch.Tensor:
        """Convert note events to frame-level chord labels."""
        from src.theory.chord_builder import ChordBuilder

        builder = ChordBuilder()
        frame_duration = self.hop_length / self.sample_rate
        labels = torch.zeros(n_frames, dtype=torch.long)

        for frame_idx in range(n_frames):
            t = frame_idx * frame_duration
            active_pitches = []
            for note in notes:
                if note["onset"] <= t < note["offset"]:
                    active_pitches.append(note["pitch"] % 12)

            if not active_pitches:
                continue

            chord_symbol = builder.identify_chord_from_pitches(
                list(set(active_pitches))
            )
            if self.chord_vocabulary:
                labels[frame_idx] = self.chord_vocabulary.encode(chord_symbol)

        return labels
