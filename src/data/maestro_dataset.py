"""
MAESTRO dataset loader.

Loads paired audio/MIDI piano data for pre-training chord recognition.
MIDI is converted to frame-level chord labels using pitch grouping.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pretty_midi


class MAESTRODataset(Dataset):
    """
    Dataset for MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization).

    Loads audio segments paired with MIDI-derived chord labels.
    Used for pre-training the chord recognition model on piano data.

    Expected directory structure:
        maestro_root/
            maestro-v3.0.0.csv   (or .json)
            2004/
                MIDI-Unprocessed_Chamber3_MID--AUDIO_...wav
                MIDI-Unprocessed_Chamber3_MID--AUDIO_...midi
            ...
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        segment_duration: float = 10.0,
        hop_length: int = 512,
        sample_rate: int = 44100,
        feature_extractor=None,
        chord_vocabulary=None,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.segment_duration = segment_duration
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.chord_vocabulary = chord_vocabulary
        self.transform = transform

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        """Load and filter MAESTRO metadata CSV."""
        csv_path = self.root_dir / "maestro-v3.0.0.csv"
        if not csv_path.exists():
            # Try JSON format
            json_path = self.root_dir / "maestro-v3.0.0.json"
            if json_path.exists():
                df = pd.read_json(json_path)
            else:
                raise FileNotFoundError(
                    f"MAESTRO metadata not found in {self.root_dir}. "
                    "Download from https://magenta.tensorflow.org/datasets/maestro"
                )
        else:
            df = pd.read_csv(csv_path)

        # Filter by split
        return df[df["split"] == self.split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        audio_path = self.root_dir / row["audio_filename"]
        midi_path = self.root_dir / row["midi_filename"]

        # Load audio features
        if self.feature_extractor:
            waveform, _ = self.feature_extractor.load_audio(str(audio_path))
            cqt = self.feature_extractor.compute_cqt(waveform)
        else:
            cqt = torch.zeros(1, 84, 100)  # placeholder

        # Load MIDI and extract chord labels
        chord_labels = self._midi_to_chord_labels(
            str(midi_path), n_frames=cqt.shape[-1]
        )

        sample = {
            "cqt": cqt,
            "chord_labels": chord_labels,
            "audio_path": str(audio_path),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _midi_to_chord_labels(
        self, midi_path: str, n_frames: int
    ) -> torch.Tensor:
        """
        Convert MIDI note events into frame-level chord labels.

        Groups simultaneously sounding notes into chords at each frame,
        then identifies chord symbols using the ChordBuilder.
        """
        from src.theory.chord_builder import ChordBuilder

        builder = ChordBuilder()
        frame_duration = self.hop_length / self.sample_rate
        labels = torch.zeros(n_frames, dtype=torch.long)

        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return labels

        for frame_idx in range(n_frames):
            t = frame_idx * frame_duration
            # Get all notes sounding at time t
            active_pitches = []
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    if note.start <= t < note.end:
                        active_pitches.append(note.pitch % 12)

            if not active_pitches:
                labels[frame_idx] = 0  # N.C.
                continue

            # Identify chord from active pitch classes
            chord_symbol = builder.identify_chord_from_pitches(
                list(set(active_pitches))
            )

            if self.chord_vocabulary:
                labels[frame_idx] = self.chord_vocabulary.encode(chord_symbol)

        return labels
