"""
GuitarSet dataset loader.

GuitarSet provides 360 recordings of isolated guitar with rich annotations:
  - Audio: mic + pickup recordings (WAV, 44.1 kHz)
  - JAMS annotations: chord labels, beat/downbeat times, key, BPM,
    note-level pitch contours (hex-pickup per string)

Three uses in this project:
  1. ChordRecognitionDataset  — CQT + frame-level chord labels
     → fine-tunes the CRNN chord recogniser on real guitar audio
  2. TuningDataset            — short onset windows + known tuning label
     → trains / evaluates the TuningDetector
  3. SourceSeparationDataset  — isolated guitar as target for U-Net training
     → the mic track is the "clean guitar" target; a synthetic mix is made
       by summing with noise/other stems

Dataset download
----------------
The dataset is available from Zenodo:
  https://zenodo.org/record/3371780

Expected directory structure:
    guitarset_root/
        audio_mono-mic/          # mono microphone recordings
            00_BN1-129-Eb_comp.wav
            ...
        audio_mono-pickup_mix/   # hex-pickup mono-mix recordings
            00_BN1-129-Eb_comp.wav
            ...
        annotation/              # JAMS annotation files
            00_BN1-129-Eb_comp.jams
            ...

JAMS format note
----------------
Install jams via:  pip install jams
Each .jams file contains multiple annotation namespaces; we use:
  - chord_harte  → chord labels with time intervals
  - beat_jams    → beat positions
  - key_mode     → key + mode
  - tempo        → BPM
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Map from Harte chord quality to our internal vocabulary labels
# (mirrors ChordVocabulary.CHORD_QUALITIES in chord_recogniser.py)
_HARTE_QUALITY_MAP = {
    "maj":   "",          # plain major
    "min":   "m",
    "dom":   "7",
    "maj7":  "maj7",
    "min7":  "m7",
    "dim":   "dim",
    "dim7":  "dim7",
    "hdim7": "m7b5",
    "aug":   "aug",
    "sus2":  "sus2",
    "sus4":  "sus4",
    "":      "",          # Harte uses empty string for major
}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]


def _harte_to_symbol(harte_chord: str) -> str:
    """
    Convert a Harte chord label (e.g. 'C:maj', 'A:min7', 'N') to our
    internal chord symbol format (e.g. 'C', 'Am7', 'N.C.').
    """
    if harte_chord in ("N", "X", "N.C."):
        return "N.C."

    if ":" not in harte_chord:
        # Bare root = major
        return harte_chord.replace("b", "b").replace("#", "#")

    root, quality_str = harte_chord.split(":", 1)

    # Strip bass note if present (e.g. 'C:maj/5' → quality='maj')
    quality_str = quality_str.split("/")[0]

    # Strip shorthand extensions in parentheses (e.g. 'maj(9)' → 'maj')
    if "(" in quality_str:
        quality_str = quality_str[: quality_str.index("(")]

    suffix = _HARTE_QUALITY_MAP.get(quality_str, quality_str)

    # Normalise root (Harte uses 'b' for flat, '#' for sharp)
    root = root.replace("b", "b")   # already correct
    if suffix == "":
        return root
    if suffix.startswith("m") or suffix in ("dim", "dim7", "aug", "sus2", "sus4"):
        return f"{root}{suffix}"
    return f"{root}{suffix}"


class GuitarSetChordDataset(Dataset):
    """
    CQT spectrogram segments paired with frame-level chord labels.

    Each item:
        cqt   : torch.Tensor (1, n_bins, n_frames)  — log-CQT segment
        label : torch.Tensor (n_frames,)             — chord index per frame

    Args:
        root_dir:          Path to GuitarSet root directory.
        split:             'train', 'val', or 'test'
                           (uses player IDs 00-04 for train, 05 for test,
                            random 10 % of train for val).
        segment_duration:  Length of each training segment in seconds.
        hop_length:        STFT hop length in samples.
        sample_rate:       Target sample rate (audio is resampled if needed).
        chord_vocabulary:  ChordVocabulary instance for encoding labels.
        use_mic:           If True use mic recording, else pickup-mix.
        augment:           If True apply pitch-shift augmentation ±1 semitone.
    """

    # GuitarSet player IDs for each split
    _TRAIN_PLAYERS = {"00", "01", "02", "03", "04"}
    _TEST_PLAYERS  = {"05"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        segment_duration: float = 10.0,
        hop_length: int = 512,
        sample_rate: int = 44100,
        chord_vocabulary=None,
        use_mic: bool = True,
        augment: bool = False,
    ):
        self.root          = Path(root_dir)
        self.split         = split
        self.seg_dur       = segment_duration
        self.hop_length    = hop_length
        self.sample_rate   = sample_rate
        self.augment       = augment and (split == "train")

        from src.models.chord_recogniser import ChordVocabulary
        self.vocab = chord_vocabulary or ChordVocabulary(level="extended")

        audio_dir = "audio_mono-mic" if use_mic else "audio_mono-pickup_mix"
        self.audio_dir = self.root / audio_dir
        self.annot_dir = self.root / "annotation"

        self._segments: list[tuple[Path, Path, float]] = []
        self._build_index()

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self):
        """Enumerate all (audio, jams, start_time) segments."""
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"GuitarSet audio directory not found: {self.audio_dir}\n"
                f"Download from https://zenodo.org/record/3371780"
            )

        train_files, test_files = [], []
        for wav in sorted(self.audio_dir.glob("*.wav")):
            player_id = wav.stem.split("_")[0][:2]
            jams_path = self.annot_dir / (wav.stem + ".jams")
            if not jams_path.exists():
                continue
            if player_id in self._TEST_PLAYERS:
                test_files.append((wav, jams_path))
            else:
                train_files.append((wav, jams_path))

        # 10 % of train → val
        n_val = max(1, len(train_files) // 10)
        if self.split == "test":
            chosen = test_files
        elif self.split == "val":
            chosen = train_files[-n_val:]
        else:
            chosen = train_files[:-n_val]

        seg_samples = int(self.seg_dur * self.sample_rate)

        for wav, jams in chosen:
            import soundfile as sf
            info   = sf.info(str(wav))
            n_samp = info.frames
            if info.samplerate != self.sample_rate:
                # Recalculate using native sr
                seg_native = int(self.seg_dur * info.samplerate)
                starts = range(0, n_samp - seg_native, seg_native)
                for s in starts:
                    start_sec = s / info.samplerate
                    self._segments.append((wav, jams, start_sec))
            else:
                starts = range(0, n_samp - seg_samples, seg_samples)
                for s in starts:
                    self._segments.append((wav, jams, s / self.sample_rate))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> dict:
        wav_path, jams_path, start_sec = self._segments[idx]

        # Load audio segment
        audio = self._load_audio(wav_path, start_sec, self.seg_dur)

        # Optional pitch-shift augmentation
        if self.augment:
            semitones = np.random.choice([-1, 0, 0, 0, 1])
            if semitones != 0:
                import librosa
                audio = librosa.effects.pitch_shift(
                    audio, sr=self.sample_rate, n_steps=semitones
                )

        # CQT features
        cqt = self._compute_cqt(audio)

        # Chord labels
        labels = self._load_chord_labels(jams_path, start_sec, cqt.shape[-1])

        return {
            "cqt":        cqt,                          # (1, bins, frames)
            "labels":     torch.tensor(labels, dtype=torch.long),
            "file":       wav_path.stem,
            "start_sec":  start_sec,
        }

    # ── Audio loading ─────────────────────────────────────────────────────────

    def _load_audio(
        self, path: Path, start_sec: float, duration: float
    ) -> np.ndarray:
        import soundfile as sf
        import librosa

        info       = sf.info(str(path))
        start_samp = int(start_sec * info.samplerate)
        n_samp     = int(duration  * info.samplerate)

        data, sr = sf.read(
            str(path), start=start_samp, frames=n_samp, dtype="float32"
        )
        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != self.sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)

        # Pad if shorter than requested
        target = int(duration * self.sample_rate)
        if len(data) < target:
            data = np.pad(data, (0, target - len(data)))

        return data[:target]

    # ── Feature extraction ────────────────────────────────────────────────────

    def _compute_cqt(self, audio: np.ndarray) -> torch.Tensor:
        import librosa
        cqt = np.abs(librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=84,
            bins_per_octave=12,
            fmin=32.7,
        ))
        log_cqt = np.log(cqt + 1e-8)
        return torch.from_numpy(log_cqt).unsqueeze(0).float()  # (1, bins, T)

    # ── Chord label loading ───────────────────────────────────────────────────

    def _load_chord_labels(
        self,
        jams_path: Path,
        start_sec: float,
        n_frames: int,
    ) -> list[int]:
        """
        Read Harte chord annotations from JAMS and produce a
        frame-level integer label array.
        """
        try:
            import jams
        except ImportError:
            raise ImportError(
                "jams package is required for GuitarSet.\n"
                "Install with: pip install jams"
            )

        jam         = jams.load(str(jams_path))
        frame_dur   = self.hop_length / self.sample_rate
        labels      = [self.vocab.encode("N.C.")] * n_frames

        chord_anns = jam.search(namespace="chord")
        if not chord_anns:
            chord_anns = jam.search(namespace="chord_harte")
        if not chord_anns:
            return labels

        ann = chord_anns[0]
        for obs in ann.data:
            chord_start = float(obs.time.total_seconds())
            chord_end   = chord_start + float(obs.duration.total_seconds())
            chord_value = obs.value

            # Clip to segment window
            seg_end = start_sec + n_frames * frame_dur
            if chord_end < start_sec or chord_start > seg_end:
                continue

            symbol = _harte_to_symbol(chord_value)
            idx    = self.vocab.encode(symbol)

            frame_start = max(0, int((chord_start - start_sec) / frame_dur))
            frame_end   = min(n_frames, int((chord_end   - start_sec) / frame_dur))
            for f in range(frame_start, frame_end):
                labels[f] = idx

        return labels


class GuitarSetTuningDataset(Dataset):
    """
    Short onset windows with tuning labels for training/evaluating
    the TuningDetector.

    GuitarSet recordings are all in standard tuning, but the dataset
    is useful for:
      - Confirming the detector returns 'standard' for standard recordings
      - Providing real isolated guitar windows as positive examples for
        the pyin-based detector

    Each item:
        audio:  np.ndarray (n_samples,) — 150 ms onset window
        tuning: str — tuning key (always 'standard' for GuitarSet)
    """

    def __init__(self, root_dir: str, use_mic: bool = True):
        self.root      = Path(root_dir)
        audio_dir      = "audio_mono-mic" if use_mic else "audio_mono-pickup_mix"
        self.audio_dir = self.root / audio_dir
        self.sample_rate = 44100
        self._windows: list[tuple[Path, float]] = []
        self._build_index()

    def _build_index(self):
        import soundfile as sf
        import librosa

        for wav in sorted(self.audio_dir.glob("*.wav")):
            info = sf.info(str(wav))
            dur  = info.frames / info.samplerate
            # Sample an onset window every 2 seconds
            for t in np.arange(0.5, dur - 0.5, 2.0):
                self._windows.append((wav, float(t)))

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        wav_path, start_sec = self._windows[idx]
        import soundfile as sf
        import librosa

        info       = sf.info(str(wav_path))
        start_samp = int(start_sec * info.samplerate)
        n_samp     = int(0.15 * info.samplerate)

        data, sr = sf.read(
            str(wav_path), start=start_samp, frames=n_samp, dtype="float32"
        )
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)

        target = int(0.15 * self.sample_rate)
        if len(data) < target:
            data = np.pad(data, (0, target - len(data)))

        return {"audio": data[:target], "tuning": "standard"}


class GuitarSetSeparationDataset(Dataset):
    """
    Isolated guitar recordings for training the U-Net source separator.

    The mic recording is used as the clean guitar target.
    A synthetic mixture is made by adding freely-available noise stems
    (or by mixing with other GuitarSet tracks to simulate a band mix).

    Each item:
        mixture:  torch.Tensor (1, n_samples) — synthetic mixed signal
        target:   torch.Tensor (1, n_samples) — clean guitar (mic)
    """

    def __init__(
        self,
        root_dir: str,
        segment_duration: float = 5.0,
        sample_rate: int = 44100,
        noise_dir: Optional[str] = None,
    ):
        """
        Args:
            root_dir:         GuitarSet root directory.
            segment_duration: Length of each segment in seconds.
            sample_rate:      Target sample rate.
            noise_dir:        Optional path to a directory of noise WAV files
                              (drums, room ambience, etc.) to mix with guitar.
                              If None, white noise is used as a simple surrogate.
        """
        self.root        = Path(root_dir)
        self.seg_dur     = segment_duration
        self.sample_rate = sample_rate
        self.noise_dir   = Path(noise_dir) if noise_dir else None

        self.mic_dir = self.root / "audio_mono-mic"
        self._segments: list[tuple[Path, float]] = []
        self._build_index()

    def _build_index(self):
        import soundfile as sf

        for wav in sorted(self.mic_dir.glob("*.wav")):
            info     = sf.info(str(wav))
            dur      = info.frames / info.samplerate
            seg_samp = int(self.seg_dur * self.sample_rate)
            for s in range(0, int(dur * self.sample_rate) - seg_samp, seg_samp):
                self._segments.append((wav, s / self.sample_rate))

    def __len__(self):
        return len(self._segments)

    def __getitem__(self, idx: int) -> dict:
        wav_path, start_sec = self._segments[idx]
        guitar = self._load_audio(wav_path, start_sec)
        noise  = self._load_noise(len(guitar))

        # Mix at a random SNR between 0 and 10 dB
        snr_db    = np.random.uniform(0, 10)
        snr_lin   = 10 ** (snr_db / 20)
        g_rms     = np.sqrt((guitar ** 2).mean() + 1e-8)
        n_rms     = np.sqrt((noise  ** 2).mean() + 1e-8)
        noise_scaled = noise * (g_rms / (n_rms * snr_lin + 1e-8))
        mixture   = guitar + noise_scaled

        # Normalise
        peak = np.abs(mixture).max()
        if peak > 1.0:
            mixture = mixture / peak
            guitar  = guitar  / peak

        return {
            "mixture": torch.from_numpy(mixture).unsqueeze(0).float(),
            "target":  torch.from_numpy(guitar).unsqueeze(0).float(),
        }

    def _load_audio(self, path: Path, start_sec: float) -> np.ndarray:
        import soundfile as sf
        import librosa

        info       = sf.info(str(path))
        start_samp = int(start_sec * info.samplerate)
        n_samp     = int(self.seg_dur * info.samplerate)

        data, sr = sf.read(
            str(path), start=start_samp, frames=n_samp, dtype="float32"
        )
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)

        target = int(self.seg_dur * self.sample_rate)
        if len(data) < target:
            data = np.pad(data, (0, target - len(data)))
        return data[:target].astype(np.float32)

    def _load_noise(self, n_samples: int) -> np.ndarray:
        """Load a random noise clip or generate white noise."""
        if self.noise_dir and self.noise_dir.exists():
            import soundfile as sf
            import librosa

            noise_files = list(self.noise_dir.glob("*.wav"))
            if noise_files:
                chosen = noise_files[np.random.randint(len(noise_files))]
                info   = sf.info(str(chosen))
                dur    = info.frames / info.samplerate
                max_start = max(0.0, dur - self.seg_dur)
                start_sec = np.random.uniform(0, max_start)

                data, sr = sf.read(
                    str(chosen),
                    start=int(start_sec * info.samplerate),
                    frames=int(self.seg_dur * info.samplerate),
                    dtype="float32",
                )
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != self.sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
                if len(data) < n_samples:
                    data = np.pad(data, (0, n_samples - len(data)))
                return data[:n_samples].astype(np.float32)

        # Fallback: shaped white noise
        noise = np.random.randn(n_samples).astype(np.float32) * 0.1
        return noise
