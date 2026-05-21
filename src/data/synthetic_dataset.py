"""
Synthetic guitar chord dataset using Karplus-Strong string synthesis.

Generates realistic plucked guitar chord audio programmatically — no audio
files or external dependencies beyond numpy are required.

Covers the full chord vocabulary (up to 170 classes) with multiple strum
style and acoustic variations per chord, giving the model guitar-specific
harmonic examples with perfect ground-truth labels.

Intended role in the training pipeline
────────────────────────────────────────
    Phase 1   : Pre-train on piano data  (MAESTRO / MAPS)
    Phase 1b  : Train on synthetic guitar chords  ← this dataset
    Phase 2   : Fine-tune on real guitar audio  (GuitarSet / DadaGP)

The synthetic-guitar step bridges the timbre gap between piano pre-training
and real guitar fine-tuning.  Because the model later sees real recordings,
the slight unrealism of synthesised tones is corrected during fine-tuning.

Karplus-Strong algorithm
────────────────────────
Initialise a circular delay-line buffer of length ≈ sr/f0 with white noise.
At each output sample:
    output[n]  = buf[head]
    buf[head]  = 0.5 × (buf[head] + buf[head+1]) × decay
    head       = (head + 1) mod buf_size

The low-pass averaging models energy dissipation in a vibrating string and
produces a convincing plucked tone that decays naturally.  Increasing `decay`
(towards 1.0) gives more sustain; decreasing it gives a more muted, dampened
sound.

Output format (matches GuitarSetChordDataset)
─────────────────────────────────────────────
Each __getitem__ returns a dict:
    "cqt"    : torch.Tensor  (1, 84, n_frames)   log-CQT spectrogram
    "labels" : torch.Tensor  (n_frames,)          chord index (constant)
    "chord"  : str                                chord symbol
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Standard tuning MIDI note numbers for each open string (low→high) ────────
_OPEN_MIDI = [40, 45, 50, 55, 59, 64]   # E2 A2 D3 G3 B3 E4

# ── Strum styles ──────────────────────────────────────────────────────────────
STRUM_STYLES = ("block", "down", "up", "arpeggio")


# ── Karplus-Strong synthesis ──────────────────────────────────────────────────

def _karplus_strong(
    freq: float,
    sr: int,
    duration: float,
    decay: float = 0.996,
    noise_amp: float = 1.0,
) -> np.ndarray:
    """
    Synthesise one plucked string tone.

    Args:
        freq:      Fundamental frequency in Hz.
        sr:        Sample rate.
        duration:  Output duration in seconds.
        decay:     Per-sample energy retention (0.99–0.999 for guitar).
        noise_amp: Initial excitation amplitude.

    Returns:
        1-D float32 array of length int(sr * duration).
    """
    n_samples = int(sr * duration)
    buf_size  = max(2, round(sr / freq))

    # Initialise buffer with band-limited noise
    buf = (np.random.randn(buf_size) * noise_amp).astype(np.float64)

    output = np.zeros(n_samples, dtype=np.float64)
    idx    = 0
    for i in range(n_samples):
        output[i] = buf[idx]
        next_idx  = (idx + 1) % buf_size
        buf[idx]  = 0.5 * (buf[idx] + buf[next_idx]) * decay
        idx       = next_idx

    return output.astype(np.float32)


def _synthesise_chord(
    freqs: list[float],
    sr: int,
    duration: float,
    style: str = "down",
    strum_ms: float = 20.0,
    decay: float = 0.996,
    velocity_jitter: float = 0.15,
) -> np.ndarray:
    """
    Combine multiple string tones into a strummed or arpeggiated chord.

    Args:
        freqs:          List of note frequencies to play (muted strings excluded).
        sr:             Sample rate.
        duration:       Total clip length in seconds.
        style:          One of "block", "down", "up", "arpeggio".
        strum_ms:       Time between successive strings in ms (for non-block styles).
        decay:          Karplus-Strong decay factor.
        velocity_jitter: ±fraction randomness applied to each string amplitude.

    Returns:
        Normalised float32 audio array.
    """
    n_total = int(sr * duration)
    mix     = np.zeros(n_total, dtype=np.float32)

    if not freqs:
        return mix

    n_strings  = len(freqs)
    strum_samp = int(strum_ms * sr / 1000)

    # Determine per-string onset offsets (in samples)
    if style == "block":
        offsets = [0] * n_strings
    elif style == "down":
        # Low string first → high string last
        offsets = [i * strum_samp for i in range(n_strings)]
    elif style == "up":
        # High string first → low string last
        offsets = [(n_strings - 1 - i) * strum_samp for i in range(n_strings)]
    elif style == "arpeggio":
        # Wider spacing — strings plucked individually
        offsets = [i * strum_samp * 4 for i in range(n_strings)]
    else:
        offsets = [0] * n_strings

    for i, (freq, onset) in enumerate(zip(freqs, offsets)):
        amp     = 1.0 + np.random.uniform(-velocity_jitter, velocity_jitter)
        tone    = _karplus_strong(freq, sr, duration, decay=decay, noise_amp=amp)
        end     = min(onset + len(tone), n_total)
        mix[onset:end] += tone[:end - onset]

    # Normalise to peak ≈ 0.9
    peak = np.abs(mix).max()
    if peak > 1e-6:
        mix = mix * (0.9 / peak)

    return mix


def _chord_to_freqs(chord_symbol: str) -> list[float]:
    """
    Return a list of note frequencies for a chord, in low-to-high string order.

    Prefers the ChordVoicingEngine lookup/algorithmic voicing (which gives an
    actual playable guitar shape); falls back to distributing the chord's pitch
    classes across the guitar range when no voicing is available.
    """
    try:
        from src.theory.chord_voicings import ChordVoicingEngine
        engine  = ChordVoicingEngine()
        voicing = engine.get_voicing(chord_symbol)

        if voicing is not None:
            freqs = []
            for string_idx, fret in enumerate(voicing):
                if fret < 0:   # muted
                    continue
                midi = _OPEN_MIDI[string_idx] + fret
                freqs.append(440.0 * 2.0 ** ((midi - 69) / 12.0))
            if freqs:
                return freqs
    except Exception:
        pass

    # ── Fallback: distribute pitch classes across the guitar range ────────────
    # Root PC of standard tuning E2 = pitch class 4.
    # For each chord PC, find the nearest MIDI note ≥ E2 (MIDI 40).
    try:
        from src.theory.chord_voicings import ChordVoicingEngine
        pcs = ChordVoicingEngine()._chord_to_pitch_classes(chord_symbol)
    except Exception:
        pcs = set()

    if not pcs:
        return []

    base_midi = 40   # E2
    freqs = []
    for pc in sorted(pcs):
        # Semitones from E (pc=4) to this pc, wrapping upward
        diff = (pc - 4) % 12
        midi = base_midi + diff
        freqs.append(440.0 * 2.0 ** ((midi - 69) / 12.0))

    # Add root an octave higher for a fuller sound (mimics high strings)
    if freqs:
        freqs.append(freqs[0] * 2.0)

    return freqs


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticChordDataset(Dataset):
    """
    Procedurally generated guitar chord clips for CRNN pre-training.

    Args:
        vocabulary_level:    'basic' (25), 'sevenths' (73), or 'extended' (170).
        variations_per_chord: Number of audio variations generated per chord.
                              Each variation randomises style, decay, noise, and
                              pitch shift.  Default 20 → 170 × 20 = 3,400 items.
        duration:            Clip length in seconds.
        sample_rate:         Audio sample rate (Hz).
        hop_length:          CQT hop length in samples.
        noise_snr_db_range:  (min, max) SNR in dB for additive noise.
                             Set to None to disable noise augmentation.
        pitch_shift_range:   Max ±semitones to randomly transpose the clip while
                             keeping the chord label correct.
        cache_dir:           If provided, pre-computed CQT tensors are saved here
                             and reloaded on subsequent runs — strongly recommended
                             when running multiple training epochs.
        seed:                Random seed for reproducibility.
    """

    def __init__(
        self,
        vocabulary_level:     str                     = "extended",
        variations_per_chord: int                     = 20,
        duration:             float                   = 3.0,
        sample_rate:          int                     = 22050,
        hop_length:           int                     = 512,
        noise_snr_db_range:   Optional[tuple[float, float]] = (12.0, 30.0),
        pitch_shift_range:    int                     = 2,
        cache_dir:            Optional[str]           = None,
        seed:                 int                     = 42,
    ):
        from src.models.chord_recogniser import ChordVocabulary

        self.vocab               = ChordVocabulary(level=vocabulary_level)
        self.variations          = variations_per_chord
        self.duration            = duration
        self.sr                  = sample_rate
        self.hop_length          = hop_length
        self.noise_snr_db_range  = noise_snr_db_range
        self.pitch_shift_range   = pitch_shift_range
        self.cache_dir           = Path(cache_dir) if cache_dir else None
        self.seed                = seed

        # Build index: list of (chord_symbol, chord_idx, variation_idx)
        self._index: list[tuple[str, int, int]] = []
        for chord_idx in range(1, len(self.vocab)):   # skip N.C. (idx=0)
            symbol = self.vocab.decode(chord_idx)
            if symbol == "N.C.":
                continue
            for v in range(variations_per_chord):
                self._index.append((symbol, chord_idx, v))

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        symbol, chord_idx, variation = self._index[idx]

        # Deterministic seed per item so the dataset is reproducible across
        # epochs, but different for each variation of each chord.
        item_seed = self.seed + idx * 1000

        # Check cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{idx:06d}.pt"
            if cache_path.exists():
                data = torch.load(cache_path, weights_only=True)
                return data

        cqt = self._generate_cqt(symbol, variation, item_seed)
        n_frames = cqt.shape[-1]

        result = {
            "cqt":    cqt,
            "labels": torch.full((n_frames,), chord_idx, dtype=torch.long),
            "chord":  symbol,
        }

        if self.cache_dir:
            torch.save(result, self.cache_dir / f"{idx:06d}.pt")

        return result

    # ── Audio + CQT generation ────────────────────────────────────────────────

    def _generate_cqt(
        self, chord_symbol: str, variation: int, seed: int
    ) -> torch.Tensor:
        """Synthesise chord audio and compute its log-CQT."""
        rng = np.random.RandomState(seed)

        # Randomise synthesis parameters
        style      = STRUM_STYLES[variation % len(STRUM_STYLES)]
        strum_ms   = rng.uniform(15.0, 35.0)
        decay      = rng.uniform(0.990, 0.998)
        vel_jitter = rng.uniform(0.05, 0.20)

        # Get note frequencies for this chord
        freqs = _chord_to_freqs(chord_symbol)
        if not freqs:
            # Unknown chord — return silence labelled N.C. equivalent
            n_frames = int(np.ceil(self.duration * self.sr / self.hop_length))
            return torch.zeros(1, 84, n_frames)

        # Optional pitch shift (keeps label unchanged — just shifts the key)
        if self.pitch_shift_range > 0:
            shift = rng.randint(-self.pitch_shift_range, self.pitch_shift_range + 1)
            if shift != 0:
                ratio = 2.0 ** (shift / 12.0)
                freqs = [f * ratio for f in freqs]

        # Synthesise
        audio = _synthesise_chord(
            freqs, self.sr, self.duration,
            style=style, strum_ms=strum_ms,
            decay=decay, velocity_jitter=vel_jitter,
        )

        # Additive noise
        if self.noise_snr_db_range is not None:
            snr_db  = rng.uniform(*self.noise_snr_db_range)
            snr_lin = 10.0 ** (snr_db / 20.0)
            sig_rms = np.sqrt((audio ** 2).mean() + 1e-8)
            noise   = rng.randn(len(audio)).astype(np.float32)
            noise  *= sig_rms / (snr_lin + 1e-8)
            audio   = audio + noise

        return self._audio_to_cqt(audio)

    def _audio_to_cqt(self, audio: np.ndarray) -> torch.Tensor:
        """Compute log-CQT spectrogram from a raw audio array."""
        import librosa
        cqt = np.abs(librosa.cqt(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=84,
            bins_per_octave=12,
            fmin=32.7,
        ))
        log_cqt = np.log(cqt + 1e-8)
        return torch.from_numpy(log_cqt).unsqueeze(0).float()   # (1, 84, T)

    # ── Offline pre-generation ────────────────────────────────────────────────

    def pre_generate(self, num_workers: int = 4, verbose: bool = True):
        """
        Pre-compute and cache all CQT tensors to disk.

        Call this once before training to avoid on-the-fly synthesis overhead.
        Requires cache_dir to be set.

        Example:
            ds = SyntheticChordDataset(cache_dir="data/synthetic_cache")
            ds.pre_generate()
            # Now DataLoader will load from disk instead of synthesising
        """
        if self.cache_dir is None:
            raise ValueError("Set cache_dir before calling pre_generate().")

        from torch.utils.data import DataLoader

        loader = DataLoader(self, batch_size=1, num_workers=num_workers)
        total  = len(self)
        done   = 0

        for _ in loader:
            done += 1
            if verbose and done % 100 == 0:
                print(f"  Pre-generating synthetic dataset: {done}/{total}", end="\r")

        if verbose:
            print(f"\n  Done. {total} clips cached to {self.cache_dir}")

    # ── Statistics / diagnostics ──────────────────────────────────────────────

    def chord_coverage(self) -> dict[str, int]:
        """Return a dict of chord_symbol → number of items in the dataset."""
        from collections import Counter
        return dict(Counter(sym for sym, _, _ in self._index))

    def __repr__(self) -> str:
        return (
            f"SyntheticChordDataset("
            f"chords={len(self.vocab) - 1}, "
            f"variations={self.variations}, "
            f"total={len(self)}, "
            f"duration={self.duration}s, "
            f"sr={self.sr})"
        )
