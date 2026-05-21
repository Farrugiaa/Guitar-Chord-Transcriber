"""Audio utility functions."""

import torch
import torchaudio
import numpy as np
from pathlib import Path


def load_and_preprocess(
    path: str,
    target_sr: int = 44100,
    mono: bool = True,
    normalize: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load an audio file, resample, convert to mono, and normalize."""
    waveform, sr = torchaudio.load(path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if normalize:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    return waveform, target_sr


def segment_audio(
    waveform: torch.Tensor,
    segment_length: int,
    hop_length: int | None = None,
) -> list[torch.Tensor]:
    """Split waveform into overlapping segments."""
    if hop_length is None:
        hop_length = segment_length

    total_length = waveform.shape[-1]
    segments = []

    for start in range(0, total_length - segment_length + 1, hop_length):
        segment = waveform[..., start : start + segment_length]
        segments.append(segment)

    return segments


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int = 44100,
) -> None:
    """Save waveform to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(path, waveform, sample_rate)
