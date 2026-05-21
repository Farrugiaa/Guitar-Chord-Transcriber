"""
Strumming pattern detector.

Derives the strumming rhythm by detecting note onsets in the guitar stem,
mapping each onset to the nearest 8th-note grid position, and aggregating
across all bars to produce a single representative one-bar pattern.

Algorithm
---------
1.  Onset detection  — librosa onset_detect on the guitar stem.
2.  Quantisation     — each onset is snapped to the nearest 8th-note grid
                       position within its bar using BPM and time signature.
3.  Direction        — determined purely by beat position:
                         on-beat  (grid positions 0, 2, 4, 6 …) → ↓ downstroke
                         off-beat (grid positions 1, 3, 5, 7 …) → ↑ upstroke
4.  Aggregation      — for each grid position, count occurrences across all bars.
                       Positions occupied in fewer than 25 % of bars are silent.

This replaces the previous spectral-band classifier (low vs high frequency
energy ratio) which required an FFT per onset and gave unreliable results for
fingerpicked and electric guitar.  The rhythm of notes being played, relative
to the beat grid, is the meaningful information; up/down direction follows
naturally from the beat position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StrumBeat:
    """One 8th-note grid cell in the pattern."""
    direction: str   # 'D' (down), 'U' (up), or '' (silent)
    label:     str   # '1', '&', '2', '&', …


@dataclass
class StrumPattern:
    beats:          list[StrumBeat]
    time_signature: tuple[int, int]
    bpm:            float


class StrumDetector:
    """
    Detect the strumming pattern from a guitar audio stem.

    Requires BPM and time signature to be known before calling detect(),
    so it should run after music analysis.

    Parameters
    ----------
    presence_threshold : float
        Fraction of bars in which a note must occur at a given grid position
        for that position to appear in the pattern (default 0.25 = 25 %).
    """

    def __init__(self, presence_threshold: float = 0.25):
        self._pt = presence_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        audio:          np.ndarray,
        sample_rate:    int,
        bpm:            float,
        time_signature: tuple[int, int] = (4, 4),
        max_seconds:    float = 45.0,
    ) -> StrumPattern:
        """
        Detect the strumming pattern from ``audio``.

        Parameters
        ----------
        audio          : mono float32 guitar stem
        sample_rate    : sample rate of ``audio``
        bpm            : song tempo — must be known before calling (run after
                         music analysis so the grid is correctly sized)
        time_signature : (numerator, denominator) e.g. (4, 4)
        max_seconds    : only analyse the first N seconds

        Returns
        -------
        StrumPattern — one bar's worth of 8th-note grid cells, each marked
        D (downstroke, on-beat), U (upstroke, off-beat), or '' (no note).
        """
        import librosa

        beats_per_bar = time_signature[0]
        subdivisions  = 2                              # 8th-note grid
        grid_size     = beats_per_bar * subdivisions   # 8 cells for 4/4
        beat_dur      = 60.0 / max(bpm, 1.0)
        bar_dur       = beat_dur * beats_per_bar

        n_max = int(max_seconds * sample_rate)
        clip  = audio[:n_max].astype(np.float32)

        # ── 1. Onset detection ────────────────────────────────────────────────
        hop = 512
        onset_frames = librosa.onset.onset_detect(
            y=clip, sr=sample_rate, hop_length=hop,
            units="frames", backtrack=True,
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=sample_rate, hop_length=hop
        )

        if len(onset_times) == 0:
            return self._empty_pattern(beats_per_bar, subdivisions, bpm, time_signature)

        # ── 2 & 3. Quantise to grid + direction from position ─────────────────
        # grid[i] = count of onsets that fell at that 8th-note position
        grid: list[int] = [0] * grid_size

        for t in onset_times:
            beats_elapsed = (t * bpm) / 60.0
            pos_in_bar    = beats_elapsed % beats_per_bar   # 0 … beats_per_bar
            grid_idx      = int(round(pos_in_bar * subdivisions)) % grid_size
            grid[grid_idx] += 1

        # ── 4. Aggregate across bars ──────────────────────────────────────────
        total_time = float(onset_times[-1]) if len(onset_times) else 0.0
        total_bars = max(1, int(total_time / bar_dur))
        min_count  = max(1, int(total_bars * self._pt))

        labels = self._beat_labels(beats_per_bar, subdivisions)
        beats: list[StrumBeat] = []

        for i, count in enumerate(grid):
            if count >= min_count:
                # Even grid indices are on-beat (1, 2, 3, 4) → downstroke
                # Odd grid indices are off-beat (&)            → upstroke
                direction = "D" if i % 2 == 0 else "U"
            else:
                direction = ""
            beats.append(StrumBeat(direction=direction, label=labels[i]))

        return StrumPattern(beats=beats, time_signature=time_signature, bpm=bpm)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _beat_labels(beats_per_bar: int, subdivisions: int) -> list[str]:
        labels: list[str] = []
        for beat in range(1, beats_per_bar + 1):
            labels.append(str(beat))
            for _ in range(1, subdivisions):
                labels.append("&")
        return labels

    @staticmethod
    def _empty_pattern(
        beats_per_bar: int,
        subdivisions:  int,
        bpm:           float,
        time_sig:      tuple[int, int],
    ) -> StrumPattern:
        labels = StrumDetector._beat_labels(beats_per_bar, subdivisions)
        return StrumPattern(
            beats=[StrumBeat(direction="", label=lbl) for lbl in labels],
            time_signature=time_sig,
            bpm=bpm,
        )
