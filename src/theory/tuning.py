"""
Guitar tuning definitions and audio-based tuning detector.

Tuning is represented as a list of 6 MIDI note numbers (low → high string).
Standard tuning: E2 A2 D3 G3 B3 E4 = [40, 45, 50, 55, 59, 64]

The TuningDetector estimates the tuning from the audio by analysing the
lowest prominent fundamental frequency.  It cannot reliably distinguish
between all alternate tunings from a mixed recording, so it focuses on the
most common variants and falls back to standard if uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── Tuning definitions ────────────────────────────────────────────────────────
# Each entry: (display_name, [MIDI notes low→high], [string names low→high])

@dataclass(frozen=True)
class Tuning:
    name:         str
    midi_notes:   tuple[int, ...]   # 6 MIDI note numbers, low → high
    string_names: tuple[str, ...]   # 6 display names

    @property
    def pitch_classes(self) -> list[int]:
        """Open-string pitch classes (0-11) for voicing calculations."""
        return [n % 12 for n in self.midi_notes]

    @property
    def lowest_midi(self) -> int:
        return min(self.midi_notes)

    @property
    def lowest_hz(self) -> float:
        return 440.0 * 2 ** ((self.lowest_midi - 69) / 12)


TUNINGS: dict[str, Tuning] = {
    "standard": Tuning(
        name="Standard (EADGBe)",
        midi_notes=(40, 45, 50, 55, 59, 64),
        string_names=("E", "A", "D", "G", "B", "e"),
    ),
    "drop_d": Tuning(
        name="Drop D (DADGBe)",
        midi_notes=(38, 45, 50, 55, 59, 64),
        string_names=("D", "A", "D", "G", "B", "e"),
    ),
    "half_step_down": Tuning(
        name="Eb tuning (Eb Ab Db Gb Bb eb)",
        midi_notes=(39, 44, 49, 54, 58, 63),
        string_names=("Eb", "Ab", "Db", "Gb", "Bb", "eb"),
    ),
    "drop_d_half_step": Tuning(
        name="Drop D half-step down (Db Ab Db Gb Bb Eb)",
        midi_notes=(37, 44, 49, 54, 58, 63),
        string_names=("Db", "Ab", "Db", "Gb", "Bb", "Eb"),
    ),
    "open_g": Tuning(
        name="Open G (DGDGBd)",
        midi_notes=(38, 43, 50, 55, 59, 62),
        string_names=("D", "G", "D", "G", "B", "d"),
    ),
    "open_d": Tuning(
        name="Open D (DADf#Ad)",
        midi_notes=(38, 45, 50, 54, 57, 62),
        string_names=("D", "A", "D", "F#", "A", "d"),
    ),
    "dadgad": Tuning(
        name="DADGAD",
        midi_notes=(38, 45, 50, 55, 57, 62),
        string_names=("D", "A", "D", "G", "A", "D"),
    ),
    "open_e": Tuning(
        name="Open E (EBEg#Be)",
        midi_notes=(40, 47, 52, 56, 59, 64),
        string_names=("E", "B", "E", "G#", "B", "e"),
    ),
    "open_a": Tuning(
        name="Open A (EAEAc#e)",
        midi_notes=(40, 45, 52, 57, 61, 64),
        string_names=("E", "A", "E", "A", "C#", "e"),
    ),
    "drop_c": Tuning(
        name="Drop C (CGCFAd)",
        midi_notes=(36, 43, 48, 53, 57, 62),
        string_names=("C", "G", "C", "F", "A", "d"),
    ),
    "full_step_down": Tuning(
        name="Full step down (DGCFAd)",
        midi_notes=(38, 43, 48, 53, 57, 62),
        string_names=("D", "G", "C", "F", "A", "d"),
    ),
}

DEFAULT_TUNING = TUNINGS["standard"]

ACTIVE_TUNING_KEYS = (
    "standard",
    "drop_d",
    "half_step_down",
    "drop_d_half_step",
    "open_g",
    "open_d",
    "dadgad",
    "open_e",
    "open_a",
    "drop_c",
    "full_step_down",
)


# ── Detector ──────────────────────────────────────────────────────────────────

def _hz_to_midi(hz: float) -> float:
    """Convert Hz to MIDI note number (fractional)."""
    return 69.0 + 12.0 * np.log2(max(hz, 1e-6) / 440.0)


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


class TuningDetector:
    """
    Estimate guitar tuning using onset-driven pYIN pitch detection.

    Strategy
    --------
    1. Clip to the first ``max_seconds`` of audio (tuning is consistent
       throughout; analysing the whole file is wasteful).
    2. Detect note onset frames with librosa so we pitch-track individual
       notes rather than a continuous mixture.
    3. For each onset, run librosa.pyin on a short window around it to get
       a voiced fundamental frequency estimate.  pyin is much more reliable
       than FFT peak-picking in a mixed signal.
    4. Collect all voiced F0s that fall in the guitar range (55–660 Hz).
    5. Score every known tuning by how many of its open-string frequencies
       are close (≤ 0.4 semitones) to a detected F0.
    6. Normalise the score by the number of strings to get a 0–1 confidence.

    Why this is better than the previous approach
    ----------------------------------------------
    * pyin works on short monophonic windows around each onset, so the
      fundamental is not buried by harmonics or other instruments.
    * Scoring across all 6 open-string frequencies means a single low-string
      match alone is not enough — the tuning must explain several detected
      pitches, which sharply reduces false positives.
    * Confidence is now relative (matches / strings), so a genuinely
      uncertain result returns a low score and falls back to Standard.
    """

    # Guitar open-string range: low C2 (drop C) to high A4
    _F0_LO_HZ  = 60.0
    _F0_HI_HZ  = 450.0
    # A detected F0 "matches" an open string if within this many semitones
    _MATCH_ST  = 0.45

    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        max_seconds: float = 120.0,
        skip_seconds: float = 4.0,
    ) -> tuple[Tuning, float]:
        """
        Detect tuning from a mono audio array.

        Args:
            audio:        mono float32 array
            sample_rate:  sample rate in Hz
            max_seconds:  how many seconds of audio to analyse (extended to
                          120 s from 90 s — more evidence improves robustness)
            skip_seconds: seconds of audio to skip at the start (default 4 s).
                          Guitar introductions often contain feedback, slides,
                          or open-string noodling that is not representative of
                          the song's tuning; skipping them reduces false matches.

        Returns:
            (tuning, confidence)  — confidence in [0, 1].
        """
        import librosa

        skip_samples = int(skip_seconds * sample_rate)
        n_samples    = int(max_seconds * sample_rate)
        clipped = audio[skip_samples:skip_samples + n_samples]
        # If the skip eats all audio (very short file) fall back to full audio
        clip = clipped if len(clipped) >= sample_rate else audio[:n_samples]
        clip = clip.astype(np.float32)

        hop_length = 512

        # ── Step 1: onset detection ───────────────────────────────────────
        onset_frames = librosa.onset.onset_detect(
            y=clip,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=True,
        )
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)

        # ── Step 2: pyin pitch detection around each onset ────────────────
        window_samples = int(0.15 * sample_rate)   # 150 ms window per onset
        detected_f0s: list[float] = []

        for onset in onset_samples:
            start = int(onset)
            end   = min(start + window_samples, len(clip))
            if end - start < 1024:
                continue
            segment = clip[start:end]

            try:
                f0, voiced_flag, _ = librosa.pyin(
                    segment,
                    fmin=self._F0_LO_HZ,
                    fmax=self._F0_HI_HZ,
                    sr=sample_rate,
                    hop_length=hop_length // 2,
                )
                voiced_f0s = f0[voiced_flag & ~np.isnan(f0)]
                if len(voiced_f0s) > 0:
                    # Use median to avoid octave-jump outliers
                    detected_f0s.append(float(np.median(voiced_f0s)))
            except Exception:
                continue

        if not detected_f0s:
            return DEFAULT_TUNING, 0.0

        # Correct for recordings that are not at A=440 (e.g. A=432, tape drift).
        # librosa.estimate_tuning returns the deviation in semitones; subtracting
        # it from the detected MIDI notes aligns them to the nearest equal-tempered
        # pitch so that tuning matching is not thrown off by a global pitch shift.
        try:
            tuning_offset = librosa.estimate_tuning(y=clip, sr=sample_rate)
        except Exception:
            tuning_offset = 0.0

        detected_midi = np.array([_hz_to_midi(f) for f in detected_f0s]) - tuning_offset

        # ── Step 3: score active tunings only ────────────────────────────
        best_tuning    = DEFAULT_TUNING
        best_score     = -1.0
        standard_score = 0.0

        for key in ACTIVE_TUNING_KEYS:
            tuning = TUNINGS[key]
            string_midi = np.array([_hz_to_midi(_midi_to_hz(m))
                                    for m in tuning.midi_notes])
            matches = 0
            for sm in string_midi:
                dists = np.abs(detected_midi - sm)
                if dists.min() <= self._MATCH_ST:
                    matches += 1

            score = matches / len(string_midi)   # 0–1
            if score > best_score:
                best_score  = score
                best_tuning = tuning
            if key == "standard":
                standard_score = score

        # ── Step 4: open-tuning disambiguation ───────────────────────────
        # Open G and Open D share all their open-string pitch classes with
        # common chords played in Standard tuning:
        #   Open G  → D G B  (= G-major triad, pitch classes {2, 7, 11})
        #   Open D  → D F# A (= D-major triad, pitch classes {2, 6, 9})
        # When the detected F0s fall entirely within one of these overlap
        # sets there is no discriminating information, so we default to
        # Standard rather than guess an open tuning.
        _OPEN_TUNING_CHORD_PCS = (
            frozenset({2, 7, 11}),   # Open G ≡ G-major chord
            frozenset({2, 6, 9}),    # Open D ≡ D-major chord
            frozenset({4, 11, 4}),   # Open E ≡ E-major chord
            frozenset({9, 4, 1}),    # Open A ≡ A-major chord
        )
        detected_pcs = frozenset(int(round(m)) % 12 for m in detected_midi)

        for overlap in _OPEN_TUNING_CHORD_PCS:
            if detected_pcs <= overlap:
                return DEFAULT_TUNING, standard_score

        # Apply a flat Standard prior.
        # Prior must be high enough to absorb coincidental matches: Drop C's open
        # G string (43 MIDI) and C string (48 MIDI) are common chord tones in
        # standard-tuning songs, giving Drop C a spurious 2/6 = 0.33 score even
        # when the guitar is unambiguously in standard tuning.
        # Prior 0.35 + required margin 0.10 means a non-Standard tuning must score
        # at least (standard_score + 0.45) to win — ruling out coincidental matches
        # while still allowing genuine alternate tunings (Drop C full match = 1.0).
        _STANDARD_PRIOR  = 0.35
        _SWITCH_MARGIN   = 0.10
        if best_score <= standard_score + _STANDARD_PRIOR + _SWITCH_MARGIN:
            best_tuning = DEFAULT_TUNING
            best_score  = standard_score

        confidence = best_score

        # Pitch-floor confirmation: onset-driven open-string matching
        # systematically underestimates confidence because players use fretted
        # notes, not open strings, so 2–3/6 is a realistic ceiling even for a
        # perfectly standard-tuned song.
        #
        # Physical constraint: the 20th-percentile of all detected pitches is the
        # effective pitch floor. If the floor is at or above the selected tuning's
        # lowest string (within 2 semitones of tolerance for pYIN error), the
        # tuning selection is physically confirmed and confidence is boosted to 0.75.
        # This is independent of and additive to the open-string match score.
        if len(detected_midi) >= 10:
            floor_midi = float(np.percentile(detected_midi, 20))
            if floor_midi >= best_tuning.lowest_midi - 2.0:
                confidence = max(confidence, 0.75)

        if confidence < self.confidence_threshold:
            return DEFAULT_TUNING, confidence

        return best_tuning, confidence
