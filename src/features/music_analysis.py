"""BPM, key, and time signature detection from audio."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class MusicAnalysisResult:
    """Results from analysing musical properties of an audio file."""
    bpm: float
    key: str
    key_confidence: float
    scale_type: str  # "major" or "natural_minor"
    time_signature: tuple[int, int]  # (numerator, denominator)
    beat_times: np.ndarray  # timestamps of detected beats
    downbeat_times: np.ndarray  # timestamps of detected downbeats


class MusicAnalyser:
    """Detect BPM, key, and time signature from audio."""

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def analyse(self, waveform: torch.Tensor) -> MusicAnalysisResult:
        """
        Run full music analysis on a waveform.

        Args:
            waveform: Audio tensor (1, N)

        Returns:
            MusicAnalysisResult with BPM, key, time signature.
        """
        import librosa

        audio = waveform.squeeze().numpy()

        # BPM and beat tracking
        bpm, beat_frames = self._detect_bpm_and_beats(audio)
        beat_times = librosa.frames_to_time(
            beat_frames, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Key detection
        key, scale_type, confidence = self._detect_key(audio)

        # Time signature estimation
        time_sig, downbeat_times = self._detect_time_signature(
            audio, beat_times, bpm
        )

        return MusicAnalysisResult(
            bpm=bpm,
            key=key,
            key_confidence=confidence,
            scale_type=scale_type,
            time_signature=time_sig,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
        )

    def _detect_bpm_and_beats(
        self, audio: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Detect BPM via tempogram energy scoring across half/double/two-thirds tempo
        candidates. Tempogram autocorrelation handles swing and mixed subdivisions
        better than beat regularity.
        """
        import librosa

        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        bpm, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            start_bpm=90,
        )

        if hasattr(bpm, '__len__'):
            bpm = float(bpm[0]) if len(bpm) > 0 else 120.0
        else:
            bpm = float(bpm)

        tgram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        tempo_bins = librosa.tempo_frequencies(
            tgram.shape[0],
            hop_length=self.hop_length,
            sr=self.sample_rate,
        )

        def _tgram_score(candidate_bpm: float) -> float:
            if candidate_bpm <= 0:
                return 0.0
            idx = int(np.argmin(np.abs(tempo_bins - candidate_bpm)))
            half_w = 2
            lo = max(0, idx - half_w)
            hi = min(tgram.shape[0], idx + half_w + 1)
            energy = float(tgram[lo:hi].mean())
            # Mild prior: guitar songs are almost always 55–145 BPM
            if candidate_bpm < 55 or candidate_bpm > 145:
                energy *= 0.85
            return energy

        candidates: list[tuple[float, np.ndarray]] = [(bpm, beat_frames)]

        def _try_tempo(seed: float) -> None:
            if seed < 30 or seed > 250:
                return
            actual_bpm, frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                start_bpm=seed,
            )
            if hasattr(actual_bpm, '__len__'):
                actual_bpm = float(actual_bpm[0]) if len(actual_bpm) > 0 else seed
            else:
                actual_bpm = float(actual_bpm)
            candidates.append((actual_bpm, frames))

        _try_tempo(bpm / 2.0)
        _try_tempo(bpm * 2.0)
        _try_tempo(bpm * 2.0 / 3.0)

        # 110–150 BPM zone: often "doubled slow songs" where the tempogram peaks
        # at 8th-note subdivisions. Prefer halved tempo when it falls in 50–85.
        best_bpm, best_frames = max(candidates, key=lambda c: _tgram_score(c[0]))
        if 110 <= best_bpm <= 150:
            half_bpm = best_bpm / 2.0
            if 50 <= half_bpm <= 85:
                half_score = _tgram_score(half_bpm)
                full_score = _tgram_score(best_bpm)
                if half_score >= full_score * 0.60:
                    _try_tempo(half_bpm)
                    best_bpm, best_frames = min(
                        candidates, key=lambda c: abs(c[0] - half_bpm)
                    )
        bpm, beat_frames = best_bpm, best_frames

        return bpm, beat_frames

    def _detect_key(
        self, audio: np.ndarray
    ) -> tuple[str, str, float]:
        """
        Krumhansl-Schmuckler key detection with major/minor disambiguation.
        K-S often returns the parallel major for minor keys; if the minor 3rd
        energy exceeds the major 3rd and the minor correlation is within 0.06,
        the minor reading is preferred.
        """
        import librosa
        from src.theory.scales import ScaleAnalyser
        from src.theory.intervals import NOTE_NAMES

        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        histogram = chroma.mean(axis=1)
        histogram = histogram / (histogram.sum() + 1e-8)

        analyser = ScaleAnalyser()
        candidates = analyser.detect_key(histogram.tolist())

        if not candidates:
            return "C", "major", 0.0

        best_key, best_scale, best_corr = candidates[0]

        if best_scale == "major" and best_key in NOTE_NAMES:
            root_pc   = NOTE_NAMES.index(best_key)
            minor_3rd = float(histogram[(root_pc + 3) % 12])
            major_3rd = float(histogram[(root_pc + 4) % 12])
            if minor_3rd > major_3rd:
                minor_cands = [c for c in candidates if c[0] == best_key and c[1] == "natural_minor"]
                if minor_cands:
                    _, _, minor_corr = minor_cands[0]
                    if minor_corr > best_corr - 0.06:
                        best_key, best_scale, best_corr = best_key, "natural_minor", minor_corr

        # D#/A#/G# → Eb/Bb/Ab; C# and F# stay as-is (both idiomatic on guitar)
        _PREFER_FLAT = {"D#": "Eb", "A#": "Bb", "G#": "Ab"}
        if best_key in _PREFER_FLAT:
            best_key = _PREFER_FLAT[best_key]

        return best_key, best_scale, best_corr

    def _detect_time_signature(
        self,
        audio: np.ndarray,
        beat_times: np.ndarray,
        bpm: float,
    ) -> tuple[tuple[int, int], np.ndarray]:
        """
        Beat-strength autocorrelation to estimate time signature.
        Phase-invariant (unlike fixed-phase grouping). Priors: 4/4=1.0, 3/4=0.65, 6/4=0.55.
        """
        import librosa

        if len(beat_times) < 8:
            return (4, 4), beat_times[::4] if len(beat_times) > 0 else np.array([])

        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        beat_frames = librosa.time_to_frames(
            beat_times, sr=self.sample_rate, hop_length=self.hop_length
        )

        beat_strengths = np.array([
            float(onset_env[bf]) if bf < len(onset_env) else 0.0
            for bf in beat_frames
        ])

        normed = beat_strengths - beat_strengths.mean()
        if normed.std() > 1e-6:
            normed /= normed.std()

        n = len(normed)
        full_acorr = np.correlate(normed, normed, mode="full")
        acorr = full_acorr[n - 1:]            # keep non-negative lags
        acorr /= (acorr[0] + 1e-8)            # normalise: lag-0 = 1.0

        # Average lag g and 2g: genuine g-beat metre peaks at all multiples.
        # 4/4 safety margin prevents false 3/4 on uniformly strummed 4/4 songs.
        priors      = {3: 0.65, 4: 1.00, 6: 0.55}
        _4_4_MARGIN = 0.10

        scores: dict[int, float] = {}
        for g in [3, 4, 6]:
            if g >= len(acorr):
                scores[g] = 0.0
                continue
            s = float(acorr[g])
            if 2 * g < len(acorr):
                s = (s + float(acorr[2 * g])) / 2.0
            scores[g] = s * priors[g]

        best_grouping = max(scores, key=lambda g: scores[g])
        if best_grouping != 4 and scores[best_grouping] <= scores[4] + _4_4_MARGIN:
            best_grouping = 4

        time_sig       = (best_grouping, 4)
        downbeat_times = beat_times[::best_grouping]

        return time_sig, downbeat_times
