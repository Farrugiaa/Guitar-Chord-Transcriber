"""
DSP-based guitar extraction without a trained separator model.

Pipeline (applied in sequence):
    1. Stereo downmix to mono
    2. Harmonic-percussive separation (HPSS) — removes drums/transients
    3. Mel-domain Wiener filter — suppresses residual non-guitar harmonic
       content by comparing harmonic-to-mix energy in Mel frequency bins
    4. Frequency-domain EQ mask — final suppression of out-of-band content

The Mel Wiener filter (Stage 3) is the key guitar-specific step.  It computes
the ratio of harmonic-to-mixed energy at Mel-scale resolution, which gives
finer control in the low-frequency guitar range (82–500 Hz) where
fundamentals live, then maps the mask back to STFT space to preserve phase
for clean reconstruction.

Note: separating guitar from vocals/strings that share the same frequency
range still requires the trained U-Net separator.  This DSP pipeline reliably
removes drums and out-of-band noise, which still improves chord detection.
"""

import numpy as np
import torch


# Guitar frequency range (Hz)
#   Sub-bass / DC below 40 Hz: not musically relevant for chords, suppress gently
#   Low E fundamental: ~82 Hz — must be fully preserved
#   Body resonance / warmth: 100-400 Hz — critical for chord identity
#   Presence / harmonics: up to 8 kHz
#   Air band above 12 kHz: not needed for chord detection
_GUITAR_LO_HZ  = 40.0       # gentle roll-off start
_GUITAR_HI_HZ  = 8_000.0   # roll-off begins here — preserves guitar presence and string harmonics
_ROLLOFF_HI_HZ = 12_000.0  # complete cut by 12 kHz; above this is air band not relevant to chords


class GuitarExtractor:
    """
    Extract the guitar layer from a mixed audio signal using signal processing.

    No trained model is required.  The three stages are cumulative: each one
    removes a class of interference that would otherwise pollute chord
    detection (vocals → drums → off-band noise).
    """

    def __init__(
        self,
        hpss_margin: float = 3.0,
        hpss_kernel: int = 31,
        eq_blend: float = 0.7,      # higher = more band-limited (vocal-reduced) signal
    ):
        """
        Args:
            hpss_margin:  Separation aggressiveness for HPSS (higher = cleaner
                          harmonic, but may introduce artefacts). 3–4 is typical.
            hpss_kernel:  Median-filter kernel size for HPSS.
            eq_blend:     Weight of the band-passed signal in the final mix
                          (higher = more frequency-limited, more vocal suppression).
        """
        self.hpss_margin = hpss_margin
        self.hpss_kernel = hpss_kernel
        self.eq_blend    = eq_blend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """
        Extract the guitar layer from a waveform.

        Args:
            waveform:    Audio tensor, shape (channels, N).  May be mono or
                         stereo.
            sample_rate: Sample rate in Hz.

        Returns:
            Mono guitar waveform tensor, shape (1, N).
        """
        # Downmix to mono — centre-channel subtraction is intentionally
        # skipped because guitar is almost always centre-panned; subtracting
        # the centre removes the guitar along with the vocals.
        if waveform.shape[0] > 1:
            audio = waveform.mean(dim=0).numpy()
        else:
            audio = waveform.squeeze(0).numpy()

        # Stage 1 — harmonic/percussive separation (removes drums/transients)
        audio_mix      = audio.copy()
        audio_harmonic = self._harmonic_separation(audio, sample_rate)
        # Normalise harmonic to same RMS as mix to prevent level drop
        rms_mix  = np.sqrt(np.mean(audio_mix ** 2)) + 1e-8
        rms_harm = np.sqrt(np.mean(audio_harmonic ** 2)) + 1e-8
        audio_harmonic = audio_harmonic * (rms_mix / rms_harm)

        # Stage 2 — Mel-domain Wiener filter (suppresses non-guitar harmonics)
        audio = self._mel_wiener_filter(audio_mix, audio_harmonic, sample_rate)
        # Normalise after Wiener to prevent clipping before EQ
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.9

        # Stage 3 — frequency EQ mask (suppresses sub-bass and air band)
        audio = self._frequency_eq_mask(audio, sample_rate)

        # Final normalise
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.95

        return torch.from_numpy(audio).unsqueeze(0).float()

    # ------------------------------------------------------------------
    # Stage 1 — Harmonic-Percussive Source Separation (HPSS)
    # ------------------------------------------------------------------

    def _harmonic_separation(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Use librosa's HPSS to separate sustained harmonic content (guitar,
        piano, vocals) from transients (drums, percussion, pick attacks).

        We keep only the harmonic component, which strongly suppresses kick,
        snare, and hi-hat bleed that would otherwise confuse chord detection.
        """
        import librosa

        harmonic, _ = librosa.effects.hpss(
            audio,
            kernel_size=self.hpss_kernel,
            margin=self.hpss_margin,
        )
        return harmonic

    # ------------------------------------------------------------------
    # Stage 2 — Mel-domain Wiener filter
    # ------------------------------------------------------------------

    def _mel_wiener_filter(
        self,
        audio_mix:      np.ndarray,
        audio_harmonic: np.ndarray,
        sample_rate:    int,
        n_mels:         int = 128,
        n_fft:          int = 2048,
        hop_length:     int = 512,
        power:          float = 1.5,
    ) -> np.ndarray:
        """
        Suppress non-guitar harmonic content using a Mel-domain Wiener mask.

        The Mel scale gives ~3× finer resolution in the guitar fundamental
        range (80–500 Hz) than in the high-frequency range, so the mask is
        more precise where guitar notes actually live.

        Algorithm:
            1. Compute STFT of the original mix and the HPSS harmonic component.
            2. Project both magnitude spectrograms into Mel space via the Mel
               filterbank (128 bins, 80 Hz – 8 kHz).
            3. Compute a power Wiener mask per Mel bin:
                   W_mel = M_harm^p / (M_mix^p + ε)
               where p=2 (power Wiener) gives sharper separation than p=1.
            4. Project the Mel mask back to STFT frequency bins via a
               normalised transpose of the filterbank.
            5. Apply the STFT mask to the harmonic STFT while keeping its
               original phase — no Griffin-Lim required.
            6. Reconstruct audio via iSTFT.

        The mask is high (≈1) where harmonic content dominates the mix
        (guitar regions) and low (≈0) where the mix has energy but the
        harmonic component does not (percussion residual, noise, air band).
        """
        import librosa

        # Clip lengths must match
        length = min(len(audio_mix), len(audio_harmonic))
        audio_mix      = audio_mix[:length]
        audio_harmonic = audio_harmonic[:length]

        # ── STFTs ────────────────────────────────────────────────────────────
        D_mix  = librosa.stft(audio_mix,      n_fft=n_fft, hop_length=hop_length)
        D_harm = librosa.stft(audio_harmonic, n_fft=n_fft, hop_length=hop_length)

        S_mix  = np.abs(D_mix)
        S_harm = np.abs(D_harm)

        # ── Mel filterbank (guitar range: 80 Hz – 8 kHz) ─────────────────────
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=80.0, fmax=8_000.0,
        )   # shape (n_mels, n_fft//2+1)

        # ── Mel-domain magnitudes ─────────────────────────────────────────────
        M_mix  = mel_fb @ S_mix  + 1e-8   # (n_mels, T)
        M_harm = mel_fb @ S_harm + 1e-8   # (n_mels, T)

        # ── Power Wiener mask in Mel domain ───────────────────────────────────
        # W ∈ [0, 1]:  high where harmonic dominates, low where mix > harmonic
        # power=1.5 is softer than 2.0 — avoids over-suppression artifacts.
        W_mel = (M_harm ** power) / (M_mix ** power + 1e-8)
        W_mel = np.clip(W_mel, 0.0, 1.0)

        # Temporal smoothing: median filter across ~115 ms of frames to prevent
        # rapid mask fluctuations that cause pitch artifacts / amplitude modulation.
        from scipy.ndimage import median_filter
        W_mel = median_filter(W_mel, size=(1, 9))

        # Floor of 0.15 prevents hard silence jumps (which cause clicks/pops)
        W_mel = np.clip(W_mel, 0.15, 1.0)

        # ── Project mask back to STFT bins ────────────────────────────────────
        # Each STFT bin is spread across Mel bins by mel_fb; the transpose
        # maps Mel masks back, normalised by each bin's total Mel weight so
        # the mask values remain in [0, 1].
        mel_bin_sum = mel_fb.sum(axis=0) + 1e-8   # (n_stft_bins,)
        W_stft = (mel_fb.T @ W_mel) / mel_bin_sum[:, None]
        W_stft = np.clip(W_stft, 0.15, 1.0)

        # ── Apply mask and reconstruct ────────────────────────────────────────
        D_masked = D_harm * W_stft
        audio_out = librosa.istft(D_masked, hop_length=hop_length, length=length)
        return audio_out.astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 3 — Frequency-domain EQ mask
    # ------------------------------------------------------------------

    def _frequency_eq_mask(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Apply a soft guitar-shaped frequency mask via STFT.

        Rather than a hard bandpass filter (which creates audible ringing),
        we compute a magnitude spectrogram, multiply each frequency bin by a
        smooth weight that peaks in the guitar range, and reconstruct the
        waveform via inverse STFT with the original phase.

        Weight profile:
            0 Hz  → 0.0   (DC / sub-bass)
            75 Hz → 1.0   (guitar low E)
            6 kHz → 1.0   (guitar high harmonics)
            12 kHz → 0.0  (gradual roll-off)
            above → 0.0   (air band, noise)
        """
        import numpy.fft as fft

        n_fft    = 4096
        hop      = n_fft // 4
        n_frames = 1 + (len(audio) - n_fft) // hop

        # Build frequency weight vector
        freqs   = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        weights = self._guitar_eq_curve(freqs)

        # Process frame-by-frame with overlap-add
        output = np.zeros(len(audio), dtype=np.float32)
        window = np.hanning(n_fft).astype(np.float32)

        for i in range(n_frames):
            start = i * hop
            end   = start + n_fft
            if end > len(audio):
                break

            frame   = audio[start:end] * window
            spectrum = np.fft.rfft(frame)

            # Apply soft EQ mask
            masked  = spectrum * weights

            # Reconstruct (keep original phase, scale magnitude)
            reconstructed = np.fft.irfft(masked).astype(np.float32)
            output[start:end] += reconstructed * window

        # Normalise by window overlap
        normaliser = np.zeros(len(audio), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            end   = start + n_fft
            if end > len(audio):
                break
            normaliser[start:end] += window ** 2

        safe = normaliser > 1e-8
        output[safe] /= normaliser[safe]

        # Blend with pre-EQ signal for naturalness
        blend_len = min(len(output), len(audio))
        result    = (
            self.eq_blend * output[:blend_len]
            + (1.0 - self.eq_blend) * audio[:blend_len]
        )
        return result.astype(np.float32)

    @staticmethod
    def _guitar_eq_curve(freqs: np.ndarray) -> np.ndarray:
        """
        Smooth trapezoidal EQ weight for each frequency bin.

        The curve is piecewise-linear so it produces no ringing:
            0 → _GUITAR_LO_HZ  : ramp  0 → 1
            _GUITAR_LO_HZ → _GUITAR_HI_HZ : flat 1
            _GUITAR_HI_HZ → _ROLLOFF_HI_HZ: ramp 1 → 0
            above _ROLLOFF_HI_HZ           : flat 0
        """
        weights = np.zeros(len(freqs), dtype=np.float32)
        for k, f in enumerate(freqs):
            if f < _GUITAR_LO_HZ:
                weights[k] = f / _GUITAR_LO_HZ
            elif f <= _GUITAR_HI_HZ:
                weights[k] = 1.0
            elif f <= _ROLLOFF_HI_HZ:
                weights[k] = 1.0 - (f - _GUITAR_HI_HZ) / (_ROLLOFF_HI_HZ - _GUITAR_HI_HZ)
            else:
                weights[k] = 0.0
        return weights
