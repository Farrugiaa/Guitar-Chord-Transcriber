"""
Audio feature extraction for chord recognition and music analysis.

Extracts CQT spectrograms, chroma features, mel spectrograms,
and onset strength for downstream models.
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


class AudioFeatureExtractor:
    """Extract audio features for chord recognition and analysis."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 512,
        n_mels: int = 128,
        n_cqt_bins: int = 84,
        bins_per_octave: int = 12,
        fmin: float = 32.7,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_cqt_bins = n_cqt_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        self.stft_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )

    def load_audio(
        self, path: str, mono: bool = True
    ) -> tuple[torch.Tensor, int]:
        """Load audio file and resample if needed.

        Args:
            path: Path to audio file.
            mono: If True (default) downmix to mono.  Pass False to keep
                  stereo so callers can use both channels (e.g. vocal removal).
        """
        import soundfile as sf

        try:
            data, sr = sf.read(path, dtype="float32")
        except Exception:
            # soundfile doesn't support MP3; fall back to librosa which does.
            # librosa returns (channels, N) for stereo vs soundfile's (N, channels),
            # so transpose stereo arrays to match the layout the code below expects.
            import librosa
            data, sr = librosa.load(path, sr=None, mono=False)
            data = data.astype(np.float32)
            if data.ndim == 2:
                data = data.T  # (channels, N) → (N, channels)
        # soundfile returns (samples,) for mono or (samples, channels) for stereo
        waveform = torch.from_numpy(data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, N)
        else:
            waveform = waveform.T  # (channels, N)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, self.sample_rate

    def compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram."""
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel + 1e-8)
        return log_mel

    def compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute power spectrogram via STFT."""
        return self.stft_transform(waveform)

    def compute_cqt(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute Constant-Q Transform spectrogram.
        Uses librosa for CQT computation, wraps result as tensor.
        """
        import librosa

        audio_np = waveform.squeeze().numpy()
        cqt = np.abs(
            librosa.cqt(
                y=audio_np,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.n_cqt_bins,
                bins_per_octave=self.bins_per_octave,
                fmin=self.fmin,
            )
        )
        log_cqt = np.log(cqt + 1e-8)
        return torch.from_numpy(log_cqt).unsqueeze(0).float()

    def compute_chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute chromagram (12-bin pitch class profile per frame).
        """
        import librosa

        audio_np = waveform.squeeze().numpy()
        chroma = librosa.feature.chroma_cqt(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_chroma=12,
            bins_per_octave=self.bins_per_octave,
        )
        return torch.from_numpy(chroma).unsqueeze(0).float()

    def compute_pitch_class_histogram(
        self, waveform: torch.Tensor
    ) -> np.ndarray:
        """
        Compute a global pitch class histogram for key detection.
        Returns a 12-element array.
        """
        import librosa

        audio_np = waveform.squeeze().numpy()
        chroma = librosa.feature.chroma_cqt(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        histogram = chroma.mean(axis=1)
        return histogram / (histogram.sum() + 1e-8)

    def compute_onset_strength(self, waveform: torch.Tensor) -> np.ndarray:
        """Compute onset strength envelope for beat tracking."""
        import librosa

        audio_np = waveform.squeeze().numpy()
        return librosa.onset.onset_strength(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

    def extract_all(self, path: str) -> dict:
        """Extract all features from an audio file."""
        waveform, _ = self.load_audio(path)
        return {
            "waveform": waveform,
            "mel_spectrogram": self.compute_mel_spectrogram(waveform),
            "cqt": self.compute_cqt(waveform),
            "chroma": self.compute_chroma(waveform),
            "pitch_class_histogram": self.compute_pitch_class_histogram(waveform),
            "onset_strength": self.compute_onset_strength(waveform),
        }
