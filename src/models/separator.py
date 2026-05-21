"""
Guitar source separation model using a U-Net architecture.

Two classes are provided:

GuitarSeparatorUNet
    Core U-Net that maps a spectrogram (STFT or Mel) to a soft mask of the
    same shape.  The architecture is input-agnostic — only the frequency
    dimension changes between STFT (n_fft//2+1 bins) and Mel (n_mels bins).

MelGuitarSeparator
    End-to-end wrapper that handles the full pipeline from raw audio:
        1. Compute STFT (save complex values / phase).
        2. Convert magnitude STFT → Mel-spectrogram.
        3. Run GuitarSeparatorUNet to predict a Mel-domain mask.
        4. Project Mel mask back to STFT bins (normalised filterbank transpose).
        5. Apply mask to complex STFT (keeps original phase — no Griffin-Lim).
        6. Reconstruct audio via iSTFT.

    Using Mel as the U-Net input gives finer frequency resolution in the
    guitar fundamental range (80–500 Hz) where the model most needs to
    discriminate guitar from bass/vocals, while using fewer input bins
    overall (128 vs ~2049), which significantly reduces model size and
    training time.
"""

import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    """Encoder block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvBlock(nn.Module):
    """Decoder block: Upsample -> Concat skip -> ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if shapes don't match exactly
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(
            x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class GuitarSeparatorUNet(nn.Module):
    """
    U-Net for guitar source separation.

    Input:  STFT magnitude spectrogram (B, 1, F, T)
    Output: Soft mask (B, 1, F, T) applied to input to isolate guitar.

    Architecture follows the encoder-decoder pattern with skip connections,
    similar to approaches used in Open-Unmix and Wave-U-Net.
    """

    def __init__(
        self,
        channels: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = 1
        for ch in channels:
            self.encoders.append(ConvBlock(in_ch, ch, dropout))
            in_ch = ch

        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout)

        # Decoder
        self.decoders = nn.ModuleList()
        in_ch = channels[-1] * 2
        for ch in reversed(channels):
            self.decoders.append(UpConvBlock(in_ch, ch, dropout))
            in_ch = ch

        # Output: sigmoid mask
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: STFT magnitude spectrogram (B, 1, F, T)
        Returns:
            Estimated guitar spectrogram (B, 1, F, T)
        """
        skips = []

        # Encode
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Produce mask and apply to input
        mask = self.output_conv(x)
        return mask

    def separate(self, mix_stft_mag: torch.Tensor) -> torch.Tensor:
        """Apply separation: estimate mask and multiply with input."""
        mask = self.forward(mix_stft_mag)
        return mask * mix_stft_mag


# ── Mel-based end-to-end separator ───────────────────────────────────────────

class MelGuitarSeparator(nn.Module):
    """
    End-to-end guitar separator that uses a Mel-spectrogram as the U-Net
    input representation and reconstructs audio without Griffin-Lim by
    preserving the original STFT phase.

    Args:
        n_mels:    Number of Mel frequency bins (U-Net frequency dimension).
        n_fft:     STFT window size for audio ↔ spectrogram conversion.
        hop_length: STFT hop size.
        sample_rate: Audio sample rate (needed for the Mel filterbank).
        fmin:      Lowest Mel frequency (Hz). 80 Hz keeps guitar fundamentals.
        fmax:      Highest Mel frequency (Hz). 8 kHz captures guitar harmonics.
        unet_channels: Channel sizes for the U-Net encoder (default 6 levels).
    """

    def __init__(
        self,
        n_mels:        int        = 128,
        n_fft:         int        = 2048,
        hop_length:    int        = 512,
        sample_rate:   int        = 44100,
        fmin:          float      = 80.0,
        fmax:          float      = 8_000.0,
        unet_channels: list[int]  | None = None,
        dropout:       float      = 0.1,
    ):
        super().__init__()

        self.n_mels      = n_mels
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.sample_rate = sample_rate
        self.fmin        = fmin
        self.fmax        = fmax

        # U-Net operates on Mel bins (128) rather than full STFT bins (~1025)
        self.unet = GuitarSeparatorUNet(
            channels=unet_channels or [16, 32, 64, 128, 256],
            dropout=dropout,
        )

        # Register the Mel filterbank as a non-trainable buffer so it moves
        # to the correct device automatically with .to(device).
        mel_fb = self._build_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
        self.register_buffer("mel_fb",      torch.from_numpy(mel_fb).float())
        # Normalised pseudo-inverse for projecting Mel masks → STFT bins
        mel_bin_sum = mel_fb.sum(axis=0) + 1e-8   # (n_stft_bins,)
        mel_fb_inv  = mel_fb.T / mel_bin_sum[:, None]   # (n_stft_bins, n_mels)
        self.register_buffer("mel_fb_inv",  torch.from_numpy(mel_fb_inv).float())

    # ── Filterbank construction ───────────────────────────────────────────────

    @staticmethod
    def _build_mel_filterbank(
        sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
    ) -> np.ndarray:
        import librosa
        return librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )   # (n_mels, n_stft_bins)

    # ── Forward pass (training) ───────────────────────────────────────────────

    def forward(self, mel_mag: torch.Tensor) -> torch.Tensor:
        """
        Predict a Mel-domain guitar mask from a Mel magnitude spectrogram.

        Args:
            mel_mag: (B, 1, n_mels, T) Mel magnitude spectrogram.

        Returns:
            mask: (B, 1, n_mels, T) soft mask in [0, 1].
        """
        return self.unet(mel_mag)

    # ── Audio-level inference ─────────────────────────────────────────────────

    @torch.no_grad()
    def separate_audio(
        self, audio: np.ndarray, sample_rate: int | None = None
    ) -> np.ndarray:
        """
        Separate guitar from a raw mono audio array.

        Steps:
            1. STFT → save complex spectrogram (phase + magnitude).
            2. Magnitude → Mel-spectrogram → U-Net → Mel mask.
            3. Project Mel mask to STFT bins (normalised filterbank transpose).
            4. Multiply complex STFT by STFT mask (keeps original phase).
            5. iSTFT → audio.

        Args:
            audio:       1-D float32 numpy array (mono).
            sample_rate: Override sample rate (defaults to self.sample_rate).

        Returns:
            Separated guitar waveform as a float32 numpy array.
        """
        import librosa

        sr = sample_rate or self.sample_rate

        # ── Step 1: STFT ──────────────────────────────────────────────────────
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S_mag = np.abs(D).astype(np.float32)       # (n_stft_bins, T)

        # ── Step 2: STFT magnitude → Mel → U-Net mask ────────────────────────
        mel_fb_np = self.mel_fb.cpu().numpy()
        M = (mel_fb_np @ S_mag).astype(np.float32)  # (n_mels, T)

        # Batch dim + channel dim for U-Net: (1, 1, n_mels, T)
        M_t  = torch.from_numpy(M).unsqueeze(0).unsqueeze(0).to(self.mel_fb.device)
        mask_mel = self.unet(M_t)                   # (1, 1, n_mels, T)
        mask_mel = mask_mel.squeeze().cpu().numpy()  # (n_mels, T)

        # ── Step 3: Project Mel mask → STFT bins ─────────────────────────────
        mel_fb_inv_np = self.mel_fb_inv.cpu().numpy()
        W_stft = mel_fb_inv_np @ mask_mel            # (n_stft_bins, T)
        W_stft = np.clip(W_stft, 0.0, 1.0)

        # ── Step 4: Apply to complex STFT (preserves phase) ──────────────────
        D_guitar = D * W_stft

        # ── Step 5: iSTFT → audio ─────────────────────────────────────────────
        audio_out = librosa.istft(
            D_guitar, hop_length=self.hop_length, length=len(audio)
        )
        return audio_out.astype(np.float32)

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", **kwargs
    ) -> "MelGuitarSeparator":
        """Load a trained MelGuitarSeparator from a checkpoint file."""
        model = cls(**kwargs)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device).eval()
