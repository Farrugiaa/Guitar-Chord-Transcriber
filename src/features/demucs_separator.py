"""
Guitar source separation using Meta's Demucs (htdemucs model).

Separates a full mix into drums / bass / other / vocals.
The 'other' stem captures guitar and is returned as a mono float32 array.
"""
from __future__ import annotations

import numpy as np
import torch


class DemucsGuitarSeparator:
    """
    Wraps Demucs htdemucs for high-quality guitar stem extraction.
    Model is lazy-loaded on first call to separate().
    """

    def __init__(self, model_name: str = "htdemucs", device: str | None = None):
        self.model_name = model_name
        self._model = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def separate(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Return (mono_float32, sample_rate) for the guitar ('other') stem."""
        model = self._load_model()
        waveform = self._load_audio(audio_path, model.samplerate, model.audio_channels)

        print(f"  [Demucs] Separating with '{self.model_name}' on {self.device}...")
        stems = self._run_separation(model, waveform)
        print(f"  [Demucs] Done.")

        guitar_tensor = stems.get("other")
        if guitar_tensor is None:
            excluded = {"vocals", "bass", "drums"}
            parts = [v for k, v in stems.items() if k not in excluded]
            guitar_tensor = sum(parts) if parts else stems[next(iter(stems))]

        mono = guitar_tensor.mean(dim=0).cpu().numpy()
        return mono.astype(np.float32), model.samplerate

    def _load_model(self):
        if self._model is None:
            try:
                from demucs.pretrained import get_model
            except ImportError:
                raise ImportError(
                    "demucs is required.\n"
                    "Install with:  pip install demucs"
                )
            print(f"  [Demucs] Loading '{self.model_name}' model...")
            self._model = get_model(self.model_name)
            self._model.to(self.device).eval()
        return self._model

    def _load_audio(self, path: str, target_sr: int, target_channels: int) -> torch.Tensor:
        import soundfile as sf
        import librosa
        from demucs.audio import convert_audio

        try:
            audio_np, sr = sf.read(path, always_2d=True)
            audio_np = audio_np.T.astype("float32")
        except Exception:
            audio_np, sr = librosa.load(path, sr=None, mono=False)
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]

        waveform = torch.from_numpy(audio_np)
        return convert_audio(waveform, sr, target_sr, target_channels)

    def _run_separation(self, model, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        from demucs.apply import apply_model

        # Force deterministic CUDA kernels so the stem is bit-identical across
        # runs. Without this, floating-point operation ordering on the GPU varies,
        # producing subtly different stems that flip borderline tuning/key decisions.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        batch = waveform.unsqueeze(0).to(self.device)
        with torch.no_grad():
            sources = apply_model(model, batch, device=self.device)

        return {
            name: sources[0, i]
            for i, name in enumerate(model.sources)
        }
