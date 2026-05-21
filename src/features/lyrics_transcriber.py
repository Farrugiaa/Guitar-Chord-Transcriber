"""
Lyrics transcription with word-level timestamps using OpenAI Whisper.

Returns a list of segments, each containing words with start/end times,
which are then used to align chord changes to specific words in the lyrics.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WordTimestamp:
    word:  str
    start: float   # seconds
    end:   float   # seconds


@dataclass
class LyricsSegment:
    """One phrase / line of lyrics with per-word timestamps."""
    text:  str
    start: float
    end:   float
    words: list[WordTimestamp]


class LyricsTranscriber:
    """
    Wraps OpenAI Whisper for word-level transcription.

    Args:
        model_size: Whisper model variant.
            "tiny"   — fastest, lowest accuracy (~39 MB)
            "base"   — good balance for English music (~74 MB)  ← default
            "small"  — better accuracy (~244 MB)
            "medium" — best practical accuracy (~769 MB)
        device: "cuda" to use GPU (much faster), "cpu" as fallback.
            Defaults to CUDA if available, otherwise CPU.
    """

    def __init__(self, model_size: str = "base", device: str | None = None):
        self.model_size = model_size
        self._model = None   # lazy-load on first use

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

    def _load_model(self):
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for lyrics transcription. "
                "Install it with: pip install openai-whisper"
            )
        if self._model is None:
            print(f"  [Whisper] Loading '{self.model_size}' model on {self.device}…")
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str) -> list[LyricsSegment]:
        """
        Transcribe an audio file and return word-level timestamps.

        Audio is pre-loaded with librosa at Whisper's required 16 kHz sample
        rate and passed as a numpy array, bypassing the ffmpeg dependency that
        Whisper normally uses for file decoding.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of LyricsSegment objects, one per detected phrase.
        """
        import librosa
        model = self._load_model()

        # Load and resample to 16 kHz mono (Whisper's required format)
        print(f"  [Whisper] Loading audio…")
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        print(f"  [Whisper] Transcribing…")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.transcribe(
                audio,
                word_timestamps=True,
                language="en",         # force English for music
                condition_on_previous_text=False,  # avoid hallucination loops
            )

        segments: list[LyricsSegment] = []
        for seg in result.get("segments", []):
            raw_words = seg.get("words", [])
            words = [
                WordTimestamp(
                    word=w["word"].strip(),
                    start=float(w["start"]),
                    end=float(w["end"]),
                )
                for w in raw_words
                if w.get("word", "").strip()
            ]
            if not words:
                continue
            segments.append(LyricsSegment(
                text=seg.get("text", "").strip(),
                start=float(seg["start"]),
                end=float(seg["end"]),
                words=words,
            ))

        print(f"  [Whisper] Transcribed {len(segments)} segments.")
        return segments
