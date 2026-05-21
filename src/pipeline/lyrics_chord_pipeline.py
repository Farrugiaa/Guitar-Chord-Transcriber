"""
Lyrics + Chord transcription pipeline.

Unlike the main GuitarChordPipeline, this pipeline:
  - Skips guitar source separation entirely (avoids muffling artefacts)
  - Runs chord detection on the raw mixed audio
  - Runs Whisper speech recognition to get word-level lyric timestamps
  - Aligns chord changes to the words being sung at the time of the change
  - Produces a chord-above-lyrics text output (like a guitar chord sheet)

This is complementary to the main pipeline: the main pipeline gives an
accurate structural/analytical view; this pipeline gives an intuitive
performance view aligned to the song's words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.features.audio_features import AudioFeatureExtractor
from src.features.lyrics_transcriber import LyricsTranscriber, LyricsSegment
from src.features.music_analysis import MusicAnalyser, MusicAnalysisResult
from src.pipeline.output_formatter import ChordEvent
from src.theory.chord_builder import ChordBuilder


@dataclass
class LyricsChordResult:
    """Output from the lyrics+chord pipeline."""
    audio_path:      str
    analysis:        MusicAnalysisResult
    segments:        list[LyricsSegment]
    chord_events:    list[ChordEvent]
    formatted_output: str      # chord-above-lyrics text

    def __str__(self) -> str:
        return self.formatted_output


class LyricsChordPipeline:
    """
    Pipeline that produces a chord-above-lyrics sheet from a mixed audio file.

    No guitar extraction is performed — chord detection runs directly on
    the mix, which avoids the filtering artefacts introduced by DSP
    separation while still producing usable chord outlines for common
    songs.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length:  int = 512,
        whisper_model: str = "base",
    ):
        self.sample_rate = sample_rate
        self.hop_length  = hop_length

        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        self.music_analyser = MusicAnalyser(
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        self.chord_builder    = ChordBuilder()
        self.transcriber      = LyricsTranscriber(model_size=whisper_model)

    def process(self, audio_path: str) -> LyricsChordResult:
        """
        Transcribe lyrics and detect chords from a mixed audio file.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            LyricsChordResult with segments, chord events, and formatted text.
        """
        # 1. Load audio as mono (no extraction needed)
        print("  [Lyrics+Chords] Loading audio…")
        waveform, _ = self.feature_extractor.load_audio(audio_path, mono=True)

        # 2. Music analysis (BPM, key, time signature)
        print("  [Lyrics+Chords] Analysing music…")
        analysis = self.music_analyser.analyse(waveform)
        print(f"  [Lyrics+Chords] Key: {analysis.key} {analysis.scale_type}, BPM: {analysis.bpm:.1f}")

        # 3. Chord detection on raw mix (chroma-based, 2-beat windows)
        print("  [Lyrics+Chords] Detecting chords…")
        frame_labels = self._chroma_chord_detection(waveform)
        chord_events = self._frames_to_chord_events(frame_labels)
        print(f"  [Lyrics+Chords] {len(chord_events)} chord events detected.")

        # 4. Whisper lyrics transcription with word timestamps
        print("  [Lyrics+Chords] Transcribing lyrics…")
        segments = self.transcriber.transcribe(audio_path)

        # 5. Format chord-above-lyrics output
        print("  [Lyrics+Chords] Formatting output…")
        formatted = self._format_lyrics_with_chords(
            segments, chord_events, analysis
        )

        return LyricsChordResult(
            audio_path=audio_path,
            analysis=analysis,
            segments=segments,
            chord_events=chord_events,
            formatted_output=formatted,
        )

    # ── Chord detection ───────────────────────────────────────────────────────

    def _chroma_chord_detection(self, waveform: torch.Tensor) -> list[str]:
        """
        2-beat bar-sync chroma chord detection on the raw mix.

        Same algorithm as the main pipeline fallback, but applied to the
        unmixed audio so no filtering artifacts are introduced.
        """
        import librosa

        audio       = waveform.squeeze().numpy()
        total_frames = int(np.ceil(len(audio) / self.hop_length))

        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length,
        )
        _, beat_frames = librosa.beat.beat_track(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length,
        )

        if len(beat_frames) < 2:
            avg  = chroma.mean(axis=1)
            norm = avg / (avg.max() + 1e-8)
            pcs  = [int(p) for p in np.argsort(norm)[::-1] if norm[p] > 0.5][:4]
            sym  = self.chord_builder.identify_chord_from_pitches(pcs) if pcs else "N.C."
            return [sym] * total_frames

        boundaries = np.unique(np.concatenate(
            [[0], beat_frames[::2], [total_frames]]
        )).astype(int)

        frame_labels: list[str] = []
        for i in range(len(boundaries) - 1):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            if end <= start:
                continue
            seg = chroma[:, start:end].mean(axis=1)
            seg_max = seg.max()
            if seg_max < 0.05:
                label = "N.C."
            else:
                norm = seg / (seg_max + 1e-8)
                pcs  = [int(p) for p in np.argsort(norm)[::-1] if norm[p] > 0.45][:5]
                label = self.chord_builder.identify_chord_from_pitches(pcs) if pcs else "N.C."
            frame_labels.extend([label] * (end - start))

        while len(frame_labels) < total_frames:
            frame_labels.append(frame_labels[-1] if frame_labels else "N.C.")
        return frame_labels[:total_frames]

    def _frames_to_chord_events(self, frame_labels: list[str]) -> list[ChordEvent]:
        """Collapse frame-level labels into ChordEvent objects."""
        if not frame_labels:
            return []

        hop_secs = self.hop_length / self.sample_rate
        events: list[ChordEvent] = []
        current = frame_labels[0]
        start   = 0.0

        for i, label in enumerate(frame_labels[1:], start=1):
            if label != current:
                events.append(ChordEvent(
                    symbol=current,
                    start_time=start,
                    end_time=i * hop_secs,
                ))
                current = label
                start   = i * hop_secs

        events.append(ChordEvent(
            symbol=current,
            start_time=start,
            end_time=len(frame_labels) * hop_secs,
        ))

        # Filter out very short events (< 0.5 s) — absorb into predecessor
        merged: list[ChordEvent] = []
        for ev in events:
            if merged and (ev.end_time - ev.start_time) < 0.5:
                prev = merged[-1]
                merged[-1] = ChordEvent(
                    symbol=prev.symbol,
                    start_time=prev.start_time,
                    end_time=ev.end_time,
                )
            else:
                merged.append(ev)

        return merged

    # ── Output formatting ─────────────────────────────────────────────────────

    def _format_lyrics_with_chords(
        self,
        segments:     list[LyricsSegment],
        chord_events: list[ChordEvent],
        analysis:     MusicAnalysisResult,
    ) -> str:
        """
        Build a chord-above-lyrics text sheet.

        For each phrase (Whisper segment):
          - Build a chord line and a lyric line in parallel.
          - A chord symbol is placed above a word only when the chord
            changes at (or within 400 ms before) that word's start time.
          - The column width of each word slot is the max of the word
            length and the chord symbol length, ensuring alignment.

        Example output:
            E              A      Asus4 A
            Sheets of empty canvas

            C                 Em
            Ooh and all I taught her was... everything
        """
        lines: list[str] = [
            f"{analysis.key} {analysis.scale_type}  —  {analysis.bpm:.0f} BPM  "
            f"({analysis.time_signature[0]}/{analysis.time_signature[1]})",
            "",
        ]

        last_chord: str = ""

        for seg in segments:
            if not seg.words:
                continue

            chord_parts: list[str] = []
            lyric_parts: list[str] = []

            for word_info in seg.words:
                word      = word_info.word.strip()
                if not word:
                    continue

                # Find the chord active at this word's start time
                active = self._chord_at(chord_events, word_info.start)

                # Only show a chord symbol when it changes
                show_chord = (active and active != "N.C." and active != last_chord)
                chord_str  = active if show_chord else ""
                if show_chord:
                    last_chord = active

                # Column is wide enough for both chord and word
                width = max(len(word), len(chord_str)) + 1
                chord_parts.append(chord_str.ljust(width))
                lyric_parts.append(word.ljust(width))

            chord_line = "".join(chord_parts).rstrip()
            lyric_line = "".join(lyric_parts).rstrip()

            # Only emit the chord line if it has any symbols on it
            if chord_line.strip():
                lines.append(chord_line)
            lines.append(lyric_line)
            lines.append("")   # blank line between phrases

        return "\n".join(lines)

    @staticmethod
    def _chord_at(events: list[ChordEvent], t: float) -> str:
        """Return the chord symbol active at time t."""
        result = ""
        for ev in events:
            if ev.start_time <= t:
                result = ev.symbol
            else:
                break
        return result
