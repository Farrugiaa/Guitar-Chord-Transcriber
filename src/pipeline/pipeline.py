"""
End-to-end inference pipeline: audio file → chord progression text.
See PROJECT.md for a full architecture walkthrough.
"""

from pathlib import Path
from typing import Optional

import torch
import numpy as np

from src.features.audio_features import AudioFeatureExtractor
from src.features.guitar_extractor import GuitarExtractor
from src.features.demucs_separator import DemucsGuitarSeparator
from src.features.strum_detector import StrumDetector, StrumPattern
from src.theory.tuning import TuningDetector, DEFAULT_TUNING, Tuning
from src.features.music_analysis import MusicAnalyser, MusicAnalysisResult
from src.features.structure_segmenter import StructureSegmenter, SongSection
from src.models.separator import GuitarSeparatorUNet, MelGuitarSeparator
from src.models.chord_recogniser import ChordCRNN, ChordVocabulary
from src.theory.chord_builder import ChordBuilder
from src.theory.scales import ScaleAnalyser
from src.theory.intervals import IntervalAnalyser
from src.pipeline.output_formatter import OutputFormatter, ChordEvent


def _is_minor_maj7(label: str) -> bool:
    """True for minor-major-7 labels like 'Em(maj7)' — triggers chroma retry."""
    if not label or len(label) < 3:
        return False
    i = 1
    if i < len(label) and label[i] in "#b":
        i += 1
    if i < len(label) and label[i] == "m":
        return "maj7" in label[i:]
    return False


class GuitarChordPipeline:
    """
    End-to-end pipeline for extracting guitar chords from audio.

    Usage:
        pipeline = GuitarChordPipeline.from_checkpoints(
            separator_path="checkpoints/separator.pt",
            recogniser_path="checkpoints/chord_recogniser.pt",
        )
        result = pipeline.process("song.wav")
        print(result.formatted_output)
    """

    def __init__(
        self,
        separator: Optional[MelGuitarSeparator] = None,
        recogniser: Optional[ChordCRNN] = None,
        vocabulary: Optional[ChordVocabulary] = None,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_fft: int = 4096,
        device: str = "cpu",
        tuning: Optional[Tuning] = None,   # None = auto-detect
    ):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        self.music_analyser = MusicAnalyser(
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        self.chord_builder    = ChordBuilder()
        self.guitar_extractor = GuitarExtractor()
        self._forced_tuning   = tuning
        self._demucs          = None  # lazy-loaded on first use
        self.structure_segmenter = StructureSegmenter(
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        self.formatter = OutputFormatter()

        self.vocabulary = vocabulary or ChordVocabulary(level="extended")

        self.separator = separator
        if separator:
            self.separator.to(self.device).eval()

        self.recogniser = recogniser
        if recogniser:
            self.recogniser.to(self.device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        separator_path: Optional[str] = None,
        recogniser_path: Optional[str] = None,
        vocab_level: str = "extended",
        device: str = "auto",
        tuning: Optional[Tuning] = None,
    ) -> "GuitarChordPipeline":
        """Load pipeline from saved model checkpoints."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        separator = None
        if separator_path and Path(separator_path).exists():
            separator = MelGuitarSeparator.from_checkpoint(
                separator_path, device=device
            )

        vocabulary = ChordVocabulary(level=vocab_level)

        recogniser = None
        if recogniser_path and Path(recogniser_path).exists():
            recogniser = ChordCRNN(n_classes=len(vocabulary))
            state = torch.load(recogniser_path, map_location=device, weights_only=True)
            recogniser.load_state_dict(state)

        return cls(
            separator=separator,
            recogniser=recogniser,
            vocabulary=vocabulary,
            device=device,
            tuning=tuning,
        )

    def process(self, audio_path: str, pre_separated: bool = False) -> "PipelineResult":
        """
        Process an audio file through the full pipeline.

        Args:
            audio_path:     Path to the input audio file.
            pre_separated:  If True the file is already a guitar-only stem —
                            skip Demucs/U-Net and use the audio directly.

        Returns:
            PipelineResult with analysis, chords, and formatted output.
        """
        # Keep stereo for loading — vocal removal uses both channels
        waveform, _ = self.feature_extractor.load_audio(audio_path, mono=False)


        if pre_separated:
            # File is already a guitar-only stem; downmix to mono and use directly.
            print("  [Guitar extraction] Skipped — using pre-separated guitar stem.")
            guitar_waveform = (
                waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
            )
        elif self.separator:
            print("  [Guitar extraction] Running Mel U-Net separator...")
            mono = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
            guitar_waveform = self._separate_guitar(mono)
            print("  [Guitar extraction] Done.")
        else:
            print("  [Guitar extraction] Running Demucs source separation...")
            if self._demucs is None:
                self._demucs = DemucsGuitarSeparator(device=self.device.type)
            guitar_np, demucs_sr = self._demucs.separate(audio_path)
            if demucs_sr != self.sample_rate:
                import librosa
                guitar_np = librosa.resample(
                    guitar_np, orig_sr=demucs_sr, target_sr=self.sample_rate
                )
            guitar_waveform = torch.from_numpy(guitar_np).unsqueeze(0)
            print("  [Guitar extraction] Done.")

        # Tuning detection on guitar stem so pYIN doesn't pick up vocals or bass
        guitar_mono = guitar_waveform.squeeze().numpy()
        if self._forced_tuning is not None:
            tuning = self._forced_tuning
            tuning_confidence = 1.0
        else:
            print("  [Tuning] Detecting tuning from guitar signal...")
            tuning_detector = TuningDetector()
            tuning, tuning_confidence = tuning_detector.detect(guitar_mono, self.sample_rate)
        print(f"  [Tuning] {tuning.name}  (confidence: {tuning_confidence:.2f})")

        analysis = self.music_analyser.analyse(guitar_waveform)


        print("  [Strum] Detecting strumming pattern…")
        guitar_mono_np = guitar_waveform.squeeze().numpy()
        strum_pattern  = StrumDetector().detect(
            audio=guitar_mono_np,
            sample_rate=self.sample_rate,
            bpm=analysis.bpm,
            time_signature=analysis.time_signature,
        )
        print(f"  [Strum] Done. Pattern: "
              f"{''.join(b.direction or '·' for b in strum_pattern.beats)}")

        cqt = self.feature_extractor.compute_cqt(guitar_waveform)

        # Chroma is always computed — it feeds the UI confidence display and
        # acts as a fallback when the CRNN confidence is below threshold.
        chroma_labels, window_confidence_data = self._chroma_chord_detection(
            guitar_waveform, key=analysis.key, scale_type=analysis.scale_type,
            tuning=tuning,
        )

        if self.recogniser:
            chord_indices, frame_confidences = self._recognise_chords(cqt)
            chord_indices = self._median_smooth(chord_indices, kernel=9)
            crnn_labels = [
                self.vocabulary.decode(idx) for idx in chord_indices
            ]
            n = min(len(crnn_labels), len(chroma_labels), len(frame_confidences))
            # 0.35: in-domain model hits 0.9+; wrong predictions on OOD audio sit below 0.3
            threshold = 0.35
            frame_labels = [
                crnn_labels[i] if frame_confidences[i] >= threshold
                else chroma_labels[i]
                for i in range(n)
            ]
        else:
            frame_labels = chroma_labels

        # Tab-derived chords: fret+tuning → exact pitch classes — more authoritative
        # than spectral chroma. onset_root_pc intentionally skipped: when the CNN mutes
        # the bass string the "lowest active" string gives the wrong root.
        try:
            from src.features.tab_exporter import PitchTabExporter as _PTE
            _ckpt = Path(__file__).parent.parent.parent / "guitar_only" / "checkpoints" / "tab_cnn_best.pt"
            _tab_exp = _PTE(
                sample_rate=self.sample_rate,
                tuning=tuning,
                tab_cnn_checkpoint=str(_ckpt) if _ckpt.exists() else None,
            )
            _scale_pcs = (
                set(ScaleAnalyser().get_scale_pitch_classes(analysis.key, analysis.scale_type))
                if analysis.key else None
            )
            tab_labels = _tab_exp.detect_chords(
                guitar_waveform,
                beat_times=analysis.beat_times,
                bpm=analysis.bpm,
                time_signature=analysis.time_signature,
                key=analysis.key,
                scale_pcs=_scale_pcs,
            )
            n = min(len(tab_labels), len(frame_labels))
            frame_labels = [
                tab_labels[i] if tab_labels[i] != "N.C." else frame_labels[i]
                for i in range(n)
            ] + frame_labels[n:]
        except Exception as _e:
            print(f"  [Tab chords] skipped: {_e}")

        # Snap non-diatonic roots to the nearest key-diatonic chord
        if analysis.key_confidence >= 0.4:
            frame_labels = self._snap_chords_to_key(
                frame_labels, analysis.key, analysis.scale_type
            )

        # Replace complex-extension labels with expected diatonic quality.
        # TabCNN sympathetic resonance produces names like "Ebmaddb13" — the key
        # says Eb is minor (i), so that's the correct answer. 7th chords are kept.
        if analysis.key_confidence >= 0.4:
            frame_labels = self._validate_chord_quality_with_key(
                frame_labels, analysis.key, analysis.scale_type
            )

        # Chord builder outputs all-sharps internally; convert to flats for flat keys
        frame_labels = self._respell_for_key(frame_labels, analysis.key)

        sections = self.structure_segmenter.segment(guitar_waveform)

        chord_events = self.formatter.frames_to_chord_events(
            frame_labels,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )

        if window_confidence_data:
            for event in chord_events:
                mid_frame = int(
                    (event.start_time + event.end_time) / 2
                    * self.sample_rate / self.hop_length
                )
                for w_start, w_end, conf in window_confidence_data:
                    if w_start <= mid_frame < w_end:
                        event.note_confidences = conf
                        break

        formatted = self.formatter.format_progression(
            chord_events,
            bpm=analysis.bpm,
            time_signature=analysis.time_signature,
            key=f"{analysis.key} {analysis.scale_type}",
            sections=sections,
        )

        result = PipelineResult(
            audio_path=audio_path,
            analysis=analysis,
            chord_events=chord_events,
            frame_labels=frame_labels,
            formatted_output=formatted,
            guitar_waveform=guitar_waveform,
            sample_rate=self.sample_rate,
            sections=sections,
            tuning=tuning,
            tuning_confidence=tuning_confidence,
            strum_pattern=strum_pattern,
        )

        from pathlib import Path as _Path
        tab_path = str(_Path(audio_path).with_name(_Path(audio_path).stem + "_tab.txt"))
        try:
            result.export_tab(tab_path)
            print(f"  [Tab] Saved: {tab_path}")
        except Exception as _e:
            print(f"  [Tab] Export failed: {_e}")

        return result

    def _snap_chords_to_key(
        self, frame_labels: list[str], key: str, scale_type: str
    ) -> list[str]:
        """
        Post-process chord labels so that any chord whose root falls outside
        the detected key is replaced by the nearest diatonic triad.
        Chords whose root is already a scale note are kept unchanged.
        """
        scale_analyser = ScaleAnalyser()
        scale_pcs = set(scale_analyser.get_scale_pitch_classes(key, scale_type))
        diatonic = scale_analyser.get_diatonic_chords(key, scale_type, include_sevenths=False)
        pc_to_diatonic: dict[int, str] = {
            IntervalAnalyser.note_to_pitch_class(c["root"]): c["symbol"]
            for c in diatonic
        }

        result: list[str] = []
        for label in frame_labels:
            if label in ("N.C.", ""):
                result.append(label)
                continue
            root_str = self._parse_chord_root(label)
            if root_str is None:
                result.append(label)
                continue
            root_pc = IntervalAnalyser.note_to_pitch_class(root_str)
            if root_pc in scale_pcs:
                # Root is diatonic — keep it unless the quality is UNKNOWN ("?")
                if label.endswith("?"):
                    result.append(pc_to_diatonic.get(root_pc, label))
                else:
                    result.append(label)
            else:
                nearest_pc = min(
                    scale_pcs,
                    key=lambda pc: min(abs(pc - root_pc), 12 - abs(pc - root_pc)),
                )
                result.append(pc_to_diatonic.get(nearest_pc, label))
        return result

    @staticmethod
    def _simplify_chord(label: str) -> str:
        """Reduce chord labels to root + basic triad quality (major/minor/dim/aug).

        7th chords and all higher extensions are stripped back to the triad.
        The chord builder over-detects 7ths when adjacent chord tones bleed
        into a window; showing the simpler triad matches guitar chord-sheet
        convention and avoids misleading the player.

        Examples: Ebm7→Ebm, Bmaj7→B, Db7→Db, Gbsus4(13)→Gb, Ebm9→Ebm.
        """
        if not label or label == "N.C.":
            return label

        # Extract root (letter + optional accidental)
        i = 1
        if i < len(label) and label[i] in '#b':
            i += 1
        root = label[:i]
        remainder = label[i:]

        # Quality pairs: (detected_prefix, simplified_output).
        # Checked in priority order; 'm7' before 'm' prevents 'm' matching 'maj'.
        # All 7th/extended variants collapse to their base triad quality.
        QUALITY_PAIRS = [
            ('m7',   'm'),    # Ebm7  → Ebm
            ('maj7', ''),     # Bmaj7 → B
            ('dim7', 'dim'),  # Bdim7 → Bdim
            ('m',    'm'),
            ('dim',  'dim'),
            ('aug',  'aug'),
            ('7',    ''),     # F7, Db7 → F, Db
            ('',     ''),
        ]
        for q_key, q_out in QUALITY_PAIRS:
            if not q_key:
                return root + q_out
            if remainder.startswith(q_key):
                after = remainder[len(q_key):]
                if (not after
                        or after[0].isdigit()
                        or after[0] in '(,/'
                        or after.startswith(('add', 'sus', 'b', '#'))):
                    return root + q_out
        return root

    @staticmethod
    def _validate_chord_quality_with_key(
        frame_labels: list[str], key: str, scale_type: str
    ) -> list[str]:
        """Replace complex-extension labels with their expected diatonic chord.

        When the chord's root is a diatonic scale degree but the detected label
        contains sus/add/parenthesised extensions or altered-interval suffixes
        (b13, #11…), substitute the expected diatonic chord symbol for that root.
        Plain triads and 7th chords (m7, maj7, 7, dim7) are passed through as-is.

        Example (Eb minor key):
            Ebmaddb13  →  Ebm     (i chord)
            Gbsus4(13) →  Gb      (III chord)
            Dbsus2(9)  →  Db      (VII chord)
            Badd9      →  B       (VI chord)
            Ebm7       →  Ebm7    (kept — 7th may be intentional)
        """
        import re
        _COMPLEX = re.compile(r'(sus|add|\(|b\d|#\d)')

        scale_analyser = ScaleAnalyser()
        diatonic = scale_analyser.get_diatonic_chords(key, scale_type, include_sevenths=False)
        pc_to_symbol: dict[int, str] = {
            IntervalAnalyser.note_to_pitch_class(c["root"]): c["symbol"]
            for c in diatonic
        }

        result: list[str] = []
        for label in frame_labels:
            if not label or label == "N.C.":
                result.append(label)
                continue
            if _COMPLEX.search(label):
                root_str = GuitarChordPipeline._parse_chord_root(label)
                if root_str:
                    root_pc = IntervalAnalyser.note_to_pitch_class(root_str)
                    if root_pc in pc_to_symbol:
                        result.append(pc_to_symbol[root_pc])
                        continue
            result.append(label)
        return result

    @staticmethod
    def _respell_for_key(frame_labels: list[str], key: str) -> list[str]:
        """Convert sharp roots (D#, F#, C#, G#, A#) to flat equivalents in flat keys."""
        _FLAT_KEY_ROOTS = {'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'}
        if key not in _FLAT_KEY_ROOTS:
            return frame_labels
        _S2F = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
        result = []
        for label in frame_labels:
            if not label or label == 'N.C.':
                result.append(label)
                continue
            root = label[:2] if len(label) > 1 and label[1] == '#' else label[:1]
            result.append(_S2F.get(root, root) + label[len(root):])
        return result

    @staticmethod
    def _parse_chord_root(symbol: str) -> Optional[str]:
        """Extract the root note (e.g. 'C#', 'Bb', 'G') from a chord symbol."""
        if not symbol:
            return None
        root = symbol[0]
        if len(symbol) > 1 and symbol[1] in ("#", "b"):
            root += symbol[1]
        return root

    def _separate_guitar(self, waveform: torch.Tensor) -> torch.Tensor:
        """Isolate guitar via MelGuitarSeparator (STFT→Mel→U-Net→iSTFT, phase preserved)."""
        audio = waveform.squeeze().numpy()
        audio_out = self.separator.separate_audio(audio, sample_rate=self.sample_rate)
        return torch.from_numpy(audio_out).unsqueeze(0)

    def _recognise_chords(self, cqt: torch.Tensor) -> tuple[list[int], list[float]]:
        """Run CRNN on CQT. Returns (predicted_indices, per-frame max softmax confidence)."""
        if cqt.dim() == 3:
            cqt = cqt.unsqueeze(0)

        with torch.no_grad():
            cqt = cqt.to(self.device)
            logits = self.recogniser(cqt)  # (1, T, n_classes)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (T, C)
            confs, preds = probs.max(dim=-1)

        return preds.cpu().tolist(), confs.cpu().tolist()

    @staticmethod
    def _median_smooth(indices: list[int], kernel: int = 9) -> list[int]:
        """Median-filter a sequence of class indices to remove flicker."""
        if kernel <= 1 or len(indices) < kernel:
            return list(indices)
        half = kernel // 2
        out = list(indices)
        for i in range(half, len(indices) - half):
            window = indices[i - half : i + half + 1]
            # Mode over a small window == median for label sequences
            counts: dict[int, int] = {}
            for v in window:
                counts[v] = counts.get(v, 0) + 1
            out[i] = max(counts.items(), key=lambda kv: kv[1])[0]
        return out

    def _chroma_chord_detection(
        self,
        waveform: torch.Tensor,
        key: Optional[str] = None,
        scale_type: Optional[str] = None,
        tuning: Optional[Tuning] = None,
    ) -> tuple[list[str], list[tuple[int, int, dict]]]:
        """
        Chroma-based chord detection over 2-beat half-bar windows.
        Returns (frame_labels, window_confidence_data).
        """
        import librosa

        scale_pcs: Optional[set[int]] = None
        if key and scale_type:
            scale_pcs = set(ScaleAnalyser().get_scale_pitch_classes(key, scale_type))

        audio = waveform.squeeze().numpy()
        total_frames = int(np.ceil(len(audio) / self.hop_length))

        # Sliding-window RMS normalisation — equalises gain between soft fingerpicked
        # and loud strummed passages so the onset detector doesn't miss quiet beats.
        win_samples = int(2.0 * self.sample_rate)
        hop_samples = win_samples // 2
        audio_norm  = np.zeros_like(audio)
        weight_sum  = np.zeros_like(audio)
        window      = np.hanning(win_samples).astype(np.float32)

        for start in range(0, len(audio) - win_samples + 1, hop_samples):
            seg  = audio[start : start + win_samples]
            rms  = np.sqrt(np.mean(seg ** 2))
            gain = 1.0 / (rms + 1e-6)
            gain = min(gain, 20.0)   # cap at 20× to avoid amplifying pure silence
            audio_norm[start : start + win_samples] += seg * gain * window
            weight_sum[start : start + win_samples] += window

        safe = weight_sum > 1e-8
        audio_norm[safe] /= weight_sum[safe]
        # Normalised signal → onset/beat detection; original → chroma so silent
        # sections still produce low energy and get labelled N.C.

        # CENS chroma: log compression + temporal smoothing + L2 normalisation per
        # frame gives stable averages under fingerpicking dynamics and bass bleed.
        chroma = librosa.feature.chroma_cens(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length,
        )

        # Normalised audio so fingerpicked soft-attack notes contribute to onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio_norm, sr=self.sample_rate, hop_length=self.hop_length,
        )
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        # Bass string hit first on strum → dominant pitch class at onset = chord root.
        # Disambiguates e.g. G major vs Em7 (same pitch-class set).
        onset_frames_arr = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,   # snap to energy minimum before the peak
        )
        onset_set = set(int(f) for f in onset_frames_arr)

        from src.theory.intervals import NOTE_NAMES as _NOTE_NAMES

        if len(beat_frames) < 2:
            # Fallback: single window over entire track
            avg = chroma.mean(axis=1)
            norm = avg / (avg.max() + 1e-8)
            sorted_pcs = np.argsort(norm)[::-1]
            active_pcs = [int(p) for p in sorted_pcs if norm[p] > 0.6][:3]
            symbol = self.chord_builder.identify_chord_from_pitches(
                active_pcs, key_context=key, scale_pcs=scale_pcs
            ) if active_pcs else "N.C."
            conf = {_NOTE_NAMES[p]: float(norm[p]) for p in active_pcs}
            return [symbol] * total_frames, [(0, total_frames, conf)]

        # At <80 BPM a single beat (~0.8 s) isn't long enough for fingerpicking to
        # cover the full chord — use 2-beat windows to get stable chroma coverage.
        if len(beat_frames) >= 2:
            avg_beat_dur = (
                (int(beat_frames[-1]) - int(beat_frames[0]))
                / max(len(beat_frames) - 1, 1)
                * self.hop_length / self.sample_rate
            )
            detected_bpm = 60.0 / avg_beat_dur if avg_beat_dur > 0 else 120.0
        else:
            detected_bpm = 120.0
        beats_per_window = 2 if detected_bpm < 80 else 1
        window_starts = beat_frames[::beats_per_window]
        boundaries = np.unique(np.concatenate([[0], window_starts, [total_frames]])).astype(int)

        # Open-string pitch classes for root-note weighting
        open_string_pcs: Optional[set[int]] = None
        if tuning is not None:
            open_string_pcs = set(n % 12 for n in tuning.midi_notes)

        window_labels:      list[str] = []
        window_spans:       list[tuple[int, int]] = []
        window_confidences: list[dict] = []

        for i in range(len(boundaries) - 1):
            start = int(boundaries[i])
            end   = int(boundaries[i + 1])
            if end <= start:
                continue

            seg = chroma[:, start:end]
            n   = seg.shape[1]

            # 2× weight on the second half of each window: old strings decay there,
            # new chord tones are fully established. Reduces "ringing contamination"
            # from the previous chord without discarding the attack energy.
            if n >= 4:
                weights = np.ones(n)
                weights[n // 2:] = 2.0
                seg_chroma = (seg * weights).sum(axis=1) / weights.sum()
            else:
                seg_chroma = seg.mean(axis=1)
            seg_chroma = seg_chroma.copy()

            if open_string_pcs:
                for pc in open_string_pcs:
                    seg_chroma[pc] *= 1.25

            # >15% onset energy gate: genuine strummed root spikes well above this;
            # fingerpicking distributes energy broadly and won't clear it.
            onset_root_pc: Optional[int] = None
            window_onsets = [f for f in onset_set if start <= f < end]
            if window_onsets:
                of = window_onsets[0]
                of_end = min(chroma.shape[1], of + 4)
                onset_col = chroma[:, of:of_end].mean(axis=1).copy()
                onset_total = onset_col.sum() + 1e-8
                onset_col_norm = onset_col / onset_total
                best_pc = int(np.argmax(onset_col_norm))
                if onset_col_norm[best_pc] > 0.15:
                    onset_root_pc = best_pc

            seg_max = seg_chroma.max()

            if seg_max < 0.05:
                label = "N.C."
                conf: dict = {}
            else:
                # 0.08 threshold (lowered from 0.10) keeps weaker chord tones
                # that are still establishing when a previous chord is ringing out
                l1 = seg_chroma.sum() + 1e-8
                norm_l1 = seg_chroma / l1
                sorted_pcs = np.argsort(norm_l1)[::-1]
                active_pcs = [int(p) for p in sorted_pcs if norm_l1[p] > 0.08][:5]
                label = self.chord_builder.identify_chord_from_pitches(
                    active_pcs, key_context=key, scale_pcs=scale_pcs,
                    onset_root_pc=onset_root_pc,
                ) if active_pcs else "N.C."

                # minor-maj7 almost never appears in pop/rock — it usually means
                # previous-chord open strings are ringing. Retry at lower threshold
                # (0.06) so the genuine new chord tones can tip the balance.
                # B7-specific trigger: Em + D#/Eb energy is a strong signal that
                # Em open strings are masking an incoming B7.
                _b7_signal = (
                    label.startswith("Em")
                    and norm_l1[3] > 0.06   # D#/Eb has meaningful energy
                    and 11 in active_pcs    # B is present (root of B7, 5th of Em)
                )
                if _is_minor_maj7(label) or _b7_signal:
                    extended_pcs = [int(p) for p in sorted_pcs if norm_l1[p] > 0.06][:6]
                    if len(extended_pcs) > len(active_pcs):
                        retry = self.chord_builder.identify_chord_from_pitches(
                            extended_pcs, key_context=key, scale_pcs=scale_pcs,
                            onset_root_pc=onset_root_pc,
                        )
                        if retry and not _is_minor_maj7(retry):
                            label = retry
                            active_pcs = extended_pcs

                conf = {_NOTE_NAMES[p]: float(norm_l1[p]) for p in active_pcs}

            window_labels.append(label)
            window_spans.append((start, end))
            window_confidences.append(conf)

        # Two passes remove isolated single-window spikes without cascading into
        # genuine alternating progressions (convergence would flatten G B G B → G).
        for _pass in range(2):
            if len(window_labels) < 3:
                break
            for i in range(1, len(window_labels) - 1):
                if (window_labels[i - 1] == window_labels[i + 1]
                        and window_labels[i] != window_labels[i - 1]):
                    window_labels[i] = window_labels[i - 1]

        frame_labels: list[str] = []
        window_data: list[tuple[int, int, dict]] = []
        for label, (start, end), conf in zip(window_labels, window_spans, window_confidences):
            frame_labels.extend([label] * (end - start))
            window_data.append((start, end, conf))

        while len(frame_labels) < total_frames:
            frame_labels.append(frame_labels[-1] if frame_labels else "N.C.")
        return frame_labels[:total_frames], window_data


class PipelineResult:
    """Container for pipeline output."""

    def __init__(
        self,
        audio_path: str,
        analysis: MusicAnalysisResult,
        chord_events: list[ChordEvent],
        frame_labels: list[str],
        formatted_output: str,
        guitar_waveform: Optional[torch.Tensor] = None,
        sample_rate: int = 44100,
        sections: Optional[list[SongSection]] = None,
        tuning=None,
        tuning_confidence: float = 0.0,
        strum_pattern=None,
    ):
        self.audio_path = audio_path
        self.analysis = analysis
        self.chord_events = chord_events
        self.frame_labels = frame_labels
        self.formatted_output = formatted_output
        self.guitar_waveform = guitar_waveform
        self.tuning = tuning or DEFAULT_TUNING
        self.tuning_confidence = tuning_confidence
        self.sample_rate = sample_rate
        self.sections = sections or []
        self.strum_pattern = strum_pattern

    def save_guitar_audio(self, output_path: str) -> str:
        """Save the extracted guitar audio to a WAV file."""
        import soundfile as sf

        if self.guitar_waveform is None:
            raise ValueError("No guitar waveform available")

        audio_np = self.guitar_waveform.squeeze().numpy()
        sf.write(output_path, audio_np, self.sample_rate)
        return output_path

    def export_midi(self, output_path: str) -> str:
        """Export chord progression as MIDI (simultaneous notes per voicing)."""
        from src.features.midi_exporter import MidiExporter
        from src.theory.chord_voicings import ChordVoicingEngine

        engine = ChordVoicingEngine(tuning=self.tuning)
        exporter = MidiExporter(voicing_engine=engine, tuning=self.tuning)

        pm = exporter.export(
            self.chord_events,
            bpm=self.analysis.bpm,
            time_signature=self.analysis.time_signature,
        )
        pm.write(output_path)
        return output_path

    def export_tab(self, output_path: str) -> str:
        """Export full-song ASCII guitar tab using beat-window CQT pitch detection."""
        from src.features.tab_exporter import PitchTabExporter
        from pathlib import Path

        name = Path(self.audio_path).stem

        # Auto-load TabCNN if the trained checkpoint exists
        _ckpt = Path(__file__).parent.parent.parent / "guitar_only" / "checkpoints" / "tab_cnn_best.pt"
        tab_cnn_checkpoint = str(_ckpt) if _ckpt.exists() else None
        if tab_cnn_checkpoint:
            print(f"  [TabCNN] Loading checkpoint: {_ckpt.name}")

        exporter = PitchTabExporter(
            sample_rate=self.sample_rate,
            tuning=self.tuning,
            tab_cnn_checkpoint=tab_cnn_checkpoint,
        )
        tab_text = exporter.export(
            waveform=self.guitar_waveform,
            beat_times=self.analysis.beat_times,
            bpm=self.analysis.bpm,
            time_signature=self.analysis.time_signature,
            sections=self.sections,
            song_name=name,
            chord_events=self.chord_events,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tab_text)
        return output_path

    def __str__(self) -> str:
        return self.formatted_output

    def summary(self) -> str:
        """Short summary of analysis results."""
        a = self.analysis
        header = (
            f"File: {self.audio_path}\n"
            f"Key: {a.key} {a.scale_type} (confidence: {a.key_confidence:.2f})\n"
            f"BPM: {a.bpm:.1f}\n"
            f"Time Signature: {a.time_signature[0]}/{a.time_signature[1]}\n"
            f"Chords detected: {len(self.chord_events)}\n"
        )
        return f"{header}\n{self.formatted_output}"
