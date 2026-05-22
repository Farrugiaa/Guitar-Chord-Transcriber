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
    """End-to-end pipeline: audio file → chord progression."""

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
        """Run the full pipeline on an audio file. pre_separated skips source separation."""
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

        # Polyphonic audio-to-MIDI via basic-pitch (ONNX backend, no TF needed)
        print("  [Audio-to-MIDI] Running basic-pitch transcription...")
        basic_pitch_midi = self._run_basic_pitch(guitar_waveform, self.sample_rate)
        print("  [Audio-to-MIDI] Done." if basic_pitch_midi else "  [Audio-to-MIDI] Skipped.")

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
            beat_times=analysis.beat_times,
        )
        print(f"  [Strum] Done. Pattern: "
              f"{''.join(b.direction or '·' for b in strum_pattern.beats)}")

        cqt = self.feature_extractor.compute_cqt(guitar_waveform)

        # Chroma feeds confidence display and acts as fallback when CRNN is below threshold
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
            # 0.35 threshold: in-domain hits 0.9+, out-of-domain wrong predictions sit below 0.3
            threshold = 0.35
            frame_labels = [
                crnn_labels[i] if frame_confidences[i] >= threshold
                else chroma_labels[i]
                for i in range(n)
            ]
        else:
            frame_labels = chroma_labels

        # Tab-derived chords: fret+tuning → exact pitch classes, more accurate than chroma
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

        # Final vocabulary filter: keep only major, minor, 7th, maj7, m7, power, sus
        frame_labels = [self._simplify_chord(lbl) for lbl in frame_labels]

        # MIDI override: replaces chroma labels where basic-pitch detected real chords
        if basic_pitch_midi is not None:
            frame_labels = self._apply_midi_chord_override(
                frame_labels, basic_pitch_midi,
                analysis.bpm, analysis.key, analysis.scale_type,
                tuning=tuning,
            )

        sections = self.structure_segmenter.segment(guitar_waveform)

        chord_events = self.formatter.frames_to_chord_events(
            frame_labels,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            bpm=analysis.bpm,
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
            basic_pitch_midi=basic_pitch_midi,
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
        """Snap non-diatonic roots to the nearest scale note. Keeps borrowed chords (bVII, bIII, bVI)."""
        scale_analyser = ScaleAnalyser()
        scale_pcs = set(scale_analyser.get_scale_pitch_classes(key, scale_type))
        diatonic = scale_analyser.get_diatonic_chords(key, scale_type, include_sevenths=False)
        pc_to_diatonic: dict[int, str] = {
            IntervalAnalyser.note_to_pitch_class(c["root"]): c["symbol"]
            for c in diatonic
        }

        # roots from the parallel minor are valid borrowed chords — keep them
        tonic_str = key.split()[0] if " " in key else key
        tonic_pc = IntervalAnalyser.note_to_pitch_class(tonic_str)
        _minor_intervals = (0, 2, 3, 5, 7, 8, 10)
        parallel_minor_pcs = {(tonic_pc + i) % 12 for i in _minor_intervals}

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
                # Diatonic root — keep, but resolve unknown quality chords
                if label.endswith("?"):
                    result.append(pc_to_diatonic.get(root_pc, label))
                else:
                    result.append(label)
            elif root_pc in parallel_minor_pcs:
                result.append(label)
            else:
                # genuinely non-diatonic — snap upward (CQT smear biases downward)
                nearest_up = min(
                    scale_pcs,
                    key=lambda pc: (pc - root_pc) % 12,
                )
                result.append(pc_to_diatonic.get(nearest_up, label))
        return result

    @staticmethod
    def _simplify_chord(label: str) -> str:
        """Reduce to: major, minor, 7, maj7, m7, power (5), sus2/sus4."""
        if not label or label in ("N.C.", ""):
            return label

        i = 1
        if i < len(label) and label[i] in "#b":
            i += 1
        root   = label[:i]
        suffix = label[i:]

        # split slash chords, simplify root, reattach bass note
        bass = ""
        if "/" in suffix:
            slash_idx = suffix.index("/")
            bass   = suffix[slash_idx:]   # e.g. "/B"
            suffix = suffix[:slash_idx]

        if suffix == "5" or suffix.startswith("5("):
            return root + "5" + bass

        if "sus" in suffix:
            return root + ("sus2" if "sus2" in suffix else "sus4") + bass

        if suffix.startswith(("m7b5", "ø", "m7♭5")):
            return root + "m7" + bass

        if suffix.startswith("m7") or suffix.startswith("min7"):
            return root + "m7" + bass

        # 'maj' must come before 'm' check
        if suffix.startswith("maj") or suffix.startswith("M7"):
            if suffix.startswith(("maj7", "M7")):
                return root + "maj7" + bass
            if len(suffix) > 3 and suffix[3].isdigit():
                return root + "maj7" + bass   # maj9/11/13 → maj7
            return root + bass

        if suffix.startswith("m") and len(suffix) > 1 and suffix[1].isdigit():
            return root + "m7" + bass   # m9, m11, m13 → m7

        if suffix.startswith(("m", "min", "dim")):
            return root + "m" + bass

        if suffix.startswith(("7", "9", "11", "13")):
            return root + "7" + bass

        if suffix.startswith("aug"):
            return root + bass

        return root + bass

    @staticmethod
    def _validate_chord_quality_with_key(
        frame_labels: list[str], key: str, scale_type: str
    ) -> list[str]:
        """Replace sus/add/extended labels with the expected diatonic chord. Plain triads and 7ths pass through."""
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

    def _apply_midi_chord_override(
        self,
        frame_labels: list[str],
        pm,
        bpm: float,
        key: str,
        scale_type: str,
        tuning=None,
    ) -> list[str]:
        """Three-pass MIDI override: suppress intro (pass 0), mark riff frames N.C. (pass 1), assign chord labels (pass 2)."""
        from src.features.midi_chord_analyser import MidiChordAnalyser

        analyser = MidiChordAnalyser(tuning=tuning)
        chord_events = analyser.analyse(pm, bpm, key=key, scale_type=scale_type)

        # Collect every MIDI note (with the same pitch ceiling as the analyser)
        all_notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)
        if analyser.max_pitch is not None:
            all_notes = [n for n in all_notes if n.pitch <= analyser.max_pitch]

        if not all_notes and not chord_events:
            return frame_labels

        hop_sec = self.hop_length / self.sample_rate
        result  = list(frame_labels)

        # Pass 0: if the song starts with non-guitar audio, mark everything before first chord as N.C.
        if chord_events and chord_events[0].start_time > 2.0:
            pre_end = int(chord_events[0].start_time / hop_sec)
            for f in range(min(pre_end, len(result))):
                result[f] = "N.C."

        # Pass 1: mark every MIDI note frame as N.C. (riff suppression). Skip artefacts < 50ms.
        riff_covered: set[int] = set()
        for note in all_notes:
            if (note.end - note.start) < 0.05:
                continue
            s = int(note.start / hop_sec)
            e = int(note.end   / hop_sec)
            riff_covered.update(range(s, min(e + 1, len(result))))

        for f in riff_covered:
            result[f] = "N.C."

        # Pass 2: write MIDI chord labels over the chord event frames.
        for event in chord_events:
            symbol = self._simplify_chord(
                self._respell_for_key([event.symbol], key)[0]
            )
            if not symbol or symbol == "N.C.":
                continue
            s = int(event.start_time / hop_sec)
            e = int(event.end_time   / hop_sec)
            for f in range(s, min(e, len(result))):
                result[f] = symbol

        return result

    @staticmethod
    def _run_basic_pitch(guitar_waveform: torch.Tensor, sample_rate: int):
        """Polyphonic pitch detection via basic-pitch ONNX model."""
        import tempfile, os, logging
        import soundfile as sf

        # suppress backend-not-found warnings — we only use the ONNX backend
        _prev = logging.root.level
        logging.root.setLevel(logging.ERROR)
        try:
            from basic_pitch.inference import predict, Model
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            logging.root.setLevel(_prev)
            return None
        finally:
            logging.root.setLevel(_prev)

        audio_np = guitar_waveform.squeeze().numpy()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_np, sample_rate)
            model = Model(ICASSP_2022_MODEL_PATH)
            _, midi_data, _ = predict(tmp_path, model)
            return midi_data
        except Exception as e:
            print(f"  [Audio-to-MIDI] Error: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

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

        # Sliding-window RMS normalisation so quiet and loud passages are treated equally
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

        # CQT chroma — E2 to D#6, covers all guitar chord tones
        _fmin = librosa.note_to_hz('E2')
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length,
            fmin=_fmin, n_octaves=4,
        )

        # Flatness near 1.0 = noise/muted strum, near 0 = tonal — gates out dead strings
        spec_flatness = librosa.feature.spectral_flatness(
            y=audio, hop_length=self.hop_length
        )[0]

        # Normalised audio so fingerpicked soft-attack notes contribute to onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio_norm, sr=self.sample_rate, hop_length=self.hop_length,
        )
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        # Bass string hits first — strongest pitch class at onset = chord root hint
        onset_frames_arr = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,   # snap to energy minimum before the peak
        )
        onset_set = set(int(f) for f in onset_frames_arr)

        from src.theory.intervals import NOTE_NAMES as _NOTE_NAMES

        if len(beat_frames) < 2:
            # fallback: single window over the whole track
            avg = chroma.mean(axis=1)
            norm = avg / (avg.max() + 1e-8)
            sorted_pcs = np.argsort(norm)[::-1]
            active_pcs = [int(p) for p in sorted_pcs if norm[p] > 0.6][:3]
            symbol = self.chord_builder.identify_chord_from_pitches(
                active_pcs, key_context=key, scale_pcs=scale_pcs
            ) if active_pcs else "N.C."
            conf = {_NOTE_NAMES[p]: float(norm[p]) for p in active_pcs}
            return [symbol] * total_frames, [(0, total_frames, conf)]

        if len(beat_frames) >= 2:
            avg_beat_dur = (
                (int(beat_frames[-1]) - int(beat_frames[0]))
                / max(len(beat_frames) - 1, 1)
                * self.hop_length / self.sample_rate
            )
            detected_bpm = 60.0 / avg_beat_dur if avg_beat_dur > 0 else 120.0
        else:
            detected_bpm = 120.0

        beat_dur_sec   = 60.0 / max(detected_bpm, 1.0)
        # Merge onsets within one beat — strum/arpeggio counts as one chord
        min_window_sec = max(0.25, beat_dur_sec * 1.0)

        onset_list = sorted(onset_set)
        if len(onset_list) >= 2:
            raw_onset_times = librosa.frames_to_time(
                onset_list, sr=self.sample_rate, hop_length=self.hop_length
            )
            merged_times: list[float] = [raw_onset_times[0]]
            for ot in raw_onset_times[1:]:
                if ot - merged_times[-1] >= min_window_sec:
                    merged_times.append(ot)
            merged_frames = librosa.time_to_frames(
                merged_times, sr=self.sample_rate, hop_length=self.hop_length
            )
            boundaries = np.unique(
                np.concatenate([[0], merged_frames, [total_frames]])
            ).astype(int)
        else:
            # fallback: use beat-aligned windows for quiet passages
            beats_per_window = 2 if detected_bpm < 80 else 1
            window_starts = beat_frames[::beats_per_window]
            boundaries = np.unique(
                np.concatenate([[0], window_starts, [total_frames]])
            ).astype(int)

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

            # 2x weight on second half — chord is settled there, reduces ringing from previous chord
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

            # root hint: dominant pitch at >15% onset energy is likely the bass root
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

            # muted strums are broadband noise (flatness > 0.15) — label as N.C.
            window_flatness = float(
                spec_flatness[start : min(end, len(spec_flatness))].mean()
            ) if end > start else 0.0

            if seg_max < 0.05 or window_flatness > 0.15:
                label = "N.C."
                conf: dict = {}
            else:
                l1 = seg_chroma.sum() + 1e-8
                norm_l1 = seg_chroma / l1
                sorted_pcs = np.argsort(norm_l1)[::-1]
                active_pcs = [int(p) for p in sorted_pcs if norm_l1[p] > 0.10][:4]
                label = self.chord_builder.identify_chord_from_pitches(
                    active_pcs, key_context=key, scale_pcs=scale_pcs,
                    onset_root_pc=onset_root_pc,
                ) if active_pcs else "N.C."

                # minor-maj7 almost never appears in rock — usually ringing open strings.
                # Retry with lower threshold. Also triggers for Em → B7 transitions.
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

                # distorted guitar smears 5th harmonics into complex labels — check onset for power chord
                _is_dim_or_complex = (
                    'dim' in label or '#11' in label or 'aug' in label
                )
                if _is_dim_or_complex and onset_root_pc is not None and window_onsets:
                    _of  = window_onsets[0]
                    _ow  = chroma[:, _of : min(_of + 8, chroma.shape[1])].mean(axis=1)
                    _tot = _ow.sum() + 1e-8
                    _5pc = (onset_root_pc + 7) % 12
                    _r_e = _ow[onset_root_pc] / _tot
                    _5_e = _ow[_5pc] / _tot
                    _m3  = _ow[(onset_root_pc + 3) % 12] / _tot
                    _M3  = _ow[(onset_root_pc + 4) % 12] / _tot
                    if _r_e > 0.13 and _5_e > 0.07 and _5_e > max(_m3, _M3):
                        label = f"{_NOTE_NAMES[onset_root_pc]}5"

                conf = {_NOTE_NAMES[p]: float(norm_l1[p]) for p in active_pcs}

            window_labels.append(label)
            window_spans.append((start, end))
            window_confidences.append(conf)

        # Two passes smooth out isolated chord flickers without flattening alternating progressions
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
        basic_pitch_midi=None,
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
        self.basic_pitch_midi = basic_pitch_midi

    def save_guitar_audio(self, output_path: str) -> str:
        """Save the extracted guitar audio to a WAV file."""
        import soundfile as sf

        if self.guitar_waveform is None:
            raise ValueError("No guitar waveform available")

        audio_np = self.guitar_waveform.squeeze().numpy()
        sf.write(output_path, audio_np, self.sample_rate)
        return output_path

    def export_midi(self, output_path: str) -> str:
        """Export MIDI from basic-pitch transcription (with quantised copy), or chord voicings as fallback."""
        from pathlib import Path as _Path
        from src.features.midi_exporter import MidiExporter

        if self.basic_pitch_midi is not None:
            # Strip high-register artefacts (overtones, reverb tails above C5)
            pm_clean = MidiExporter.filter_pitch(self.basic_pitch_midi)
            pm_clean.write(output_path)

            # Quantised copy — 16th-note grid so strum notes group on the same beat
            q_path = str(
                _Path(output_path).with_name(
                    _Path(output_path).stem + "_quantised.mid"
                )
            )
            pm_q = MidiExporter.quantise_midi(pm_clean, self.analysis.bpm)
            pm_q.write(q_path)
            print(f"  [MIDI] Quantised copy saved: {_Path(q_path).name}")
            return output_path

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
