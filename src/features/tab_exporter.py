"""
Full-song guitar tab exporters.

Two exporters are provided:

TabExporter (chord-voicing tab)
    Shows the chord voicing identified by the chord detector at each beat.
    Useful for verifying chord label accuracy — the voicing is looked up from
    a chord-voicing dictionary, so every 8th note in the bar shows the same
    fret numbers.  Good for checking whether the detected chord symbol is
    right, but does NOT show individual note movement within a bar.

PitchTabExporter (pitch-detection tab)
    Shows the actual pitches detected in the guitar stem audio at each 8th
    note position, mapped to string/fret assignments.  This is closer to a
    human-readable tablature of what the guitarist is actually playing —
    open strings, pull-offs, hammer-ons, arpeggiation, and riffs are all
    visible as changing fret numbers within the bar.

    Example (arpeggio over G chord):

        [Intro]
         1
        e|0--3--0--3--|
        B|0--0--0--1--|
        G|0--0--2--0--|
        D|2--0--0--0--|
        A|3--x--x--x--|
        E|x--x--x--x--|

Each column is one 8th note (beat_times subdivided by 2).
"""

from __future__ import annotations

import numpy as np


# ── String display order (high e at top, low E at bottom) ────────────────────
_STR_NAMES = ["e", "B", "G", "D", "A", "E"]   # display order: high→low
_STR_IDX   = [5, 4, 3, 2, 1, 0]               # voicing index: 5=high e, 0=low E


class TabExporter:
    """
    Generate full-song ASCII guitar tab from chord events and beat positions.

    Resolution: 8th notes (beat_times subdivided by 2). Each subdivision
    column shows the current chord voicing's fret numbers on all 6 strings.
    """

    BARS_PER_LINE  = 3    # bars to print per row of tab
    SUBDIVISIONS   = 2    # subdivisions per beat (2 = 8th notes)

    def __init__(self, tuning=None):
        from src.theory.chord_voicings import ChordVoicingEngine
        self._engine = ChordVoicingEngine(tuning=tuning)

    def export(
        self,
        chord_events: list,
        beat_times: np.ndarray,
        bpm: float,
        time_signature: tuple[int, int],
        sections: list | None = None,
        song_name: str = "",
    ) -> str:
        """
        Generate the full ASCII tab string.

        Args:
            chord_events:   ChordEvent list from PipelineResult
            beat_times:     numpy array of beat timestamps (seconds)
            bpm:            tempo in BPM
            time_signature: (numerator, denominator)
            sections:       SongSection list (optional — for section labels)
            song_name:      shown in the header (optional)

        Returns:
            Multi-line string suitable for writing to a .txt file.
        """
        beats_per_bar = time_signature[0]

        # ── Subdivide beat times to 8th-note resolution ───────────────────────
        sub_times = self._subdivide(beat_times, self.SUBDIVISIONS)

        # ── Assign the current chord to each subdivision ──────────────────────
        sub_chords = _assign_chords(chord_events, sub_times)

        sub_voicings = [self._resolve(sym) for sym in sub_chords]

        # ── Section label lookup: subdivision index → label ───────────────────
        section_labels = _build_section_map(sections, sub_times)

        # ── Format ────────────────────────────────────────────────────────────
        return _format_tab(
            sub_voicings,
            sub_chords,
            beats_per_bar * self.SUBDIVISIONS,   # subdivisions per bar
            bpm,
            time_signature,
            section_labels,
            self.BARS_PER_LINE,
            song_name,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _subdivide(beat_times: np.ndarray, n: int) -> np.ndarray:
        return _subdivide(beat_times, n)

    def _resolve(self, symbol: str) -> list[int] | None:
        """Return voicing (6 frets) or None for N.C./unknown chords."""
        if symbol in ("N.C.", ""):
            return None
        return self._engine.get_voicing(symbol)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _subdivide(beat_times: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate beat_times into n subdivisions per beat."""
    if len(beat_times) < 2:
        return beat_times
    times = []
    for i in range(len(beat_times) - 1):
        step = (beat_times[i + 1] - beat_times[i]) / n
        for j in range(n):
            times.append(beat_times[i] + j * step)
    times.append(float(beat_times[-1]))
    return np.array(times)


def _assign_chords(chord_events, times: np.ndarray) -> list[str]:
    """Return the chord symbol active at each timestamp."""
    # Small epsilon resolves floating-point comparison issues: chord event
    # start_time is computed via frame * (hop/sr) while beat_times come from
    # librosa as frame * hop / sr — different operation order, different ULP.
    _EPS = 1e-9
    result = []
    for t in times:
        label = "N.C."
        for ev in chord_events:
            if ev.start_time - _EPS <= t < ev.end_time + _EPS:
                label = ev.symbol
                break
        result.append(label)
    return result


def _build_section_map(sections, times: np.ndarray) -> dict[int, str]:
    """Map subdivision index → section label for the first subdivision in each section."""
    mapping: dict[int, str] = {}
    if not sections:
        return mapping
    for sec in sections:
        for i, t in enumerate(times):
            if t >= sec.start_time:
                if i not in mapping:
                    mapping[i] = sec.label
                break
    return mapping


def _fret_cell(fret: int, col_w: int) -> str:
    """Format a single fret value into a fixed-width cell."""
    if fret < 0:
        s = "x"
    else:
        s = str(fret)
    # pad with dashes to col_w
    return s + "-" * (col_w - len(s))


def _format_tab(
    sub_voicings: list,
    sub_chords: list[str],
    subs_per_bar: int,
    bpm: float,
    time_signature: tuple[int, int],
    section_labels: dict[int, str],
    bars_per_line: int,
    song_name: str,
) -> str:
    col_w   = 3   # characters per subdivision column  (e.g. "0--", "12-", "x--")
    sep     = "|"

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    if song_name:
        lines.append(song_name)
        lines.append("")
    lines.append(f"  Tempo: {bpm:.0f} bpm   Time: {time_signature[0]}/{time_signature[1]}")
    lines.append("")

    n_subs    = len(sub_voicings)
    sub_idx   = 0
    bar_idx   = 0

    while sub_idx < n_subs:
        # ── Check for section label at start of this row ──────────────────────
        row_start = sub_idx
        row_end   = min(sub_idx + bars_per_line * subs_per_bar, n_subs)

        for i in range(row_start, row_end):
            if i in section_labels:
                lines.append(f"  [{section_labels[i]}]")
                break

        # ── Bar number header ─────────────────────────────────────────────────
        bar_num_row = "   "
        subs_in_row = row_end - row_start
        for b in range((subs_in_row + subs_per_bar - 1) // subs_per_bar):
            num = str(bar_idx + b + 1)
            bar_num_row += num + " " * (subs_per_bar * col_w + 1 - len(num))
        lines.append(bar_num_row)

        # ── Six string rows ───────────────────────────────────────────────────
        str_rows = [""] * 6
        for disp_i, str_i in enumerate(_STR_IDX):
            row = _STR_NAMES[disp_i] + sep
            for j in range(row_start, row_end):
                voicing = sub_voicings[j]
                if voicing is None:
                    cell = "-" * col_w
                else:
                    cell = _fret_cell(voicing[str_i], col_w)
                row += cell
                # Bar separator every subs_per_bar subdivisions
                if (j - row_start + 1) % subs_per_bar == 0:
                    row += sep
            str_rows[disp_i] = row

        for row in str_rows:
            lines.append(row)

        # ── Chord name row — label at each position where chord changes ──────
        chord_row = "  "
        for b in range((subs_in_row + subs_per_bar - 1) // subs_per_bar):
            s = row_start + b * subs_per_bar
            bar_end = min(s + subs_per_bar, n_subs)
            bar_width = subs_per_bar * col_w + 1

            changes: list[tuple[int, str]] = []
            prev = None
            for k in range(s, bar_end):
                lbl = sub_chords[k] if sub_chords[k] not in ("N.C.", "") else ""
                if lbl != prev:
                    changes.append((k - s, lbl))
                    prev = lbl

            bar_chars = [" "] * bar_width
            for sub_off, lbl in changes:
                if not lbl:
                    continue
                char_pos = sub_off * col_w
                for li, ch in enumerate(lbl):
                    if char_pos + li < bar_width:
                        bar_chars[char_pos + li] = ch

            chord_row += "".join(bar_chars)
        lines.append(chord_row)
        lines.append("")

        bar_idx += (row_end - row_start + subs_per_bar - 1) // subs_per_bar
        sub_idx  = row_end

    return "\n".join(lines)


# ── PitchTabExporter ─────────────────────────────────────────────────────────

class PitchTabExporter:
    """
    Generate full-song ASCII guitar tab by detecting actual pitches from the
    guitar stem audio, not from chord-voicing lookup.

    Two-stage processing:

    Stage 1 — Rhythm guitar isolation via spectral median filter:
        Many recordings have two guitar tracks (rhythm + lead) merged into one
        Demucs guitar stem.  A time-axis median filter on the STFT magnitude
        separates the "sustained background" (strummed chord guitar, which
        holds notes for 1+ beats) from the "transient foreground" (lead riffs
        and melodic fills, which are shorter than the filter window).  This is
        equivalent to the REPET (REpeating-Pattern Extraction Technique) family
        of methods — the rhythm guitar IS the repeating pattern.

        Filter window: 1.5 beats at the detected BPM.  A note must be present
        for at least half the window (≈0.75 beats) to survive.  Chords held
        for a full bar easily survive; a 16th-note lead fill does not.

    Stage 2 — CQT pitch-to-fret mapping with per-string fret ceilings:
        The Constant-Q Transform of the isolated rhythm audio is computed.  At
        each 8th-note subdivision the loudest active MIDI pitches are mapped to
        guitar strings via a lowest-fret heuristic.

        Per-string fret ceilings prevent bass-bleed notes from appearing on the
        low strings at implausibly high fret positions:
            Low E: max fret 10  (note D3)  — bass guitar bleeds into this range
            A:     max fret 12  (note E3)
            D:     max fret 15  (note A3)
            G, B, e: unrestricted up to fret 22

        Without ceilings, a bass note like D3 would appear as fret 11 on the
        low E string — correct mathematically but wrong for guitar tab context.
        With the ceiling, D3 is instead assigned to the A string at fret 7 or
        the D string at fret 2, both of which are plausible guitar positions.
    """

    BARS_PER_LINE = 3
    SUBDIVISIONS  = 2      # 8th-note resolution
    MAX_PITCHES   = 6      # at most one note per string

    # Per-string maximum fret (index 0 = low E, 5 = high e).
    # Notes requiring a higher fret on a low string are re-routed to a
    # higher string rather than appearing as implausible high-fret positions.
    _FRET_CEILING = [3, 7, 10, 14, 19, 22]

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        tuning=None,
        tab_cnn_checkpoint: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.hop_length  = hop_length
        # Open-string MIDI values: index 0 = low E, index 5 = high e.
        # Tuning.midi_notes is ordered low→high string (same index convention).
        # Without this, fret assignments are wrong for non-standard tunings:
        # e.g. Eb standard open E = MIDI 39, not 40 — every fret off by 1.
        if tuning is not None and hasattr(tuning, 'midi_notes'):
            self._open_midi: list[int] = list(tuning.midi_notes)
        else:
            self._open_midi = [40, 45, 50, 55, 59, 64]   # standard tuning

        # Optional TabCNN model for learned fret detection
        self._tab_cnn = None
        if tab_cnn_checkpoint:
            self._load_tab_cnn(tab_cnn_checkpoint)

    def _load_tab_cnn(self, checkpoint_path: str) -> None:
        import torch
        from guitar_only.models.tab_cnn import TabCNN
        model = TabCNN(n_bins=192, context_frames=9, n_frets=20)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self._tab_cnn = model
        print(f"[TabCNN] loaded checkpoint: {checkpoint_path}")

    def export(
        self,
        waveform,
        beat_times: np.ndarray,
        bpm: float,
        time_signature: tuple[int, int],
        sections: list | None = None,
        song_name: str = "",
        chord_events: list | None = None,
    ) -> str:
        """
        Generate the pitch-detection tab string.

        Args:
            waveform:        Guitar stem as torch.Tensor (1, N) or numpy array.
            beat_times:      Beat timestamps in seconds.
            bpm:             Detected tempo (used to size the median filter window).
            time_signature:  (numerator, denominator).
            sections:        SongSection list for section labels (optional).
            song_name:       Shown in header (optional).
            chord_events:    ChordEvent list — when supplied, the chord symbol
                             detected by the pipeline is printed below each bar,
                             showing what the chord detector recognised from the
                             same audio that produced the note display above.

        Returns:
            Multi-line string ready to write to a .txt file.
        """
        import librosa

        audio: np.ndarray
        if hasattr(waveform, 'squeeze'):
            audio = waveform.squeeze().numpy()
        else:
            audio = np.asarray(waveform, dtype=np.float32)

        # The guitar stem is already Demucs-separated, so the median filter
        # used for rhythm/lead isolation in full-mix scenarios is not needed
        # here and actively suppresses quiet fingerpicked notes (their short
        # per-string duration falls below the filter's survival threshold).
        sub_times = _subdivide(beat_times, self.SUBDIVISIONS)

        # ── Stage 2: CQT over the guitar pitch range ──────────────────────────
        # fmin = lowest open string of the detected tuning (e.g. E2 = 82.4 Hz)
        # 60 bins = 5 octaves covers E2→E7, well past the playable guitar range
        fmin = librosa.midi_to_hz(self._open_midi[0])

        if self._tab_cnn is not None:
            sub_frets = self._tab_cnn_frets(audio, sub_times, fmin)
        else:
            # CQT with beat-window averaging.
            # Rather than sampling the CQT at each 8th-note frame independently
            # (which hits noisy transient frames), average the CQT magnitude over
            # each full beat window and hold that stable vector for both 8th-notes
            # within the beat.  Averaging ~48 frames at 108 BPM / 512-hop
            # suppresses:
            #   • single-frame onset transients (attack noise at note start)
            #   • inter-string harmonic bleed that varies frame-to-frame
            #   • decay-shape artefacts (partials damp at different rates)
            # This produces a stable per-beat fret pattern that matches the
            # sustained chord shapes shown in guitar tablature.
            C = np.abs(librosa.cqt(
                audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=fmin,
                n_bins=60,
                bins_per_octave=12,
            ))
            n_total = C.shape[1]

            # One averaged CQT vector per beat → one fret array per beat
            beat_frets: list[list[int]] = []
            for bi in range(len(beat_times) - 1):
                s = int(librosa.time_to_frames(
                    float(beat_times[bi]),
                    sr=self.sample_rate, hop_length=self.hop_length,
                ))
                e = int(librosa.time_to_frames(
                    float(beat_times[bi + 1]),
                    sr=self.sample_rate, hop_length=self.hop_length,
                ))
                s = max(0, min(s, n_total - 1))
                e = max(s + 1, min(e, n_total))
                seg = C[:, s:e]
                n_frames = e - s
                if n_frames >= 4:
                    weights = np.ones(n_frames)
                    weights[n_frames // 2:] = 2.0
                    avg_col = (seg * weights).sum(axis=1) / weights.sum()
                else:
                    avg_col = seg.mean(axis=1)
                beat_frets.append(self._frame_to_frets(avg_col))

            # Map each 8th-note subdivision back to its beat window
            sub_frets = []
            for t in sub_times:
                bi = max(0, int(np.searchsorted(beat_times, t, side="right")) - 1)
                bi = min(bi, len(beat_frets) - 1)
                sub_frets.append(list(beat_frets[bi]))

        section_labels = _build_section_map(sections, sub_times)
        beats_per_bar  = time_signature[0]
        sub_chords     = _assign_chords(chord_events, sub_times) if chord_events else None

        return _format_pitch_tab(
            sub_frets,
            beats_per_bar * self.SUBDIVISIONS,
            bpm,
            time_signature,
            section_labels,
            self.BARS_PER_LINE,
            song_name,
            sub_chords=sub_chords,
        )

    def _tab_cnn_frets(
        self,
        audio: np.ndarray,
        sub_times: np.ndarray,
        fmin: float,
    ) -> list[list[int]]:
        """Run TabCNN inference to get per-string fret predictions."""
        import torch
        import librosa

        N_BINS        = 192
        CONTEXT       = 9
        BINS_PER_OCT  = 24
        half          = CONTEXT // 2

        # Resample to 44100 if needed (TabCNN trained at 44100)
        if self.sample_rate != 44100:
            audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=44100)
            sr = 44100
        else:
            sr = self.sample_rate

        # Clamp n_bins so the highest CQT bin stays below the Nyquist frequency.
        # When a capo raises fmin, the same 192 bins would reach above Nyquist.
        import math
        n_bins_safe   = int(BINS_PER_OCT * math.log2((sr / 2.0) / fmin))
        n_bins_compute = min(N_BINS, n_bins_safe)

        C = np.abs(librosa.cqt(
            audio, sr=sr, hop_length=self.hop_length,
            fmin=fmin, n_bins=n_bins_compute, bins_per_octave=BINS_PER_OCT,
        ))

        # Zero-pad back to N_BINS so the TabCNN model always sees 192 bins.
        # High-frequency bins that exceed Nyquist are correctly represented as zero.
        if n_bins_compute < N_BINS:
            pad = np.zeros((N_BINS - n_bins_compute, C.shape[1]), dtype=C.dtype)
            C = np.vstack([C, pad])

        C = librosa.amplitude_to_db(C, ref=np.max)
        C = (C - C.min()) / (C.max() - C.min() + 1e-8)
        C = C.astype(np.float32)

        C_padded = np.pad(C, ((0, 0), (half, half)), mode="edge")
        n_frames = C.shape[1]

        # Build batch of context windows for all subdivisions at once
        frames_idx = [
            min(
                int(librosa.time_to_frames(t, sr=sr, hop_length=self.hop_length)),
                n_frames - 1,
            )
            for t in sub_times
        ]
        batch = np.stack([
            C_padded[:, f: f + CONTEXT] for f in frames_idx
        ])                                              # (N, n_bins, ctx)
        x = torch.from_numpy(batch).unsqueeze(1)       # (N, 1, n_bins, ctx)

        with torch.no_grad():
            preds = self._tab_cnn.predict_frets(x)     # (N, 6)  class indices

        # Convert class indices back to fret numbers
        # class 0 → not played (-1);  class k → fret k-1
        sub_frets: list[list[int]] = []
        for row in preds.tolist():
            frets = [(c - 1) if c > 0 else -1 for c in row]
            sub_frets.append(frets)
        return sub_frets

    def _isolate_rhythm_guitar(self, audio: np.ndarray, bpm: float) -> np.ndarray:
        """
        Suppress lead guitar and transient artefacts with a time-axis median filter.

        The filter window is 1.5 beats long.  Any spectral component that is
        NOT sustained for at least ~0.75 beats (half the window) is attenuated.
        Strummed chords held for a full bar (4 beats) easily survive; a quarter-
        note lead fill (~0.5 beats at 120 BPM) or a bass transient does not.

        The filtered magnitude is recombined with the original STFT phase and
        inverted back to audio.  STFT hop = hop_length to match the rest of the
        pipeline.
        """
        import librosa
        from scipy.ndimage import median_filter

        # STFT
        D     = librosa.stft(audio, hop_length=self.hop_length)
        S     = np.abs(D)
        phase = np.angle(D)

        # Window size in frames: 0.75 beats at detected BPM.
        # Survival threshold = window/2 = 0.375 beats, so 8th-note arpeggios
        # (0.5 beats) survive while 16th-note lead fills (0.25 beats) are
        # suppressed.  The previous 1.5-beat window had a 0.75-beat survival
        # threshold which silenced fingerpicked intros like "Black" (Pearl Jam)
        # where each 8th note only rings for ~26 frames < 38-frame threshold.
        beat_seconds   = 60.0 / max(bpm, 40.0)
        window_seconds = 0.75 * beat_seconds
        window_frames  = int(window_seconds * self.sample_rate / self.hop_length)
        window_frames  = max(window_frames, 11)           # minimum 11 frames
        if window_frames % 2 == 0:
            window_frames += 1                            # median filter needs odd

        # Median filter along the time axis (axis=1 in [freq × time])
        S_smooth = median_filter(S, size=(1, window_frames))

        # Reconstruct audio from smoothed magnitude + original phase
        D_smooth = S_smooth * np.exp(1j * phase)
        out = librosa.istft(D_smooth, hop_length=self.hop_length, length=len(audio))
        return out.astype(np.float32)

    def _frame_to_frets(self, cqt_col: np.ndarray) -> list[int]:
        """
        Map a single CQT magnitude column to per-string fret assignments.

        Per-string fret ceilings (see _FRET_CEILING) prevent bass-bleed notes
        from appearing at implausibly high positions on the low strings.

        Returns:
            List[int] of length 6.  Index 0 = low E string, 5 = high e string.
            -1 means the string is not active in this frame.
        """
        frets = [-1] * 6

        col_max = float(cqt_col.max())
        if col_max < 1e-8:
            return frets   # silence

        threshold = col_max * 0.15

        candidates: list[tuple[int, float]] = []
        for i, energy in enumerate(cqt_col):
            if energy >= threshold:
                midi = self._open_midi[0] + i
                candidates.append((midi, float(energy)))

        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:self.MAX_PITCHES]

        # Assign each pitch to a string.
        # Heuristic: lowest fret wins, subject to _FRET_CEILING[string].
        # This routes bass-range notes away from the low E string to higher
        # strings where they appear at more plausible fret numbers.
        assigned: set[int] = set()
        for midi, _ in candidates:
            best_cost = None
            best_fret = None
            best_str  = None
            for s_idx, open_m in enumerate(self._open_midi):
                if s_idx in assigned:
                    continue
                fret = midi - open_m
                ceiling = self._FRET_CEILING[s_idx]
                if 0 <= fret <= ceiling:
                    # Cost = fret number; prefer lowest fret across valid strings
                    if best_cost is None or fret < best_cost:
                        best_cost = fret
                        best_fret = fret
                        best_str  = s_idx
            if best_str is not None:
                frets[best_str] = best_fret
                assigned.add(best_str)

        # ── Fret-span constraint ──────────────────────────────────────────────
        # A human hand spans at most 3 frets in one chord position (index
        # finger to pinky = frets N to N+3).  Phantom notes from bass
        # harmonics always land higher than the played chord, so iteratively
        # drop the highest fret until the remaining span is ≤ 3.
        fretted_vals = [f for f in frets if f > 0]
        if len(fretted_vals) >= 2:
            working = sorted(fretted_vals)
            while len(working) >= 2 and working[-1] - working[0] > 3:
                working.pop()   # drop highest phantom
            valid_frets = set(working)
            frets = [f if (f <= 0 or f in valid_frets) else -1 for f in frets]

        return frets

    def detect_chords(
        self,
        waveform,
        beat_times: np.ndarray,
        bpm: float,
        time_signature: tuple[int, int],
        key: str | None = None,
        scale_pcs: set[int] | None = None,
    ) -> list[str]:
        """
        Derive chord labels directly from fret positions.

        Runs the same fret detection pipeline as export() but instead of
        rendering ASCII tab, converts fret + tuning → pitch class → chord
        symbol.  The lowest active string on each beat is used as the bass
        root hint, which resolves ambiguities like G/B vs G or Em vs G.

        Returns one chord label per CQT frame (same length convention as the
        chroma-based chord detection) so the result can be merged directly
        into pipeline frame_labels.
        """
        import librosa
        from src.theory.chord_builder import ChordBuilder

        if hasattr(waveform, 'squeeze'):
            audio = waveform.squeeze().numpy()
        else:
            audio = np.asarray(waveform, dtype=np.float32)

        total_frames = int(np.ceil(len(audio) / self.hop_length))
        fmin = librosa.midi_to_hz(self._open_midi[0])
        chord_builder = ChordBuilder()

        # ── Get per-beat fret arrays ──────────────────────────────────────────
        if self._tab_cnn is not None:
            sub_times = _subdivide(beat_times, self.SUBDIVISIONS)
            sub_frets_all = self._tab_cnn_frets(audio, sub_times, fmin)
            # Per-string mode-fret aggregation across subdivisions.
            #
            # Taking the UNION of all PCs across subdivisions gives 5–6 unique
            # pitch classes per beat (fingerpicking touches different strings on
            # each 8th note), which causes the chord builder to label simple
            # triads as exotic extended chords.
            #
            # Instead, compute the MODE fret for each string across the beat's
            # subdivisions.  Strings played consistently produce a stable fret;
            # strings touched only once produce a single observation that ties
            # with "not played" and gets the lower (more open) fret.  The resulting
            # 3–4 unique pitch classes represent the actual chord shape cleanly.
            beat_pcs_list: list[tuple[list[int], int | None]] = []
            for bi in range(max(len(beat_times) - 1, 1)):
                # Collect observed frets per string across all subdivisions
                string_obs: list[list[int]] = [[] for _ in range(6)]
                for si in range(self.SUBDIVISIONS):
                    idx = bi * self.SUBDIVISIONS + si
                    if idx >= len(sub_frets_all):
                        break
                    for s_idx, fret in enumerate(sub_frets_all[idx]):
                        if fret >= 0:
                            string_obs[s_idx].append(fret)

                pcs: list[int] = []
                for s_idx in range(6):
                    obs = string_obs[s_idx]
                    if not obs:
                        continue
                    # Mode fret; ties broken by taking the lower fret (open preference)
                    mode_fret = min(obs, key=lambda f: (-obs.count(f), f))
                    pc = (self._open_midi[s_idx] + mode_fret) % 12
                    if pc not in pcs:
                        pcs.append(pc)
                beat_pcs_list.append((pcs, None))
        else:
            C = np.abs(librosa.cqt(
                audio, sr=self.sample_rate, hop_length=self.hop_length,
                fmin=fmin, n_bins=60, bins_per_octave=12,
            ))
            n_total = C.shape[1]
            beat_pcs_list = []
            for bi in range(max(len(beat_times) - 1, 1)):
                s = int(librosa.time_to_frames(
                    float(beat_times[bi]), sr=self.sample_rate, hop_length=self.hop_length,
                ))
                e = int(librosa.time_to_frames(
                    float(beat_times[min(bi + 1, len(beat_times) - 1)]),
                    sr=self.sample_rate, hop_length=self.hop_length,
                ))
                s = max(0, min(s, n_total - 1))
                e = max(s + 1, min(e, n_total))
                seg = C[:, s:e]
                n_frames = e - s
                if n_frames >= 4:
                    weights = np.ones(n_frames)
                    weights[n_frames // 2:] = 2.0
                    avg_col = (seg * weights).sum(axis=1) / weights.sum()
                else:
                    avg_col = seg.mean(axis=1)
                frets = self._frame_to_frets(avg_col)
                pcs: list[int] = []
                bass_root: int | None = None
                for s_idx in range(6):
                    fret = frets[s_idx]
                    if fret >= 0:
                        pc = (self._open_midi[s_idx] + fret) % 12
                        if pc not in pcs:
                            pcs.append(pc)
                        if bass_root is None:
                            bass_root = pc
                beat_pcs_list.append((pcs, bass_root))

        # ── Identify chord per beat ───────────────────────────────────────────
        beat_labels: list[str] = []
        for pcs, bass_root in beat_pcs_list:
            if not pcs:
                beat_labels.append("N.C.")
            else:
                # Do NOT pass onset_root_pc here.  When the lowest string is
                # muted by TabCNN the "bass root" becomes the second string
                # (e.g. Bb instead of Eb), which biases the chord builder
                # toward the wrong root.  Template matching + scale context
                # is sufficient: Ebm is the only perfect 3-note template fit
                # for {Eb, Gb, Bb}, so it wins without a root hint.
                label = chord_builder.identify_chord_from_pitches(
                    pcs,
                    key_context=key,
                    scale_pcs=scale_pcs,
                )
                beat_labels.append(label or "N.C.")

        # ── Expand beat labels to per-frame labels ────────────────────────────
        frame_labels: list[str] = []
        # Prepend N.C. for frames before the first beat so that list indices
        # match audio frame numbers when merged with chroma frame_labels.
        if len(beat_times) > 0:
            first_beat_frame = int(librosa.time_to_frames(
                float(beat_times[0]), sr=self.sample_rate, hop_length=self.hop_length
            ))
            frame_labels.extend(["N.C."] * first_beat_frame)
        for bi, label in enumerate(beat_labels):
            t_start = float(beat_times[bi])
            t_end   = float(beat_times[bi + 1]) if bi + 1 < len(beat_times) else t_start + 60.0 / max(bpm, 1)
            fs = int(librosa.time_to_frames(t_start, sr=self.sample_rate, hop_length=self.hop_length))
            fe = int(librosa.time_to_frames(t_end,   sr=self.sample_rate, hop_length=self.hop_length))
            fe = min(fe, total_frames)
            frame_labels.extend([label] * max(0, fe - fs))

        while len(frame_labels) < total_frames:
            frame_labels.append(frame_labels[-1] if frame_labels else "N.C.")
        return frame_labels[:total_frames]


def _format_pitch_tab(
    sub_frets: list[list[int]],
    subs_per_bar: int,
    bpm: float,
    time_signature: tuple[int, int],
    section_labels: dict[int, str],
    bars_per_line: int,
    song_name: str,
    sub_chords: list[str] | None = None,
) -> str:
    """Format per-subdivision fret arrays as ASCII guitar tab.

    When sub_chords is provided, a chord-label row is printed below each
    bar showing what the chord detector recognised — so the raw note
    display (top) and the inferred chord (bottom) can be compared directly.
    """
    col_w = 3
    sep   = "|"

    lines: list[str] = []

    if song_name:
        lines.append(song_name)
        lines.append("")
    lines.append(f"  Tempo: {bpm:.0f} bpm   Time: {time_signature[0]}/{time_signature[1]}")
    lines.append("")

    n_subs  = len(sub_frets)
    sub_idx = 0
    bar_idx = 0

    while sub_idx < n_subs:
        row_start = sub_idx
        row_end   = min(sub_idx + bars_per_line * subs_per_bar, n_subs)

        # Section label
        for i in range(row_start, row_end):
            if i in section_labels:
                lines.append(f"  [{section_labels[i]}]")
                break

        # Bar number header
        bar_num_row = "   "
        subs_in_row = row_end - row_start
        for b in range((subs_in_row + subs_per_bar - 1) // subs_per_bar):
            num = str(bar_idx + b + 1)
            bar_num_row += num + " " * (subs_per_bar * col_w + 1 - len(num))
        lines.append(bar_num_row)

        # Six string rows (high e at top, low E at bottom)
        str_rows = [""] * 6
        for disp_i, str_i in enumerate(_STR_IDX):
            row = _STR_NAMES[disp_i] + sep
            for j in range(row_start, row_end):
                fret = sub_frets[j][str_i]
                if fret < 0:
                    cell = "-" * col_w          # string not active
                else:
                    cell = _fret_cell(fret, col_w)
                row += cell
                if (j - row_start + 1) % subs_per_bar == 0:
                    row += sep
            str_rows[disp_i] = row

        for row in str_rows:
            lines.append(row)

        # Chord label row — label at each position where chord changes within the bar
        if sub_chords is not None:
            chord_row = "  "
            for b in range((subs_in_row + subs_per_bar - 1) // subs_per_bar):
                s = row_start + b * subs_per_bar
                bar_end = min(s + subs_per_bar, n_subs)
                bar_width = subs_per_bar * col_w + 1

                # Find chord-change points (sub offset, label) within this bar
                changes: list[tuple[int, str]] = []
                prev = None
                for k in range(s, bar_end):
                    lbl = sub_chords[k] if sub_chords[k] not in ("N.C.", "") else ""
                    if lbl != prev:
                        changes.append((k - s, lbl))
                        prev = lbl

                # Place each label at its subdivision-aligned char position
                bar_chars = [" "] * bar_width
                for sub_off, lbl in changes:
                    if not lbl:
                        continue
                    char_pos = sub_off * col_w
                    for li, ch in enumerate(lbl):
                        if char_pos + li < bar_width:
                            bar_chars[char_pos + li] = ch

                chord_row += "".join(bar_chars)
            lines.append(chord_row)

        lines.append("")

        bar_idx += (row_end - row_start + subs_per_bar - 1) // subs_per_bar
        sub_idx  = row_end

    return "\n".join(lines)
