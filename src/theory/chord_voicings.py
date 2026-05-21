"""
Guitar chord voicing database and ASCII diagram renderer.

Two-tier approach:
  1. Lookup table for ~60 common open/barre shapes — exact fingerings.
  2. Algorithmic voicing generator for anything not in the table — searches
     the fretboard for a playable voicing given the chord's pitch classes.

ASCII diagram format (one chord = 8 lines wide, 6 lines tall):

    F#m7
    ┌─┬─┬─┬─┬─┐
 2  │▬│▬│▬│▬│▬│  <- barre
    ├─┼─┼─┼─┼─┤
 4  │●│●│ │ │ │
    ├─┼─┼─┼─┼─┤
    │ │ │ │ │ │
    └─┴─┴─┴─┴─┘
    E A D G B e
"""

from __future__ import annotations


# ── Standard tuning: pitch class per open string (low→high) ─────────────────
# E2  A2  D3  G3  B3  E4  (used as fallback when no tuning is provided)
_OPEN_PC = [4, 9, 2, 7, 11, 4]   # pitch classes 0-11

# ── Lookup table ─────────────────────────────────────────────────────────────
# Each entry: list of 6 fret numbers, low E → high e.
# -1 = muted (x), 0 = open string.
_VOICING_TABLE: dict[str, list[int]] = {
    # ── Open major ────────────────────────────────────────────────
    "C":      [-1, 3, 2, 0, 1, 0],
    "D":      [-1, -1, 0, 2, 3, 2],
    "E":      [0, 2, 2, 1, 0, 0],
    "F":      [1, 1, 2, 3, 3, 1],
    "G":      [3, 2, 0, 0, 0, 3],
    "A":      [-1, 0, 2, 2, 2, 0],
    "B":      [-1, 2, 4, 4, 4, 2],
    # ── Open minor ────────────────────────────────────────────────
    "Cm":     [-1, 3, 5, 5, 4, 3],
    "Dm":     [-1, -1, 0, 2, 3, 1],
    "Em":     [0, 2, 2, 0, 0, 0],
    "Fm":     [1, 1, 3, 3, 2, 1],   # barre 1
    "Gm":     [3, 5, 5, 3, 3, 3],
    "Am":     [-1, 0, 2, 2, 1, 0],
    "Bm":     [-1, 2, 4, 4, 3, 2],
    # ── Dominant 7ths ─────────────────────────────────────────────
    "C7":     [-1, 3, 2, 3, 1, 0],
    "D7":     [-1, -1, 0, 2, 1, 2],
    "E7":     [0, 2, 0, 1, 0, 0],
    "F7":     [1, 1, 2, 1, 3, 1],
    "G7":     [3, 2, 0, 0, 0, 1],
    "A7":     [-1, 0, 2, 0, 2, 0],
    "B7":     [-1, 2, 1, 2, 0, 2],
    # ── Major 7ths ────────────────────────────────────────────────
    "Cmaj7":  [-1, 3, 2, 0, 0, 0],
    "Dmaj7":  [-1, -1, 0, 2, 2, 2],
    "Emaj7":  [0, 2, 1, 1, 0, 0],
    "Fmaj7":  [-1, -1, 3, 2, 1, 0],
    "Gmaj7":  [3, 2, 0, 0, 0, 2],
    "Amaj7":  [-1, 0, 2, 1, 2, 0],
    # ── Minor 7ths ────────────────────────────────────────────────
    "Cm7":    [-1, 3, 5, 3, 4, 3],
    "Dm7":    [-1, -1, 0, 2, 1, 1],
    "Em7":    [0, 2, 2, 0, 3, 0],
    "Fm7":    [1, 1, 3, 1, 2, 1],
    "Gm7":    [3, 5, 5, 3, 3, 3],
    "Am7":    [-1, 0, 2, 0, 1, 0],
    "Bm7":    [-1, 2, 4, 2, 3, 2],
    "F#m7":   [2, 4, 4, 2, 2, 2],
    # ── Sus chords ────────────────────────────────────────────────
    "Dsus2":  [-1, -1, 0, 2, 3, 0],
    "Dsus4":  [-1, -1, 0, 2, 3, 3],
    "Esus4":  [0, 2, 2, 2, 0, 0],
    "Asus2":  [-1, 0, 2, 2, 0, 0],
    "Asus4":  [-1, 0, 2, 2, 3, 0],
    "Gsus4":  [3, 3, 0, 0, 1, 3],
    "Csus2":  [-1, 3, 5, 5, 3, 3],
    "Csus4":  [-1, 3, 3, 0, 1, 1],
    "Bsus2":  [-1, 2, 4, 4, 2, 2],
    "Bsus4":  [-1, 2, 4, 4, 4, 2],
    "F#sus2": [2, 4, 4, 4, 2, 2],
    "F#sus4": [2, 4, 4, 4, 5, 2],
    # ── Barre major (A-shape: root on A string) ───────────────────
    "Bb":     [-1, 1, 3, 3, 3, 1],
    "A#":     [-1, 1, 3, 3, 3, 1],
    "Db":     [-1, 4, 6, 6, 6, 4],
    "C#":     [-1, 4, 6, 6, 6, 4],
    "Eb":     [-1, 6, 8, 8, 8, 6],
    "D#":     [-1, 6, 8, 8, 8, 6],
    # ── Barre major (E-shape: root on low E string) ───────────────
    "F#":     [2, 4, 4, 3, 2, 2],
    "Gb":     [2, 4, 4, 3, 2, 2],
    "G#":     [4, 6, 6, 5, 4, 4],
    "Ab":     [4, 6, 6, 5, 4, 4],
    # ── Barre minor (A-shape) ─────────────────────────────────────
    "Bbm":    [-1, 1, 3, 3, 2, 1],
    "A#m":    [-1, 1, 3, 3, 2, 1],
    "C#m":    [-1, 4, 6, 6, 5, 4],
    "Dbm":    [-1, 4, 6, 6, 5, 4],
    "D#m":    [-1, 6, 8, 8, 7, 6],
    "Ebm":    [-1, 6, 8, 8, 7, 6],
    # ── Barre minor (E-shape) ─────────────────────────────────────
    "F#m":    [2, 4, 4, 2, 2, 2],
    "Gbm":    [2, 4, 4, 2, 2, 2],
    "G#m":    [4, 6, 6, 4, 4, 4],
    "Abm":    [4, 6, 6, 4, 4, 4],
    # ── 7th barre shapes ──────────────────────────────────────────
    "Bb7":    [-1, 1, 3, 1, 3, 1],
    "A#7":    [-1, 1, 3, 1, 3, 1],
    "F#7":    [2, 4, 2, 3, 2, 2],
    "Gb7":    [2, 4, 2, 3, 2, 2],
    "C#7":    [-1, 4, 6, 4, 6, 4],
    "Db7":    [-1, 4, 6, 4, 6, 4],
    "Bbm7":   [-1, 1, 3, 1, 2, 1],
    "A#m7":   [-1, 1, 3, 1, 2, 1],
    "F#m7":   [2, 4, 4, 2, 2, 2],
    "C#m7":   [-1, 4, 6, 4, 5, 4],
    "A7sus4": [-1, 0, 2, 0, 3, 0],
    "D7sus4": [-1, -1, 0, 2, 1, 3],
    "E7sus4": [0, 2, 2, 2, 3, 0],
    "G7sus4": [3, 3, 0, 0, 1, 1],
    # ── Add9 ──────────────────────────────────────────────────────
    "Cadd9":  [-1, 3, 2, 0, 3, 0],
    "Dadd9":  [-1, -1, 0, 2, 3, 0],
    "Gadd9":  [3, 2, 0, 2, 0, 3],
    "Aadd9":  [-1, 0, 2, 4, 2, 0],
    "Eadd9":  [0, 2, 2, 1, 0, 2],
    # ── Diminished / augmented ────────────────────────────────────
    "Ddim":   [-1, -1, 0, 1, 0, 1],
    "Edim":   [0, 1, 2, 0, -1, -1],
    "Adim":   [-1, 0, 1, 2, 1, -1],
    "Bdim":   [-1, 2, 3, 4, 3, -1],
    "Caug":   [-1, -1, 2, 1, 1, 0],
    "Eaug":   [0, 3, 2, 1, 1, 0],
    "Aaug":   [-1, 0, 3, 2, 2, 1],
}

# ── Chord-root transposition helpers ─────────────────────────────────────────
_PC_FROM_ROOT: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
_ROOT_FROM_PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_PC = {1: "Db", 3: "Eb", 6: "Gb", 8: "Ab", 10: "Bb"}


def _transpose_chord_symbol(symbol: str, semitones: int) -> str | None:
    """
    Return `symbol` with its root shifted by `semitones` half-steps, choosing
    whichever spelling (sharp or flat) exists in _VOICING_TABLE.
    Returns None when the root cannot be parsed.
    """
    if not symbol or symbol in ("N.C.", ""):
        return symbol
    if len(symbol) >= 2 and symbol[1] in ("#", "b"):
        root, quality = symbol[:2], symbol[2:]
    else:
        root, quality = symbol[:1], symbol[1:]
    if root not in _PC_FROM_ROOT:
        return None
    pc = (_PC_FROM_ROOT[root] + semitones) % 12
    for candidate in (_FLAT_PC.get(pc), _ROOT_FROM_PC[pc]):
        if candidate and (candidate + quality) in _VOICING_TABLE:
            return candidate + quality
    return _ROOT_FROM_PC[pc] + quality


class ChordVoicingEngine:
    """
    Return a fretboard voicing for any chord symbol.

    Checks the lookup table first (standard tuning only); falls back to the
    algorithmic generator for chords not in the table, or when a non-standard
    tuning is supplied.
    """

    def __init__(
        self,
        max_fret_span: int = 4,
        max_fret_pos:  int = 12,
        tuning=None,        # src.theory.tuning.Tuning or None (→ standard)
    ):
        self.max_fret_span = max_fret_span
        self.max_fret_pos  = max_fret_pos

        if tuning is not None:
            self._open_pc    = list(tuning.pitch_classes)
            self._str_names  = list(tuning.string_names)
            self._is_standard = (self._open_pc == _OPEN_PC)
            # Compute uniform semitone offset from standard (e.g. -1 for Eb standard).
            # Only used when all strings are shifted by the same amount.
            raw_diffs = [(self._open_pc[i] - _OPEN_PC[i]) % 12 for i in range(6)]
            signed = [(d - 12 if d > 6 else d) for d in raw_diffs]
            self._tuning_offset = signed[0] if len(set(signed)) == 1 else 0
        else:
            self._open_pc    = list(_OPEN_PC)
            self._str_names  = ["E", "A", "D", "G", "B", "e"]
            self._is_standard = True
            self._tuning_offset = 0

    def get_voicing(self, chord_symbol: str) -> list[int] | None:
        """
        Return a list of 6 fret numbers (low E → high string), or None if no
        voicing can be found.  -1 means muted (x), 0 means open string.
        """
        # 1. For uniformly-transposed tunings (e.g. Eb standard = -1), look up
        #    the physically-played shape by transposing the root back to standard.
        #    "Gb" in Eb standard → look up "G" shape (320003), not "Gb" barre.
        if self._tuning_offset != 0:
            std_sym = _transpose_chord_symbol(chord_symbol, -self._tuning_offset)
            if std_sym and std_sym in _VOICING_TABLE:
                return _VOICING_TABLE[std_sym]

        # 2. Direct lookup — for standard tuning or if the symbol is in the table
        if chord_symbol in _VOICING_TABLE:
            return _VOICING_TABLE[chord_symbol]

        # 3. Algorithmic — works for any tuning
        pitch_classes = self._chord_to_pitch_classes(chord_symbol)
        if not pitch_classes:
            return None

        voicing = self._find_voicing(pitch_classes, chord_symbol)

        # Reject all-open voicings from the algorithm — these occur when open
        # strings coincidentally cover the chord's pitch classes but no real
        # fingering was found.  Lookup-table open chords (E, Am, etc.) are
        # already returned above and are always valid.
        if voicing is not None and not any(f > 0 for f in voicing):
            voicing = None

        if voicing is not None:
            return voicing

        # Fallback: progressively simplify the chord until a known voicing exists.
        # E.g. "Emaddb9,9" → "Em7" → "Em"; "Bsus4" → already found above.
        simplified = self._simplify_chord(chord_symbol)
        if simplified and simplified != chord_symbol:
            return self.get_voicing(simplified)

        return None

    # ── ASCII diagram rendering ───────────────────────────────────────────────

    def render_diagram(self, chord_symbol: str) -> list[str]:
        """
        Return a list of text lines representing the chord diagram.

        Example output (7 lines):
            F#m7
            2fr
            ┌▬┬▬┬▬┬▬┬▬┐
            ├─┼─┼●┼─┼─┤
            ├●┼●┼─┼─┼─┤
            └─┴─┴─┴─┴─┘
            E A D G B e
        """
        voicing = self.get_voicing(chord_symbol)
        if voicing is None:
            return [chord_symbol, "(no voicing found)"]

        fretted = [f for f in voicing if f > 0]
        min_fret = min(fretted) if fretted else 1
        # For open chords start from fret 1; for barre chords from min fret
        has_open  = any(f == 0 for f in voicing if f >= 0)
        start_fret = 1 if has_open else min_fret

        n_rows = 4   # number of fret rows shown
        lines = []

        # Title line
        lines.append(chord_symbol)

        # Position marker (blank for open chords)
        if start_fret > 1:
            lines.append(f"{start_fret}fr")
        else:
            lines.append("")

        # Mute / open nut row
        nut = []
        for f in voicing:
            if f == -1:
                nut.append("×")
            elif f == 0:
                nut.append("○")
            else:
                nut.append(" ")
        lines.append("  " + " ".join(nut))

        # Determine if there is a barre
        barre_fret = self._detect_barre(voicing, start_fret)

        # Grid rows
        top_border    = "┌" + "─┬" * 5 + "─┐"
        mid_border    = "├" + "─┼" * 5 + "─┤"
        barre_border  = "┝" + "━┿" * 5 + "━┥"
        bottom_border = "└" + "─┴" * 5 + "─┘"

        for row in range(n_rows):
            fret_num = start_fret + row

            # Top/mid border
            if row == 0:
                if barre_fret == fret_num:
                    lines.append(barre_border)
                else:
                    lines.append(top_border)
            else:
                if barre_fret == fret_num:
                    lines.append(barre_border)
                else:
                    lines.append(mid_border)

            # Finger dots
            cells = []
            for f in voicing:
                if f == fret_num:
                    cells.append("●")
                else:
                    cells.append(" ")
            lines.append("│" + "│".join(cells) + "│")

        lines.append(bottom_border)
        lines.append("E A D G B e")
        return lines

    def render_diagrams_row(
        self, chord_symbols: list[str], spacing: int = 3
    ) -> str:
        """
        Render multiple chord diagrams side-by-side in a single text block.
        Returns a multi-line string.
        """
        if not chord_symbols:
            return ""

        all_diagrams = [self.render_diagram(s) for s in chord_symbols]

        # Pad all diagrams to the same height
        max_lines = max(len(d) for d in all_diagrams)
        padded = []
        for diag in all_diagrams:
            # Find the width of this diagram
            w = max(len(line) for line in diag) if diag else 0
            while len(diag) < max_lines:
                diag.append("")
            padded.append((diag, w))

        # Combine line by line
        gap = " " * spacing
        result_lines = []
        for i in range(max_lines):
            parts = []
            for diag, w in padded:
                line = diag[i] if i < len(diag) else ""
                parts.append(line.ljust(w))
            result_lines.append(gap.join(parts))

        return "\n".join(result_lines)

    # ── Algorithmic voicing generator ────────────────────────────────────────

    def _find_voicing(
        self, pitch_classes: set[int], chord_symbol: str
    ) -> list[int] | None:
        """
        Search the fretboard for the best playable voicing.

        Two searches are run in parallel:
          1. Open-chord search — allows open strings, prefers lowest position.
          2. Barre-chord search — fretted notes only, finds compact closed shapes.

        The barre result is preferred when the open search produces a voicing
        where fretted notes fall outside the 4-fret display window (i.e., the
        diagram would look all-open because high-fret notes are not rendered).
        """
        root_pc = self._root_pc(chord_symbol)
        candidates_per_string = self._candidates(pitch_classes)

        open_voicing  = self._search_open(candidates_per_string, pitch_classes, root_pc)
        barre_voicing = self._search_barre(candidates_per_string, pitch_classes, root_pc)

        # Choose the best result
        return self._pick_best(
            open_voicing, barre_voicing, pitch_classes, root_pc
        )

    def _search_open(
        self,
        candidates_per_string: list[list[int]],
        pitch_classes: set[int],
        root_pc: int,
    ) -> list[int] | None:
        """Open-chord search: open strings allowed, prefer lowest frets."""
        best_voicing = None
        best_score   = -999.0

        for window_start in range(0, self.max_fret_pos - self.max_fret_span + 2):
            window_end = window_start + self.max_fret_span

            allowed: list[list[int]] = []
            for string_candidates in candidates_per_string:
                in_window = [
                    f for f in string_candidates
                    if f == 0 or (window_start <= f <= window_end)
                ]
                in_window.append(-1)
                allowed.append(in_window)

            voicing = self._greedy_voicing(
                allowed, pitch_classes, root_pc, window_start
            )
            if voicing is None:
                continue

            score = self._score_voicing(voicing, pitch_classes, root_pc)
            if score > best_score:
                best_score   = score
                best_voicing = voicing

        return best_voicing

    def _search_barre(
        self,
        candidates_per_string: list[list[int]],
        pitch_classes: set[int],
        root_pc: int,
    ) -> list[int] | None:
        """
        Barre-chord search: no open strings — all sounding notes are fretted
        within a compact window.  Produces shapes that are always fully visible
        in the chord diagram regardless of fret position.
        """
        best_voicing = None
        best_score   = -999.0

        for window_start in range(1, self.max_fret_pos - self.max_fret_span + 2):
            window_end = window_start + self.max_fret_span

            allowed: list[list[int]] = []
            for string_candidates in candidates_per_string:
                # No open strings — only frets within the window
                in_window = [
                    f for f in string_candidates
                    if window_start <= f <= window_end
                ]
                in_window.append(-1)
                allowed.append(in_window)

            voicing = self._greedy_voicing_barre(
                allowed, pitch_classes, root_pc
            )
            if voicing is None:
                continue

            score = self._score_voicing(voicing, pitch_classes, root_pc)
            if score > best_score:
                best_score   = score
                best_voicing = voicing

        return best_voicing

    def _greedy_voicing_barre(
        self,
        allowed: list[list[int]],
        pitch_classes: set[int],
        root_pc: int,
    ) -> list[int] | None:
        """
        Greedy voicing for barre search: prefer lowest fret in window,
        no open-string preference.
        """
        voicing: list[int] = []
        covered_pcs: set[int] = set()

        for string_idx, frets in enumerate(allowed):
            chord_frets = [f for f in frets if f > 0]   # fretted only (no open, no mute)
            if chord_frets:
                chosen = min(chord_frets)               # lowest fret in window
                pc = (self._open_pc[string_idx] + chosen) % 12
                covered_pcs.add(pc)
                voicing.append(chosen)
            else:
                voicing.append(-1)

        non_muted = [f for f in voicing if f >= 0]
        fretted   = [f for f in voicing if f > 0]
        if len(non_muted) < 4:
            return None
        if not fretted:
            return None
        if root_pc not in covered_pcs:
            return None
        if len(covered_pcs) < 2:
            return None
        if not self._is_playable(voicing):
            return None
        return voicing

    @staticmethod
    def _pick_best(
        open_voicing: list[int] | None,
        barre_voicing: list[int] | None,
        pitch_classes: set[int],
        root_pc: int,
    ) -> list[int] | None:
        """
        Choose between open and barre voicings.

        Prefers the barre voicing when the open voicing has fretted notes
        above fret 4 alongside open strings — those high-fret notes fall
        outside the 4-fret display window and make the diagram look all-open.
        """
        if open_voicing is None and barre_voicing is None:
            return None
        if open_voicing is None:
            return barre_voicing
        if barre_voicing is None:
            return open_voicing

        # Check if the open voicing would look all-open in the diagram:
        # has open strings AND fretted notes all above fret 4
        open_fretted = [f for f in open_voicing if f > 0]
        has_open_str = any(f == 0 for f in open_voicing if f >= 0)
        looks_empty  = has_open_str and bool(open_fretted) and min(open_fretted) > 4

        if looks_empty:
            return barre_voicing

        return open_voicing

    def _candidates(
        self, pitch_classes: set[int]
    ) -> list[list[int]]:
        """For each string, list frets that produce a chord-tone pitch class."""
        result = []
        for open_pc in self._open_pc:
            string_candidates = []
            for fret in range(0, self.max_fret_pos + 1):
                pc = (open_pc + fret) % 12
                if pc in pitch_classes:
                    string_candidates.append(fret)
            result.append(string_candidates)
        return result

    def _greedy_voicing(
        self,
        allowed: list[list[int]],
        pitch_classes: set[int],
        root_pc: int,
        window_start: int,
    ) -> list[int] | None:
        """
        Pick one fret per string, preferring chord tones over mutes.
        Rejects voicings that are physically unplayable.
        """
        voicing = []
        covered_pcs: set[int] = set()
        for string_idx, frets in enumerate(allowed):
            chord_frets = [f for f in frets if f != -1]
            if chord_frets:
                # Prefer open strings, then lowest fret in window
                chosen = min(chord_frets, key=lambda f: (f == 0) * -10 + f)
                pc = (self._open_pc[string_idx] + chosen) % 12
                covered_pcs.add(pc)
                voicing.append(chosen)
            else:
                voicing.append(-1)

        # Basic validity checks
        non_muted = [f for f in voicing if f >= 0]
        fretted   = [f for f in voicing if f > 0]
        if len(non_muted) < 4:          # need at least 4 strings sounding
            return None
        if not fretted:                  # reject all-open voicings from algorithm
            return None
        if root_pc not in covered_pcs:
            return None
        if len(covered_pcs) < 2:
            return None
        if not self._is_playable(voicing):
            return None
        return voicing

    @staticmethod
    def _is_playable(voicing: list[int]) -> bool:
        """
        Return True only if a human hand can physically fret this voicing.

        Rules:
          1. Fret span of all pressed (non-open) notes ≤ 3 frets.
             (A standard 4-finger stretch covers frets N to N+3.)
          2. Distinct finger positions ≤ 4.
             (One finger can barre multiple strings on the same fret,
              so count unique fret values > 0, not individual strings.)
          3. At least 4 strings sounding (not muted).
        """
        fretted   = [f for f in voicing if f > 0]   # pressed, not open
        non_muted = [f for f in voicing if f >= 0]  # open + pressed

        if len(non_muted) < 4:
            return False

        if not fretted:
            return True   # all-open chord is trivially playable

        span = max(fretted) - min(fretted)
        if span > 3:
            return False

        # Count unique fret positions needed (each unique fret = 1 finger,
        # unless it's a barre which covers multiple strings at the same fret)
        unique_frets = len(set(fretted))
        if unique_frets > 4:
            return False

        return True

    def _score_voicing(
        self,
        voicing: list[int],
        pitch_classes: set[int],
        root_pc: int,
    ) -> float:
        score = 0.0
        covered: set[int] = set()

        fretted   = [f for f in voicing if f > 0]
        non_muted = [f for f in voicing if f >= 0]

        # Hard reject unplayable voicings — should not reach here, but safeguard
        if not self._is_playable(voicing):
            return -9999.0

        for string_idx, fret in enumerate(voicing):
            if fret < 0:
                score -= 0.5        # fewer mutes preferred
                continue
            pc = (self._open_pc[string_idx] + fret) % 12
            covered.add(pc)
            if fret == 0:
                score += 1.5        # open strings are easy and ring clearly
            else:
                score -= fret * 0.05  # lower positions preferred

        # Reward more strings sounding (full 6-string chord = +3)
        score += len(non_muted) * 0.5

        # Reward covering all distinct chord tones
        score += len(covered & pitch_classes) * 2.0

        # Penalise wide spans (compact voicings are easier to play)
        if fretted:
            span   = max(fretted) - min(fretted)
            score -= span * 0.4

        # Penalise many distinct finger positions
        score -= len(set(fretted)) * 0.3

        # Reward root on lowest non-muted string
        for string_idx, fret in enumerate(voicing):
            if fret >= 0:
                pc = (self._open_pc[string_idx] + fret) % 12
                if pc == root_pc:
                    score += 3.0
                break

        return score

    # ── Barre detection ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_barre(voicing: list[int], start_fret: int) -> int | None:
        """
        A barre exists when 4+ strings share the same fret == start_fret.
        Returns the barre fret number, or None.
        """
        from collections import Counter
        fret_counts = Counter(f for f in voicing if f > 0)
        for fret, count in fret_counts.items():
            if count >= 4 and fret == start_fret:
                return fret
        return None

    # ── Chord symbol parsing ─────────────────────────────────────────────────

    @staticmethod
    def _root_pc(chord_symbol: str) -> int:
        """Extract root pitch class from a chord symbol string."""
        note_pcs = {
            "C": 0, "D": 2, "E": 4, "F": 5,
            "G": 7, "A": 9, "B": 11,
        }
        if not chord_symbol:
            return 0
        root = chord_symbol[0].upper()
        pc = note_pcs.get(root, 0)
        if len(chord_symbol) > 1 and chord_symbol[1] == "#":
            pc = (pc + 1) % 12
        elif len(chord_symbol) > 1 and chord_symbol[1] == "b":
            pc = (pc - 1) % 12
        return pc

    def _simplify_chord(self, chord_symbol: str) -> str:
        """
        Return a progressively simpler chord symbol that is likely to have a
        voicing, by stripping extensions/alterations down to a core quality.

        E.g. "Em7(add9)" → "Em7" → "Em"; "Bsus4" → "Bsus4" (already simple).
        Returns empty string if no simplification is possible.
        """
        # Parse root (note + optional accidental)
        if not chord_symbol:
            return ""
        root = chord_symbol[0]
        rest = chord_symbol[1:]
        if rest and rest[0] in "#b":
            root += rest[0]
            rest = rest[1:]

        s = rest.lower()

        # Build a candidate ladder from most specific to least
        if "m7b5" in s or s.startswith("ø"):
            ladder = ["m7b5", "m7", "m"]
        elif s.startswith("dim7"):
            ladder = ["dim7", "dim", "m"]
        elif s.startswith("dim"):
            ladder = ["dim", "m"]
        elif s.startswith("aug"):
            ladder = ["aug", ""]
        elif "m(maj7)" in s:
            ladder = ["m(maj7)", "m7", "m"]
        elif s.startswith("m"):
            # Minor family — strip extensions first, then 7th
            if "maj7" in s:
                ladder = ["m(maj7)", "m7", "m"]
            elif "11" in s or "13" in s:
                ladder = ["m9", "m7", "m"]
            elif "9" in s:
                ladder = ["m9", "m7", "m"]
            else:
                ladder = ["m7", "m"]
        elif "7sus4" in s or s.startswith("7sus4"):
            ladder = ["7sus4", "sus4", ""]
        elif "sus4" in s:
            ladder = ["sus4", ""]
        elif "7sus2" in s:
            ladder = ["7sus2", "sus2", ""]
        elif "sus2" in s:
            ladder = ["sus2", ""]
        elif "maj7" in s:
            ladder = ["maj7", ""]
        else:
            # Major / dominant family
            if "13" in s:
                ladder = ["13", "9", "7", ""]
            elif "11" in s:
                ladder = ["11", "9", "7", ""]
            elif "9" in s:
                ladder = ["9", "7", ""]
            else:
                ladder = ["7", ""]

        for suffix in ladder:
            candidate = root + suffix
            if candidate != chord_symbol and candidate in _VOICING_TABLE:
                return candidate

        return ""

    def _chord_to_pitch_classes(self, chord_symbol: str) -> set[int]:
        """
        Derive the set of pitch classes for a chord symbol by delegating to
        ChordBuilder, falling back to a simple triad if parsing fails.
        """
        try:
            from src.theory.chord_builder import ChordBuilder
            from src.theory.intervals import IntervalAnalyser

            root_pc = self._root_pc(chord_symbol)
            root_note = ["C", "C#", "D", "D#", "E", "F",
                         "F#", "G", "G#", "A", "A#", "B"][root_pc]

            # Heuristic: map suffix to known interval sets
            suffix = chord_symbol[1:]
            if suffix.startswith("#") or suffix.startswith("b"):
                suffix = suffix[1:]

            # Build pitch classes from interval patterns
            return self._suffix_to_pcs(root_pc, suffix)
        except Exception:
            # Fallback: major triad
            root_pc = self._root_pc(chord_symbol)
            return {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}

    @staticmethod
    def _suffix_to_pcs(root_pc: int, suffix: str) -> set[int]:
        """Map a chord suffix to pitch class offsets relative to root."""
        # Normalise
        s = suffix.lower().strip()

        # Base intervals
        if s.startswith("m7b5") or s.startswith("ø"):
            base = [0, 3, 6, 10]
        elif s.startswith("dim7"):
            base = [0, 3, 6, 9]
        elif s.startswith("dim"):
            base = [0, 3, 6]
        elif s.startswith("aug") or s.startswith("+"):
            base = [0, 4, 8]
        elif s.startswith("m(maj7)"):
            base = [0, 3, 7, 11]
        elif s.startswith("m"):
            base = [0, 3, 7]
        elif s.startswith("sus2"):
            base = [0, 2, 7]
        elif s.startswith("sus4"):
            base = [0, 5, 7]
        else:
            base = [0, 4, 7]  # major

        # Extensions
        extra = []
        if "maj7" in s:
            extra.append(11)
        elif "7" in s and "maj" not in s:
            extra.append(10)
        if "9" in s:
            extra.append(14 % 12)
        if "11" in s:
            extra.append(17 % 12)
        if "13" in s:
            extra.append(21 % 12)
        if "add9" in s:
            extra.append(2)
        if "add11" in s or "add4" in s:
            extra.append(5)

        all_offsets = base + extra
        return {(root_pc + offset) % 12 for offset in all_offsets}
