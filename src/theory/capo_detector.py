"""
Capo detection heuristic.

Analyses the detected key and chord progression to estimate whether
a capo is likely being used, and if so, which fret and what the
open-position chord shapes would be.

Strategy:
  1. Guitar-friendly open keys are: C, G, D, A, E (major) and Am, Em, Dm (minor).
  2. If the sounding key is NOT guitar-friendly, check if transposing down
     by 1-7 semitones lands in a friendly key.
  3. Score each candidate by how many chords become common open shapes.
  4. Report the best candidate if it scores significantly better.
"""

from dataclasses import dataclass

from .intervals import NOTE_NAMES, IntervalAnalyser

# Keys that are comfortable to play on guitar without a capo
GUITAR_FRIENDLY_MAJOR = {"C", "G", "D", "A", "E"}
GUITAR_FRIENDLY_MINOR = {"Am", "Em", "Dm", "Bm"}

# Common open chord shapes (the shapes a guitarist would actually play)
OPEN_CHORD_SHAPES = {
    # Major / minor triads
    "C", "G", "D", "A", "E", "F",
    "Am", "Em", "Dm", "Bm",
    # Dominant 7ths
    "C7", "G7", "D7", "A7", "E7", "B7",
    # Major 7ths
    "Cmaj7", "Gmaj7", "Dmaj7", "Amaj7", "Emaj7", "Fmaj7",
    # Minor 7ths
    "Am7", "Em7", "Dm7", "Bm7",
    # Add9
    "Cadd9", "Gadd9", "Dadd9", "Aadd9", "Eadd9",
    # Suspended
    "Csus2", "Dsus2", "Asus2", "Esus2",
    "Csus4", "Dsus4", "Esus4", "Asus4", "Gsus4",
    # 7sus4 — very common on guitar
    "A7sus4", "D7sus4", "E7sus4", "G7sus4", "C7sus4",
    # Dsus4 specifically (Cadd9 shape moved up, very common)
    "Dsus4",
}


@dataclass
class CapoResult:
    """Result of capo detection."""
    likely_capo: bool
    fret: int                        # 0 = no capo
    sounding_key: str                # actual key heard in audio
    open_key: str                    # key as played with capo shapes
    transposed_chords: list[str]     # chord symbols as open shapes
    confidence: float                # 0.0 - 1.0
    explanation: str


class CapoDetector:
    """Detect probable capo position from detected key and chords."""

    def __init__(self):
        self._ia = IntervalAnalyser()

    def detect(
        self,
        detected_key: str,
        scale_type: str,
        chord_symbols: list[str],
    ) -> CapoResult:
        """
        Estimate capo position.

        Args:
            detected_key: Sounding key root (e.g. "Eb")
            scale_type: "major" or "natural_minor" etc.
            chord_symbols: List of unique chord symbols detected in the song

        Returns:
            CapoResult with suggested capo fret and transposed chords.
        """
        is_minor = "minor" in scale_type
        friendly = GUITAR_FRIENDLY_MINOR if is_minor else GUITAR_FRIENDLY_MAJOR

        sounding_key_label = f"{detected_key}m" if is_minor else detected_key

        # Score the original key (no capo)
        original_score = self._score_chords(chord_symbols, transpose=0)
        original_friendly = sounding_key_label in (
            GUITAR_FRIENDLY_MAJOR | {k for k in GUITAR_FRIENDLY_MINOR}
        )

        best_fret = 0
        best_score = original_score
        best_open_key = detected_key
        best_transposed = chord_symbols

        # Try capo positions 1-7
        for fret in range(1, 8):
            # Transposing down = what shapes the guitarist plays
            transposed = [self._transpose(c, -fret) for c in chord_symbols]
            transposed_key = self._transpose_note(detected_key, -fret)
            transposed_key_label = (
                f"{transposed_key}m" if is_minor else transposed_key
            )

            score = self._score_chords(transposed, transpose=0)

            # Bonus if the transposed key is guitar-friendly
            if transposed_key_label in (
                GUITAR_FRIENDLY_MAJOR | {k for k in GUITAR_FRIENDLY_MINOR}
            ):
                score += 2.0

            if score > best_score:
                best_score = score
                best_fret = fret
                best_open_key = transposed_key
                best_transposed = transposed

        # Determine confidence
        improvement = best_score - original_score
        if best_fret == 0:
            confidence = 0.0
            likely = False
            explanation = (
                f"Key of {sounding_key_label} is guitar-friendly. "
                "No capo needed."
            )
        elif improvement < 1.5:
            confidence = min(0.3, improvement / 5.0)
            likely = False
            explanation = (
                f"Key of {sounding_key_label} could use capo {best_fret} "
                f"(play in {best_open_key}), but improvement is marginal."
            )
        else:
            confidence = min(1.0, improvement / 8.0)
            likely = True
            open_key_label = (
                f"{best_open_key}m" if is_minor else best_open_key
            )
            explanation = (
                f"Capo {best_fret} suggested — "
                f"play in {open_key_label} using open shapes."
            )

        return CapoResult(
            likely_capo=likely,
            fret=best_fret,
            sounding_key=sounding_key_label,
            open_key=f"{best_open_key}m" if is_minor else best_open_key,
            transposed_chords=best_transposed,
            confidence=confidence,
            explanation=explanation,
        )

    def _score_chords(self, chord_symbols: list[str], transpose: int = 0) -> float:
        """Score how 'guitar-friendly' a set of chords is."""
        if not chord_symbols:
            return 0.0

        score = 0.0
        for symbol in chord_symbols:
            if transpose != 0:
                symbol = self._transpose(symbol, transpose)
            if symbol in OPEN_CHORD_SHAPES:
                score += 1.0
            elif self._root_of(symbol) in {"C", "G", "D", "A", "E"}:
                score += 0.3  # partial credit for right root
        return score

    def _transpose(self, chord_symbol: str, semitones: int) -> str:
        """Transpose a chord symbol by N semitones."""
        root, suffix = self._split_chord(chord_symbol)
        new_root = self._transpose_note(root, semitones)
        return f"{new_root}{suffix}"

    def _transpose_note(self, note: str, semitones: int) -> str:
        """Transpose a note name by N semitones."""
        pc = self._ia.note_to_pitch_class(note)
        new_pc = (pc + semitones) % 12
        return NOTE_NAMES[new_pc]

    @staticmethod
    def _split_chord(symbol: str) -> tuple[str, str]:
        """Split 'C#m7' into ('C#', 'm7'), 'Bbm' into ('Bb', 'm')."""
        if not symbol:
            return symbol, ""
        # Root is always a letter A-G, optionally followed by # or b
        root = symbol[0]
        rest = symbol[1:]
        if rest and rest[0] in ("#", "b"):
            # Only treat as accidental if root+accidental is a valid note
            candidate = root + rest[0]
            valid_notes = {
                "C#", "Db", "D#", "Eb", "F#", "Gb", "G#", "Ab", "A#", "Bb",
            }
            if candidate in valid_notes:
                root = candidate
                rest = rest[1:]
        return root, rest

    @staticmethod
    def _root_of(symbol: str) -> str:
        """Extract root note from chord symbol."""
        root, _ = CapoDetector._split_chord(symbol)
        return root
