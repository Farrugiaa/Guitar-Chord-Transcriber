"""
Chord symbol builder from interval analysis results.

Converts detected pitch classes into properly formatted chord symbols
(e.g. Cmaj7, Am7b5, G13#11) using scale-degree-based construction.
"""

from typing import Optional
from .intervals import (
    IntervalAnalyser,
    IntervalResult,
    ChordQuality,
    NOTE_NAMES,
    ENHARMONIC_SHARP_TO_FLAT,
)


# Keys that prefer flats in their spelling
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb"}
SHARP_KEYS = {"G", "D", "A", "E", "B", "F#"}


class ChordBuilder:
    """
    Build chord symbols from detected pitch classes.

    Uses interval-based analysis: identifies root, builds 1-3-5 triad,
    then detects extensions (7th, 9th, 11th, 13th) to produce accurate
    chord names.
    """

    def __init__(self):
        self.analyser = IntervalAnalyser()

    def build_chord_symbol(
        self,
        root: str,
        pitch_classes: list[int],
        bass_pitch_class: Optional[int] = None,
        key_context: Optional[str] = None,
    ) -> str:
        """
        Build a chord symbol string from root and detected pitch classes.

        Args:
            root: Root note name
            pitch_classes: Detected pitch classes (0-11)
            bass_pitch_class: Lowest note pitch class for slash chords
            key_context: Current key for enharmonic spelling decisions

        Returns:
            Chord symbol string (e.g. "Cmaj7", "Am7", "G7#9")
        """
        result = self.analyser.analyse(root, pitch_classes, bass_pitch_class)
        symbol = self._format_root(result.root, key_context)
        symbol += self._format_quality_and_intervals(result)

        if result.bass_note:
            bass = self._format_root(result.bass_note, key_context)
            symbol += f"/{bass}"

        return symbol

    def identify_chord_from_pitches(
        self,
        pitch_classes: list[int],
        key_context: Optional[str] = None,
        scale_pcs: Optional[set[int]] = None,
        onset_root_pc: Optional[int] = None,
    ) -> str:
        """
        Try each pitch class as root, score each interpretation, return the best.
        onset_root_pc gets +2.0 bonus — bass string hit first on strum disambiguates
        chords like G major vs Em7 (identical pitch-class sets).
        """
        if not pitch_classes:
            return "N.C."

        candidates = []
        unique_pcs = sorted(set(pitch_classes))

        for root_pc in unique_pcs:
            root = IntervalAnalyser.pitch_class_to_note(root_pc)
            result = self.analyser.analyse(root, unique_pcs)
            score = self._score_interpretation(result, root_pc, unique_pcs)
            if onset_root_pc is not None and root_pc == onset_root_pc:
                score += 2.0
            if scale_pcs and root_pc in scale_pcs:
                score += 1.5
            symbol = self.build_chord_symbol(root, unique_pcs, key_context=key_context)
            candidates.append((score, symbol, result))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates else "N.C."

    def _score_interpretation(
        self, result: IntervalResult, root_pc: int, pitch_classes: list[int]
    ) -> float:
        """Score how well an interval interpretation fits as a chord."""
        score = 0.0

        if result.has_third():
            score += 3.0
        elif result.quality in (ChordQuality.SUSPENDED_2, ChordQuality.SUSPENDED_4):
            score += 2.5  # sus chords are valid without a third

        if result.fifth == "5":
            score += 2.0
        elif result.fifth is not None:
            score += 1.0

        if result.quality != ChordQuality.UNKNOWN:
            score += 2.0

        if result.quality in (ChordQuality.MAJOR, ChordQuality.MINOR, ChordQuality.DOMINANT):
            score += 1.0
        elif result.quality in (ChordQuality.SUSPENDED_2, ChordQuality.SUSPENDED_4):
            score += 0.7

        if result.quality in (ChordQuality.SUSPENDED_2, ChordQuality.SUSPENDED_4) and result.has_seventh():
            score += 0.5  # 7sus4 is common on guitar

        score -= len(result.extensions) * 0.3

        if pitch_classes and pitch_classes[0] == root_pc:
            score += 1.0

        _COMMON_ROOTS = {4, 9, 2, 7, 11, 5}  # E, A, D, G, B, F open strings
        if root_pc in _COMMON_ROOTS:
            score += 0.4

        return score

    def _format_root(self, note: str, key_context: Optional[str] = None) -> str:
        """Format root note with correct enharmonic spelling for the key."""
        if key_context and key_context in FLAT_KEYS:
            if note in ENHARMONIC_SHARP_TO_FLAT:
                return ENHARMONIC_SHARP_TO_FLAT[note]
        return note

    def _format_quality_and_intervals(self, result: IntervalResult) -> str:
        """Build the suffix of a chord symbol from analysis result."""
        q = result.quality
        third = result.third
        fifth = result.fifth
        seventh = result.seventh
        ext = result.extensions

        if q == ChordQuality.POWER:
            return "5"
        if q == ChordQuality.UNKNOWN:
            if seventh or ext:
                return self._build_extension_suffix(third, fifth, seventh, ext)
            return "?"

        if q == ChordQuality.SUSPENDED_2:
            base = "sus2"
            if seventh == "b7":
                base = "7sus2"
            elif seventh == "7":
                base = "maj7sus2"
            return base + self._format_extensions(ext)

        if q == ChordQuality.SUSPENDED_4:
            base = "sus4"
            if seventh == "b7":
                base = "7sus4"
            elif seventh == "7":
                base = "maj7sus4"
            return base + self._format_extensions(ext)

        if q == ChordQuality.DIMINISHED:
            if seventh == "b7":
                return "m7b5" + self._format_extensions(ext)
            elif seventh == "7":
                return "dim(maj7)" + self._format_extensions(ext)
            elif seventh is None:
                return "dim" + self._format_extensions(ext)
            return "dim7" + self._format_extensions(ext)

        if q == ChordQuality.AUGMENTED:
            base = "aug"
            if seventh == "b7":
                base = "aug7"
            elif seventh == "7":
                base = "aug(maj7)"
            return base + self._format_extensions(ext)

        if q == ChordQuality.MINOR:
            return self._build_minor_symbol(seventh, ext)

        if q == ChordQuality.DOMINANT:
            return self._build_dominant_symbol(ext)

        if q == ChordQuality.MAJOR:
            return self._build_major_symbol(seventh, ext)

        return ""

    def _build_minor_symbol(
        self, seventh: Optional[str], ext: list[str]
    ) -> str:
        if seventh is None:
            if ext:
                return "m" + self._format_add_extensions(ext)
            return "m"

        if seventh == "7":
            base = "m(maj7)"
        else:
            if ext:
                highest = self._highest_extension(ext)
                if highest:
                    remaining = [e for e in ext if e != highest]
                    return f"m{highest}" + self._format_alterations(remaining)
            base = "m7"
        return base + self._format_extensions(ext)

    def _build_dominant_symbol(self, ext: list[str]) -> str:
        if not ext:
            return "7"
        highest = self._highest_extension(ext)
        if highest:
            remaining = [e for e in ext if e != highest]
            return highest + self._format_alterations(remaining)
        return "7" + self._format_extensions(ext)

    def _build_major_symbol(
        self, seventh: Optional[str], ext: list[str]
    ) -> str:
        if seventh is None:
            if ext:
                return self._format_add_extensions(ext)
            return ""

        if seventh == "7":
            if ext:
                highest = self._highest_extension(ext)
                if highest:
                    remaining = [e for e in ext if e != highest]
                    return f"maj{highest}" + self._format_alterations(remaining)
            return "maj7"

        return ""

    def _build_extension_suffix(
        self,
        third: Optional[str],
        fifth: Optional[str],
        seventh: Optional[str],
        ext: list[str],
    ) -> str:
        parts = []
        if seventh == "b7":
            parts.append("7")
        elif seventh == "7":
            parts.append("maj7")
        if fifth == "b5":
            parts.append("b5")
        elif fifth == "#5":
            parts.append("#5")
        parts.extend(ext)
        return "".join(parts)

    @staticmethod
    def _highest_extension(ext: list[str]) -> Optional[str]:
        """Get the highest natural extension (9, 11, 13)."""
        for target in ["13", "11", "9"]:
            if target in ext:
                return target
        return None

    @staticmethod
    def _format_extensions(ext: list[str]) -> str:
        if not ext:
            return ""
        return "(" + ",".join(ext) + ")"

    @staticmethod
    def _format_add_extensions(ext: list[str]) -> str:
        if not ext:
            return ""
        return "add" + ",".join(ext)

    @staticmethod
    def _format_alterations(ext: list[str]) -> str:
        if not ext:
            return ""
        return "(" + ",".join(ext) + ")"
