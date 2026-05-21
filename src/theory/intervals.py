"""
Interval analysis for chord construction from scale degrees.

Builds chords by analysing pitch classes relative to a root note,
identifying the 1-3-5 triad core and any extensions (7th, 9th, 11th, 13th).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic mappings for cleaner chord spelling
ENHARMONIC_SHARP_TO_FLAT = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"
}

# Interval definitions: semitones from root -> (scale degree, quality)
INTERVAL_MAP = {
    0:  ("1", "root"),
    1:  ("b2", "minor_second"),
    2:  ("2", "major_second"),
    3:  ("b3", "minor_third"),
    4:  ("3", "major_third"),
    5:  ("4", "perfect_fourth"),
    6:  ("b5", "diminished_fifth"),
    7:  ("5", "perfect_fifth"),
    8:  ("#5", "augmented_fifth"),
    9:  ("6", "major_sixth"),
    10: ("b7", "minor_seventh"),
    11: ("7", "major_seventh"),
}

# Extended intervals (compound intervals, same pitch class but octave+ above)
EXTENDED_INTERVAL_MAP = {
    1:  ("b9", "flat_ninth"),
    2:  ("9", "ninth"),
    3:  ("#9", "sharp_ninth"),
    5:  ("11", "eleventh"),
    6:  ("#11", "sharp_eleventh"),
    8:  ("b13", "flat_thirteenth"),
    9:  ("13", "thirteenth"),
}


class ChordQuality(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT = "dominant"
    SUSPENDED_2 = "sus2"
    SUSPENDED_4 = "sus4"
    POWER = "power"
    UNKNOWN = "unknown"


@dataclass
class IntervalResult:
    """Result of analysing a set of pitch classes against a root."""
    root: str
    quality: ChordQuality
    third: Optional[str]        # "3" or "b3" or None
    fifth: Optional[str]        # "5", "b5", "#5", or None
    seventh: Optional[str]      # "7", "b7", or None
    extensions: list[str]       # ["9", "#11", "13", etc.]
    bass_note: Optional[str]    # for slash chords (inversions)
    all_intervals: dict[int, str]  # semitone -> degree label

    def has_third(self) -> bool:
        return self.third is not None

    def has_seventh(self) -> bool:
        return self.seventh is not None

    def has_extensions(self) -> bool:
        return len(self.extensions) > 0


class IntervalAnalyser:
    """Analyse pitch classes relative to a root to determine chord intervals."""

    @staticmethod
    def note_to_pitch_class(note: str) -> int:
        """Convert note name to pitch class (0-11). Handles flats and sharps."""
        note = note.strip()
        # Handle standard flats (Db, Eb, Gb, Ab, Bb)
        flat_to_sharp = {v: k for k, v in ENHARMONIC_SHARP_TO_FLAT.items()}
        if note in flat_to_sharp:
            note = flat_to_sharp[note]
        # Handle edge-case enharmonics from transposition
        edge_cases = {
            "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
        }
        if note in edge_cases:
            note = edge_cases[note]
        try:
            return NOTE_NAMES.index(note)
        except ValueError:
            raise ValueError(f"Unknown note: {note}")

    @staticmethod
    def pitch_class_to_note(pc: int) -> str:
        """Convert pitch class (0-11) to note name."""
        return NOTE_NAMES[pc % 12]

    def analyse(
        self,
        root: str,
        pitch_classes: list[int],
        bass_pitch_class: Optional[int] = None,
    ) -> IntervalResult:
        """
        Analyse a set of pitch classes relative to a root note.

        Args:
            root: Root note name (e.g. "C", "F#")
            pitch_classes: List of detected pitch classes (0-11)
            bass_pitch_class: Lowest sounding pitch class, for slash chords

        Returns:
            IntervalResult with identified intervals and chord quality.
        """
        root_pc = self.note_to_pitch_class(root)
        intervals = {}

        for pc in set(pitch_classes):
            semitones = (pc - root_pc) % 12
            if semitones in INTERVAL_MAP:
                degree, _ = INTERVAL_MAP[semitones]
                intervals[semitones] = degree

        # Identify core triad components
        third = None
        if 4 in intervals:
            third = "3"
        elif 3 in intervals:
            third = "b3"

        fifth = None
        if 7 in intervals:
            fifth = "5"
        elif 6 in intervals:
            fifth = "b5"
        elif 8 in intervals:
            fifth = "#5"

        # Identify seventh
        seventh = None
        if 11 in intervals:
            seventh = "7"
        elif 10 in intervals:
            seventh = "b7"

        # Identify extensions (notes that go beyond the 7th)
        extensions = []
        for semitones_val, (degree, _) in EXTENDED_INTERVAL_MAP.items():
            if semitones_val in intervals or semitones_val in {
                (pc - root_pc) % 12 for pc in pitch_classes
            }:
                # Only count as extension if 7th is present or this is clearly
                # an added tone
                if seventh is not None or semitones_val in {
                    (pc - root_pc) % 12 for pc in pitch_classes
                }:
                    # Don't double-count: b3 (semitone 3) vs #9 (also 3)
                    if semitones_val == 3 and third == "b3":
                        continue
                    # #9 on a major chord only makes sense with a b7 (Hendrix
                    # chord). Without b7 the minor-3rd pitch class is chroma
                    # bleed from the adjacent major 3rd — suppress it.
                    if semitones_val == 3 and third == "3" and seventh != "b7":
                        continue
                    # Don't count 4th/11th if it's likely a sus4
                    if semitones_val == 5 and third is None:
                        continue
                    # b5/#11 disambiguation: if 5th is present, treat 6 as #11
                    if semitones_val == 6 and fifth == "5" and seventh is not None:
                        extensions.append("#11")
                        continue
                    if semitones_val in [s for s in EXTENDED_INTERVAL_MAP]:
                        extensions.append(degree)

        # Determine quality
        quality = self._determine_quality(third, fifth, seventh, intervals)

        # Bass note for slash chords
        bass_note = None
        if bass_pitch_class is not None:
            bass_pc = bass_pitch_class % 12
            if bass_pc != root_pc:
                bass_note = self.pitch_class_to_note(bass_pc)

        return IntervalResult(
            root=root,
            quality=quality,
            third=third,
            fifth=fifth,
            seventh=seventh,
            extensions=sorted(extensions, key=self._extension_sort_key),
            bass_note=bass_note,
            all_intervals=intervals,
        )

    def _determine_quality(
        self,
        third: Optional[str],
        fifth: Optional[str],
        seventh: Optional[str],
        intervals: dict[int, str],
    ) -> ChordQuality:
        """Determine overall chord quality from core intervals."""
        has_sus2 = 2 in intervals and third is None
        has_sus4 = 5 in intervals and third is None

        if has_sus2:
            return ChordQuality.SUSPENDED_2
        if has_sus4:
            return ChordQuality.SUSPENDED_4

        if third is None and fifth == "5":
            return ChordQuality.POWER

        if third == "b3":
            if fifth == "b5":
                return ChordQuality.DIMINISHED
            return ChordQuality.MINOR
        elif third == "3":
            if fifth == "#5":
                return ChordQuality.AUGMENTED
            if seventh == "b7":
                return ChordQuality.DOMINANT
            return ChordQuality.MAJOR

        return ChordQuality.UNKNOWN

    @staticmethod
    def _extension_sort_key(ext: str) -> int:
        order = {"b9": 0, "9": 1, "#9": 2, "11": 3, "#11": 4, "b13": 5, "13": 6}
        return order.get(ext, 99)
