"""
Scale definitions and scale-degree chord construction.

Provides scale templates and methods to build diatonic chords
from scale degrees, used to contextualise detected chords
within a key.
"""

from typing import Optional
from .intervals import IntervalAnalyser, NOTE_NAMES


# Scale templates as semitone intervals from root
SCALE_TEMPLATES = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":    [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "lydian":           [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "locrian":          [0, 1, 3, 5, 6, 8, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

# Roman numeral labels for diatonic scale degrees
ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]


class ScaleAnalyser:
    """Analyse chords in the context of a musical scale/key."""

    def __init__(self):
        self.interval_analyser = IntervalAnalyser()

    def get_scale_notes(self, root: str, scale_type: str = "major") -> list[str]:
        """Get the note names for a given scale."""
        if scale_type not in SCALE_TEMPLATES:
            raise ValueError(f"Unknown scale type: {scale_type}")
        root_pc = IntervalAnalyser.note_to_pitch_class(root)
        template = SCALE_TEMPLATES[scale_type]
        return [NOTE_NAMES[(root_pc + interval) % 12] for interval in template]

    def get_scale_pitch_classes(
        self, root: str, scale_type: str = "major"
    ) -> list[int]:
        """Get pitch classes for a given scale."""
        root_pc = IntervalAnalyser.note_to_pitch_class(root)
        template = SCALE_TEMPLATES[scale_type]
        return [(root_pc + interval) % 12 for interval in template]

    def build_diatonic_chord(
        self,
        root: str,
        scale_type: str = "major",
        degree: int = 1,
        extensions: int = 3,
    ) -> list[int]:
        """
        Build a diatonic chord on a given scale degree.

        Args:
            root: Scale root note
            scale_type: Type of scale
            degree: Scale degree (1-7)
            extensions: Number of notes in chord (3=triad, 4=7th, 5=9th, etc.)

        Returns:
            List of pitch classes forming the chord.
        """
        scale_pcs = self.get_scale_pitch_classes(root, scale_type)
        n = len(scale_pcs)
        start_idx = degree - 1

        chord_pcs = []
        for i in range(extensions):
            idx = (start_idx + i * 2) % n
            chord_pcs.append(scale_pcs[idx])

        return chord_pcs

    def get_diatonic_chords(
        self,
        root: str,
        scale_type: str = "major",
        include_sevenths: bool = True,
    ) -> list[dict]:
        """
        Get all diatonic chords for a scale.

        Returns list of dicts with 'degree', 'roman', 'root', 'pitch_classes'.
        """
        from .chord_builder import ChordBuilder
        builder = ChordBuilder()

        scale_notes = self.get_scale_notes(root, scale_type)
        chords = []

        extensions = 4 if include_sevenths else 3

        for i, note in enumerate(scale_notes):
            degree = i + 1
            pcs = self.build_diatonic_chord(root, scale_type, degree, extensions)
            symbol = builder.build_chord_symbol(note, pcs, key_context=root)
            chords.append({
                "degree": degree,
                "roman": ROMAN_NUMERALS[i] if i < 7 else str(degree),
                "root": note,
                "pitch_classes": pcs,
                "symbol": f"{symbol}",
            })

        return chords

    def detect_key(
        self, pitch_class_histogram: list[float]
    ) -> list[tuple[str, str, float]]:
        """
        Estimate the key from a pitch class distribution using
        Krumhansl-Schmuckler key-finding algorithm.

        Args:
            pitch_class_histogram: 12-element list of pitch class weights

        Returns:
            Sorted list of (root, scale_type, correlation) candidates.
        """
        # Krumhansl-Kessler major and minor profiles
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        candidates = []
        for root_pc in range(12):
            # Rotate histogram so root is at index 0
            rotated = [
                pitch_class_histogram[(root_pc + i) % 12] for i in range(12)
            ]
            # Correlate with major profile
            corr_major = self._pearson_correlation(rotated, major_profile)
            candidates.append(
                (NOTE_NAMES[root_pc], "major", corr_major)
            )
            # Correlate with minor profile
            corr_minor = self._pearson_correlation(rotated, minor_profile)
            candidates.append(
                (NOTE_NAMES[root_pc], "natural_minor", corr_minor)
            )

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)
