"""Tests for the music theory module."""

import pytest
from src.theory.intervals import IntervalAnalyser, ChordQuality
from src.theory.chord_builder import ChordBuilder
from src.theory.scales import ScaleAnalyser


class TestIntervalAnalyser:

    def setup_method(self):
        self.analyser = IntervalAnalyser()

    def test_c_major_triad(self):
        # C=0, E=4, G=7
        result = self.analyser.analyse("C", [0, 4, 7])
        assert result.quality == ChordQuality.MAJOR
        assert result.third == "3"
        assert result.fifth == "5"

    def test_a_minor_triad(self):
        # A=9, C=0, E=4
        result = self.analyser.analyse("A", [9, 0, 4])
        assert result.quality == ChordQuality.MINOR
        assert result.third == "b3"

    def test_g_dominant_seventh(self):
        # G=7, B=11, D=2, F=5
        result = self.analyser.analyse("G", [7, 11, 2, 5])
        assert result.quality == ChordQuality.DOMINANT
        assert result.seventh == "b7"

    def test_diminished(self):
        # B=11, D=2, F=5
        result = self.analyser.analyse("B", [11, 2, 5])
        assert result.quality == ChordQuality.DIMINISHED
        assert result.fifth == "b5"

    def test_augmented(self):
        # C=0, E=4, G#=8
        result = self.analyser.analyse("C", [0, 4, 8])
        assert result.quality == ChordQuality.AUGMENTED
        assert result.fifth == "#5"

    def test_note_to_pitch_class(self):
        assert IntervalAnalyser.note_to_pitch_class("C") == 0
        assert IntervalAnalyser.note_to_pitch_class("A") == 9
        assert IntervalAnalyser.note_to_pitch_class("Bb") == 10


class TestChordBuilder:

    def setup_method(self):
        self.builder = ChordBuilder()

    def test_c_major(self):
        symbol = self.builder.build_chord_symbol("C", [0, 4, 7])
        assert symbol == "C"

    def test_a_minor(self):
        symbol = self.builder.build_chord_symbol("A", [9, 0, 4])
        assert symbol == "Am"

    def test_g7(self):
        symbol = self.builder.build_chord_symbol("G", [7, 11, 2, 5])
        assert symbol == "G7"

    def test_cmaj7(self):
        # C=0, E=4, G=7, B=11
        symbol = self.builder.build_chord_symbol("C", [0, 4, 7, 11])
        assert symbol == "Cmaj7"

    def test_am7(self):
        # A=9, C=0, E=4, G=7
        symbol = self.builder.build_chord_symbol("A", [9, 0, 4, 7])
        assert symbol == "Am7"

    def test_identify_from_pitches(self):
        # C major: C E G
        symbol = self.builder.identify_chord_from_pitches([0, 4, 7])
        assert symbol == "C"

    def test_no_chord(self):
        symbol = self.builder.identify_chord_from_pitches([])
        assert symbol == "N.C."

    def test_power_chord(self):
        # C5: C G (no third)
        symbol = self.builder.build_chord_symbol("C", [0, 7])
        assert symbol == "C5"

    def test_slash_chord(self):
        # C/E = C major with E in bass
        symbol = self.builder.build_chord_symbol("C", [0, 4, 7], bass_pitch_class=4)
        assert symbol == "C/E"


class TestScaleAnalyser:

    def setup_method(self):
        self.analyser = ScaleAnalyser()

    def test_c_major_scale(self):
        notes = self.analyser.get_scale_notes("C", "major")
        assert notes == ["C", "D", "E", "F", "G", "A", "B"]

    def test_a_minor_scale(self):
        notes = self.analyser.get_scale_notes("A", "natural_minor")
        assert notes == ["A", "B", "C", "D", "E", "F", "G"]

    def test_diatonic_chords_c_major(self):
        chords = self.analyser.get_diatonic_chords("C", "major")
        assert len(chords) == 7
        # I chord should be Cmaj7
        assert chords[0]["root"] == "C"
        assert "maj7" in chords[0]["symbol"]

    def test_key_detection(self):
        # Perfect C major distribution
        histogram = [1.0, 0, 0.5, 0, 0.7, 0.6, 0, 0.8, 0, 0.4, 0, 0.3]
        candidates = self.analyser.detect_key(histogram)
        assert candidates[0][0] == "C"  # root should be C
