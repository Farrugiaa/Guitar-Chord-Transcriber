"""Unit tests for evaluation helper functions (written before implementation)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


# ── internal_to_harte ────────────────────────────────────────────────────────

def test_nc_maps_to_N():
    from evaluation.evaluate_wcsr import internal_to_harte
    assert internal_to_harte("N.C.") == "N"

def test_major_maps_to_harte_maj():
    from evaluation.evaluate_wcsr import internal_to_harte
    assert internal_to_harte("C") == "C:maj"
    assert internal_to_harte("F#") == "F#:maj"

def test_minor_maps_to_harte_min():
    from evaluation.evaluate_wcsr import internal_to_harte
    assert internal_to_harte("Am") == "A:min"
    assert internal_to_harte("Bbm") == "Bb:min"

def test_dom7_maps_to_harte_7():
    from evaluation.evaluate_wcsr import internal_to_harte
    assert internal_to_harte("G7") == "G:7"

def test_hdim_maps_to_harte_hdim7():
    from evaluation.evaluate_wcsr import internal_to_harte
    assert internal_to_harte("Bm7b5") == "B:hdim7"


# ── frames_to_intervals ───────────────────────────────────────────────────────

def test_single_chord_one_interval():
    from evaluation.evaluate_wcsr import frames_to_intervals, ChordVocabulary
    vocab = ChordVocabulary(level="extended")
    preds = np.zeros(10, dtype=int)  # all index 0 = N.C.
    ivs, labels = frames_to_intervals(preds, vocab, hop=512, sr=44100)
    assert ivs.shape == (1, 2)
    assert labels[0] == "N"

def test_two_chords_two_intervals():
    from evaluation.evaluate_wcsr import frames_to_intervals, ChordVocabulary
    vocab = ChordVocabulary(level="extended")
    preds = np.array([1]*5 + [2]*5, dtype=int)
    ivs, labels = frames_to_intervals(preds, vocab, hop=512, sr=44100)
    assert len(ivs) == 2
    assert len(labels) == 2

def test_intervals_are_contiguous():
    from evaluation.evaluate_wcsr import frames_to_intervals, ChordVocabulary
    vocab = ChordVocabulary(level="extended")
    preds = np.array([1, 1, 2, 2, 3], dtype=int)
    ivs, labels = frames_to_intervals(preds, vocab, hop=512, sr=44100)
    for i in range(len(ivs) - 1):
        assert abs(ivs[i, 1] - ivs[i + 1, 0]) < 1e-9, "Intervals must be contiguous"
