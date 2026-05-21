#!/usr/bin/env python3
"""
WCSR evaluation for the guitar-only CRNN chord recogniser.

Test set: GuitarSet Player 05 mic recordings (held out from CRNN training).
Metric:   Weighted Chord Symbol Recall via mir_eval.chord.evaluate

Run from project root:
    python evaluation/evaluate_wcsr.py

Optional arguments:
    --guitarset-dir PATH     default: data/guitarset
    --checkpoint    PATH     default: guitar_only/checkpoints/chord_recogniser_guitar.pt
    --output-csv    PATH     default: evaluation/wcsr_results.csv
    --confidence-gate FLOAT  default: 0.35
    --median-kernel INT      default: 9
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import numpy as np
import torch
from scipy.signal import medfilt
import mir_eval

from guitar_only.src.models.chord_recogniser import ChordCRNN, ChordVocabulary


# ---------------------------------------------------------------------------
# Vocabulary conversion: internal symbol → Harte notation for mir_eval
# ---------------------------------------------------------------------------
_QUALITY_TO_HARTE = {
    "": "maj",
    "m": "min",
    "7": "7",
    "maj7": "maj7",
    "m7": "min7",
    "dim": "dim",
    "m7b5": "hdim7",
    "dim7": "dim7",
    "aug": "aug",
    "aug7": "aug7",
    "sus2": "sus2",
    "sus4": "sus4",
    "7sus4": "7sus4",
    "9": "9",
    "m9": "min9",
    "maj9": "maj9",
    "11": "11",
    "m11": "min11",
    "13": "13",
    "m13": "min13",
    "maj13": "maj13",
    "add9": "(*9)",
    "6": "6",
    "m6": "min6",
    "5": "5",
}


def internal_to_harte(symbol: str) -> str:
    """Convert internal vocabulary symbol to Harte notation for mir_eval."""
    if symbol in ("N.C.", "?", ""):
        return "N"
    if len(symbol) >= 2 and symbol[1] in ("#", "b"):
        root, quality = symbol[:2], symbol[2:]
    else:
        root, quality = symbol[0], symbol[1:]
    harte_q = _QUALITY_TO_HARTE.get(quality, quality)
    return f"{root}:{harte_q}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    """Load CRNN and vocabulary. Auto-detects n_classes from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state

    # Read output dimension from the final linear layer weight
    n_classes = model_state["classifier.3.weight"].shape[0]

    vocab = ChordVocabulary(level="extended")
    if n_classes != len(vocab):
        print(f"  Note: checkpoint has {n_classes} classes; "
              f"ChordVocabulary(extended) has {len(vocab)}. Using {n_classes}.")

    model = ChordCRNN(n_cqt_bins=84, n_classes=n_classes)
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model, vocab


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def compute_cqt(audio: np.ndarray, sr: int = 44100, hop: int = 512) -> np.ndarray:
    """84-bin log-CQT matching training parameters. Returns (84, T)."""
    import librosa
    C = np.abs(librosa.cqt(
        y=audio, sr=sr, hop_length=hop,
        n_bins=84, bins_per_octave=12, fmin=32.7,
    ))
    return np.log(C + 1e-8)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: ChordCRNN,
    cqt: np.ndarray,
    device: torch.device,
    gate: float = 0.35,
    median_k: int = 9,
) -> np.ndarray:
    """
    Run CRNN forward pass on a full-clip CQT.

    Confidence gate: frames where max(softmax) < gate have no chroma fallback
    available in standalone evaluation, so argmax is used for all frames
    regardless of confidence. The gate is checked explicitly but does not
    change output in this context.

    Returns per-frame class indices, shape (T,).
    """
    x = torch.from_numpy(cqt).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)                              # (1, T, n_classes)
    probs = torch.softmax(logits[0], dim=-1)           # (T, n_classes)
    max_conf, preds = probs.max(dim=-1)                # both (T,)
    preds_np = preds.cpu().numpy()                     # (T,) int

    # No chroma fallback available in standalone evaluation; argmax is used
    # for all frames regardless of the gate. The gate parameter is preserved
    # for API compatibility with the full pipeline.
    _ = max_conf  # gate check would go here if fallback were available

    if median_k > 1:
        preds_np = medfilt(preds_np.astype(float), kernel_size=median_k).astype(int)
    return preds_np


# ---------------------------------------------------------------------------
# Interval conversion
# ---------------------------------------------------------------------------
def frames_to_intervals(
    preds: np.ndarray,
    vocab: ChordVocabulary,
    hop: int = 512,
    sr: int = 44100,
):
    """
    Run-length encode frame predictions into (intervals, harte_labels).

    Returns
    -------
    intervals : np.ndarray (N, 2)  start/end times in seconds
    labels    : list[str]          Harte chord strings for mir_eval
    """
    if len(preds) == 0:
        return np.zeros((0, 2)), []
    frame_dur = hop / sr
    intervals, labels = [], []
    start_idx = 0
    for i in range(1, len(preds)):
        if preds[i] != preds[start_idx]:
            intervals.append([start_idx * frame_dur, i * frame_dur])
            labels.append(internal_to_harte(vocab.decode(int(preds[start_idx]))))
            start_idx = i
    intervals.append([start_idx * frame_dur, len(preds) * frame_dur])
    labels.append(internal_to_harte(vocab.decode(int(preds[start_idx]))))
    return np.array(intervals), labels


# ---------------------------------------------------------------------------
# JAMS annotation loading
# ---------------------------------------------------------------------------
def load_jams_chords(jams_path: Path):
    """
    Load chord annotations from a GuitarSet JAMS file.

    Returns
    -------
    intervals : np.ndarray (N, 2)  start/end in seconds (already Harte from JAMS)
    labels    : list[str]          Harte chord strings
    """
    import jams
    jam = jams.load(str(jams_path))
    chord_anns = jam.search(namespace="chord") or jam.search(namespace="chord_harte")
    if not chord_anns:
        return np.zeros((0, 2)), []

    ann = chord_anns[0]
    intervals, labels = [], []
    for obs in ann.data:
        t = float(obs.time.total_seconds() if hasattr(obs.time, "total_seconds") else obs.time)
        d = float(obs.duration.total_seconds() if hasattr(obs.duration, "total_seconds") else obs.duration)
        if d <= 0:
            continue
        label = str(obs.value)
        if label in ("N", "X"):
            label = "N"
        intervals.append([t, t + d])
        labels.append(label)

    if not intervals:
        return np.zeros((0, 2)), []
    return np.array(intervals), labels


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------
def evaluate_clip(
    mic_path: Path,
    jams_path: Path,
    model: ChordCRNN,
    vocab: ChordVocabulary,
    device: torch.device,
    gate: float = 0.35,
    median_k: int = 9,
) -> dict:
    """Evaluate one clip. Returns {'wcsr', 'root_acc', 'symbol_acc'}."""
    import librosa
    audio, _ = librosa.load(str(mic_path), sr=44100, mono=True)
    cqt = compute_cqt(audio)
    preds = run_inference(model, cqt, device, gate=gate, median_k=median_k)

    est_ivs, est_labels = frames_to_intervals(preds, vocab)
    ref_ivs, ref_labels = load_jams_chords(jams_path)

    if len(ref_ivs) == 0 or len(est_ivs) == 0:
        return {"wcsr": 0.0, "root_acc": 0.0, "symbol_acc": 0.0}

    try:
        scores = mir_eval.chord.evaluate(ref_ivs, ref_labels, est_ivs, est_labels)
        return {
            "wcsr":       float(scores.get("majmin", 0.0)),
            "root_acc":   float(scores.get("root",   0.0)),
            "symbol_acc": float(scores.get("sevenths", 0.0)),
        }
    except Exception as e:
        print(f"  mir_eval error on {mic_path.name}: {e}")
        return {"wcsr": 0.0, "root_acc": 0.0, "symbol_acc": 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WCSR evaluation for CRNN chord recogniser")
    parser.add_argument("--guitarset-dir",   default="data/guitarset")
    parser.add_argument("--checkpoint",      default="guitar_only/checkpoints/chord_recogniser_guitar.pt")
    parser.add_argument("--output-csv",      default="evaluation/wcsr_results.csv")
    parser.add_argument("--confidence-gate", type=float, default=0.35)
    parser.add_argument("--median-kernel",   type=int,   default=9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")
    model, vocab = load_model(str(ckpt), device)
    print(f"Loaded: {ckpt.name}  ({len(vocab)} classes in vocabulary)")

    mic_dir = Path(args.guitarset_dir) / "audio_mono-mic"
    ann_dir = Path(args.guitarset_dir) / "annotation"

    test_files = []
    for mic_path in sorted(mic_dir.glob("05_*_mic.wav")):
        stem = mic_path.stem
        for suffix in ("_mic", "_pickup_mix", "_pickup"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        jams_path = ann_dir / f"{stem}.jams"
        if jams_path.exists():
            test_files.append((mic_path, jams_path))

    if not test_files:
        sys.exit(f"No Player 05 test files found in {mic_dir}")
    print(f"Test clips (Player 05): {len(test_files)}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    all_wcsr, all_root, all_sym = [], [], []

    for mic_path, jams_path in test_files:
        clip_id = mic_path.stem.replace("_mic", "")
        print(f"  {clip_id} ... ", end="", flush=True)
        r = evaluate_clip(mic_path, jams_path, model, vocab, device,
                          gate=args.confidence_gate, median_k=args.median_kernel)
        print(f"WCSR={r['wcsr']:.3f}  root={r['root_acc']:.3f}  sym={r['symbol_acc']:.3f}")
        results.append({"clip_id": clip_id, **r})
        all_wcsr.append(r["wcsr"])
        all_root.append(r["root_acc"])
        all_sym.append(r["symbol_acc"])

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "wcsr", "root_acc", "symbol_acc"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"WCSR EVALUATION SUMMARY  (GuitarSet Player 05, n={len(results)})")
    print(f"  Mean WCSR (majmin):    {np.mean(all_wcsr)*100:.1f}% ± {np.std(all_wcsr)*100:.1f}%")
    print(f"  Mean root accuracy:   {np.mean(all_root)*100:.1f}% ± {np.std(all_root)*100:.1f}%")
    print(f"  Mean symbol accuracy: {np.mean(all_sym)*100:.1f}% ± {np.std(all_sym)*100:.1f}%")
    print(f"  Checkpoint: {ckpt.name}")
    print(f"  Results saved to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
