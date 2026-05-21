#!/usr/bin/env python3
"""
TabCNN held-out test-set evaluation (GuitarSet Player 05).

Split note:
-----------
The TabCNN training script uses an 80/20 sorted-file split with no
explicit random seed (deterministic by alphabetical order):
  - Train : first 80% of sorted JAMS files (approx. Players 00-04)
  - Val   : last 20% (approx. last 12 of Player 04 + all 60 of Player 05)

Player 05 files appear at positions 300-359 in the sorted list (out of
360 total), entirely within the val region. They were never used for
gradient updates. This script evaluates only on Player 05 to give an
unbiased test-set estimate, then compares against the validation metrics
reported during training (93.1% per-string, 70.5% frame accuracy).

Audio note:
-----------
GuitarSetTabDataset expects audio_mono-pickup_mix/ files, which are not
available locally. This script uses audio_mono-mic/ files instead (same
recordings, different capture method). The CQT parameters are identical
to the training dataset.

Run from project root:
    python evaluation/evaluate_tabcnn_test.py

Optional arguments:
    --guitarset-dir  PATH  default: data/guitarset
    --checkpoint     PATH  default: guitar_only/checkpoints/tab_cnn_best.pt
    --output-csv     PATH  default: evaluation/tabcnn_test_results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import numpy as np
import torch

from guitar_only.models.tab_cnn import TabCNN


# ---------------------------------------------------------------------------
# Constants (must match GuitarSetTabDataset exactly)
# ---------------------------------------------------------------------------
N_BINS         = 192
HOP_LENGTH     = 512
SR             = 44100
CONTEXT_FRAMES = 9
N_FRETS        = 20
FMIN_HZ        = 82.407   # librosa.midi_to_hz(40) — open low-E string

_OPEN_MIDI = [40, 45, 50, 55, 59, 64]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device) -> TabCNN:
    """Load TabCNN from a weights-only state dict."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TabCNN(n_bins=N_BINS, context_frames=CONTEXT_FRAMES, n_frets=N_FRETS)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def compute_cqt(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    192-bin CQT at 24 bins/octave, log-amplitude, normalised to [0,1].
    Matches GuitarSetTabDataset._compute_cqt exactly.
    """
    import librosa
    C = np.abs(librosa.cqt(
        audio, sr=sr, hop_length=HOP_LENGTH,
        fmin=FMIN_HZ, n_bins=N_BINS, bins_per_octave=24,
    ))
    C = librosa.amplitude_to_db(C, ref=np.max)
    C_min, C_max = C.min(), C.max()
    return ((C - C_min) / (C_max - C_min + 1e-8)).astype(np.float32)


def make_context_windows(C: np.ndarray) -> np.ndarray:
    """
    Slide a 9-frame context window over CQT with edge-replication padding.
    Returns (N_frames, 1, N_BINS, CONTEXT_FRAMES).
    """
    half = CONTEXT_FRAMES // 2
    C_pad = np.pad(C, ((0, 0), (half, half)), mode="edge")
    n_frames = C.shape[1]
    windows = np.stack(
        [C_pad[:, f: f + CONTEXT_FRAMES] for f in range(n_frames)], axis=0
    )
    return windows[:, np.newaxis, :, :]  # (N, 1, 192, 9)


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------
def load_tab_labels(jams_path: Path, n_frames: int) -> np.ndarray:
    """
    Load per-string per-frame fret labels from JAMS note_midi annotations.

    Returns np.ndarray (n_frames, 6) int32.
    Value -1 = string not active; 0..N_FRETS = fret number.
    """
    import jams
    labels = np.full((n_frames, 6), -1, dtype=np.int32)
    jam = jams.load(str(jams_path))
    note_anns = [a for a in jam.annotations if a.namespace == "note_midi"]
    for string_idx, ann in enumerate(note_anns[:6]):
        open_midi = _OPEN_MIDI[string_idx]
        for obs in ann.data:
            fret = int(round(float(obs.value) - open_midi))
            if not (0 <= fret <= N_FRETS):
                continue
            t_start = float(obs.time)
            t_end   = t_start + float(obs.duration)
            f0 = max(0, int(t_start * SR / HOP_LENGTH))
            f1 = min(n_frames - 1, int(t_end * SR / HOP_LENGTH))
            labels[f0: f1 + 1, string_idx] = fret
    return labels


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------
def evaluate_clip(
    mic_path: Path,
    jams_path: Path,
    model: TabCNN,
    device: torch.device,
) -> dict:
    """
    Evaluate one clip.

    Accuracy is computed over all frames (including silent/not-played frames),
    matching the training-time evaluate() function in train_tab_cnn.py.
    """
    import librosa
    audio, _ = librosa.load(str(mic_path), sr=SR, mono=True)
    C = compute_cqt(audio)
    windows = make_context_windows(C)  # (N, 1, 192, 9)
    n_frames = windows.shape[0]

    labels = load_tab_labels(jams_path, n_frames)           # (N, 6), raw frets
    labels_enc = np.where(labels < 0, 0, labels + 1)        # encode: -1->0, k->k+1

    # Batch inference to avoid OOM on CPU
    preds_list = []
    batch_size = 512
    for start in range(0, n_frames, batch_size):
        batch = torch.from_numpy(windows[start: start + batch_size]).float().to(device)
        with torch.no_grad():
            logits = model(batch)                                       # list[6 x (B, C)]
        batch_preds = torch.stack([logit.argmax(-1) for logit in logits], dim=1)  # (B, 6)
        preds_list.append(batch_preds.cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)  # (N, 6)

    per_string = [float((preds[:, s] == labels_enc[:, s]).mean()) for s in range(6)]
    frame_acc  = float((preds == labels_enc).all(axis=1).mean())

    return {
        "str1": per_string[0], "str2": per_string[1], "str3": per_string[2],
        "str4": per_string[3], "str5": per_string[4], "str6": per_string[5],
        "per_string_mean": float(np.mean(per_string)),
        "frame_acc": frame_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TabCNN test-set evaluation on Player 05")
    parser.add_argument("--guitarset-dir", default="data/guitarset")
    parser.add_argument("--checkpoint",    default="guitar_only/checkpoints/tab_cnn_best.pt")
    parser.add_argument("--output-csv",    default="evaluation/tabcnn_test_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")
    model = load_model(str(ckpt), device)
    print(f"Loaded: {ckpt.name}")

    guitarset_dir = Path(args.guitarset_dir)
    mic_dir = guitarset_dir / "audio_mono-mic"
    ann_dir = guitarset_dir / "annotation"

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
    all_str_accs  = [[] for _ in range(6)]
    all_frame_acc = []

    for mic_path, jams_path in test_files:
        clip_id = mic_path.stem.replace("_mic", "")
        print(f"  {clip_id} ... ", end="", flush=True)
        r = evaluate_clip(mic_path, jams_path, model, device)
        print(f"str_mean={r['per_string_mean']*100:.1f}%  frame={r['frame_acc']*100:.1f}%")
        results.append({"clip_id": clip_id, **r})
        for s in range(6):
            all_str_accs[s].append(r[f"str{s+1}"])
        all_frame_acc.append(r["frame_acc"])

    fieldnames = ["clip_id", "str1", "str2", "str3", "str4", "str5", "str6",
                  "per_string_mean", "frame_acc"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    mean_str = [np.mean(all_str_accs[s]) for s in range(6)]
    mean_ps  = float(np.mean(mean_str))
    mean_fa  = float(np.mean(all_frame_acc))

    print("\n" + "=" * 60)
    print(f"TABCNN TEST-SET EVALUATION  (Player 05, n={len(results)} clips)")
    print()
    string_names = ["E2", "A2", "D3", "G3", "B3", "e4"]
    for s, name in enumerate(string_names):
        print(f"  String {s+1} ({name}): {mean_str[s]*100:.1f}%")
    print(f"  Mean per-string:  {mean_ps*100:.1f}%")
    print(f"  Frame accuracy:   {mean_fa*100:.1f}%")
    print()
    print(f"  {'Metric':<26} {'Val split':<15} {'Test split (Player 05)'}")
    print(f"  {'-'*56}")
    print(f"  {'Per-string accuracy':<26} {'93.1%':<15} {mean_ps*100:.1f}%")
    print(f"  {'Frame accuracy':<26} {'70.5%':<15} {mean_fa*100:.1f}%")
    print()
    if mean_ps < 0.85:
        print("  WARNING: Per-string accuracy is more than 8 pp below the val-split")
        print("  figure. Two possible causes: (1) overfitting to the val split during")
        print("  model selection; (2) audio domain shift — training used pickup-mix")
        print("  recordings, this evaluation uses mic recordings. Discuss in §5.6.")
    print(f"  Results saved to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
