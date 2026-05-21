#!/usr/bin/env python3
"""
SDR evaluation for source separation (Demucs htdemucs vs HPSS fallback).

LIMITATION:
-----------
GuitarSet recordings are solo guitar with no associated band mix, so they
cannot be used for mix-vs-reference SDR evaluation.

This script evaluates on the pilot songs in samples/:
  Creep.wav, Nutshell.wav, Pearl Jam - Black (Official Audio).wav

The *_guitar.wav files (e.g. Creep_guitar.wav) serve as pseudo-reference
guitar stems. These were produced by a prior Demucs htdemucs run, NOT by
manual multi-track separation. Consequently, Demucs SDR reported here
measures self-consistency (how similar a fresh Demucs run is to its own
prior output), not absolute separation quality relative to a ground-truth
stem. HPSS SDR is measured against the same Demucs pseudo-reference and
reflects the quality gap between HPSS and Demucs, not absolute HPSS quality.

Expected approximate values:
  Demucs htdemucs : ~20+ dB  (self-consistency — nearly deterministic)
  HPSS fallback   : ~3–8 dB  (depends on mix complexity)

Run from project root:
    python evaluation/evaluate_sdr.py

Optional arguments:
    --mixes-dir  PATH   directory to scan for mix+guitar pairs  (default: .)
    --output-csv PATH   default: evaluation/sdr_results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mir_eval")

import numpy as np
import torch
import librosa
import soundfile as sf

SR = 44100


# ---------------------------------------------------------------------------
# Demucs separation
# ---------------------------------------------------------------------------
def load_demucs_model(device: str):
    import demucs.pretrained
    model = demucs.pretrained.get_model("htdemucs")
    model.eval()
    model.to(device)
    return model


def separate_demucs(mix_path: Path, model, device: str) -> np.ndarray:
    """
    Run Demucs htdemucs on a mix file. Returns the 'other' stem as a
    mono float32 array resampled to SR=44100.

    Uses soundfile for loading to avoid torchaudio/FFmpeg issues on Windows
    (see HANDOFF.md §2.1 for the FFmpeg incompatibility that caused this fix).
    """
    import demucs.apply
    from demucs.audio import convert_audio

    audio_data, orig_sr = sf.read(str(mix_path), dtype="float32", always_2d=True)
    audio_tensor = torch.from_numpy(audio_data.T).float()  # (channels, samples)
    audio_tensor = convert_audio(audio_tensor, orig_sr, model.samplerate, model.audio_channels)
    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # (1, channels, samples)

    with torch.no_grad():
        sources = demucs.apply.apply_model(model, audio_tensor, device=device)[0]

    other_idx = list(model.sources).index("other")
    other_stem = sources[other_idx].mean(dim=0).cpu().numpy()  # mono

    if model.samplerate != SR:
        other_stem = librosa.resample(other_stem, orig_sr=model.samplerate, target_sr=SR)

    return other_stem.astype(np.float32)


# ---------------------------------------------------------------------------
# HPSS separation
# ---------------------------------------------------------------------------
def separate_hpss(mix_path: Path, margin: float = 3.0) -> np.ndarray:
    """HPSS harmonic component at SR=44100. Returns mono float32 array."""
    audio, _ = librosa.load(str(mix_path), sr=SR, mono=True)
    D = librosa.stft(audio)
    H, _ = librosa.decompose.hpss(np.abs(D), margin=margin)
    D_harmonic = D * (H / (np.abs(D) + 1e-8))
    harmonic = librosa.istft(D_harmonic, length=len(audio))
    return harmonic.astype(np.float32)


# ---------------------------------------------------------------------------
# SDR computation
# ---------------------------------------------------------------------------
def compute_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Compute SDR in dB using mir_eval.separation.bss_eval_sources."""
    import mir_eval.separation

    min_len = min(len(reference), len(estimated))
    ref = reference[:min_len][np.newaxis, :]  # (1, N)
    est = estimated[:min_len][np.newaxis, :]  # (1, N)

    # bss_eval_sources is deprecated in mir_eval 0.8 but still functional; pin mir_eval<0.9
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(ref, est)
    return float(sdr[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SDR evaluation for source separation")
    parser.add_argument("--mixes-dir",  default="samples")
    parser.add_argument("--output-csv", default="evaluation/sdr_results.csv")
    args = parser.parse_args()

    mixes_dir = Path(args.mixes_dir)

    # Discover mix/reference pairs: {stem}.wav + {stem}_guitar.wav
    pairs = []
    for mix_path in sorted(mixes_dir.glob("*.wav")):
        if "_guitar" in mix_path.stem:
            continue
        ref_path = mixes_dir / f"{mix_path.stem}_guitar.wav"
        if ref_path.exists():
            pairs.append((mix_path, ref_path))

    if not pairs:
        sys.exit(
            f"No mix/guitar pairs found in {mixes_dir}.\n"
            "Expected: <name>.wav and <name>_guitar.wav in the same directory."
        )
    print(f"Found {len(pairs)} mix/reference pair(s): {[p[0].name for p in pairs]}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    demucs_sdrs: list[float] = []
    hpss_sdrs:   list[float] = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    try:
        demucs_model = load_demucs_model(device)
        print("Demucs htdemucs model loaded.")
    except Exception as e:
        print(f"WARNING: Could not load Demucs model: {e}")
        demucs_model = None

    for mix_path, ref_path in pairs:
        clip_id = mix_path.stem
        print(f"\n{'-'*50}")
        print(f"Clip: {clip_id}")

        ref_audio, _ = librosa.load(str(ref_path), sr=SR, mono=True)

        sdr_demucs = float("nan")
        if demucs_model is not None:
            try:
                print("  Demucs htdemucs ... ", end="", flush=True)
                est = separate_demucs(mix_path, demucs_model, device)
                sdr_demucs = compute_sdr(ref_audio, est)
                demucs_sdrs.append(sdr_demucs)
                print(f"SDR = {sdr_demucs:.2f} dB")
            except Exception as e:
                print(f"FAILED: {e}")
        else:
            print("  Demucs htdemucs ... SKIPPED (model not loaded)")

        sdr_hpss = float("nan")
        try:
            print("  HPSS (margin=3.0) ... ", end="", flush=True)
            est = separate_hpss(mix_path)
            sdr_hpss = compute_sdr(ref_audio, est)
            hpss_sdrs.append(sdr_hpss)
            print(f"SDR = {sdr_hpss:.2f} dB")
        except Exception as e:
            print(f"FAILED: {e}")

        results.append({"clip_id": clip_id, "sdr_demucs": sdr_demucs, "sdr_hpss": sdr_hpss})

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "sdr_demucs", "sdr_hpss"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("SDR EVALUATION SUMMARY")
    print()
    print("NOTE: *_guitar.wav references are Demucs pseudo-references, not true")
    print("ground-truth stems. Demucs SDR measures self-consistency; HPSS SDR")
    print("reflects quality relative to Demucs output. See script header.")
    print()
    if demucs_sdrs:
        print(f"  Demucs htdemucs:  mean {np.nanmean(demucs_sdrs):.2f} dB +/- {np.nanstd(demucs_sdrs):.2f}")
    if hpss_sdrs:
        print(f"  HPSS fallback:    mean {np.nanmean(hpss_sdrs):.2f} dB +/- {np.nanstd(hpss_sdrs):.2f}")
    print(f"  Clips evaluated:  {', '.join(r['clip_id'] for r in results)}")
    print(f"  Results saved to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
