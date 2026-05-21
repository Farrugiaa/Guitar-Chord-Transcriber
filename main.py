"""
Guitar Chord Extractor - Main Entry Point

Usage:
    python main.py audio_file.wav
    python main.py audio_file.wav --separator-model checkpoints/separator_best.pt
    python main.py audio_file.wav --chord-model checkpoints/chord_finetuned_guitar.pt
"""

import argparse
import sys

from src.pipeline.pipeline import GuitarChordPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract guitar chords from audio files"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to input audio file (WAV, MP3, FLAC, etc.)",
    )
    parser.add_argument(
        "--separator-model",
        type=str,
        default=None,
        help="Path to trained separator model checkpoint",
    )
    parser.add_argument(
        "--chord-model",
        type=str,
        default=None,
        help="Path to trained chord recogniser checkpoint",
    )
    parser.add_argument(
        "--vocab-level",
        type=str,
        default="extended",
        choices=["basic", "sevenths", "extended"],
        help="Chord vocabulary level (default: extended)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (prints to stdout if not specified)",
    )

    args = parser.parse_args()

    # Build pipeline
    pipeline = GuitarChordPipeline.from_checkpoints(
        separator_path=args.separator_model,
        recogniser_path=args.chord_model,
        vocab_level=args.vocab_level,
        device=args.device,
    )

    # Process
    print(f"Processing: {args.audio_file}")
    print("=" * 50)

    result = pipeline.process(args.audio_file)

    # Always save the extracted guitar audio so it can be auditioned
    from pathlib import Path

    audio_stem = Path(args.audio_file).stem
    guitar_out = f"{audio_stem}_guitar.wav"
    result.save_guitar_audio(guitar_out)
    print(f"Extracted guitar audio saved to: {guitar_out}")
    print()

    output_text = result.summary()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Output saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
