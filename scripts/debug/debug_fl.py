from src.pipeline.pipeline import GuitarChordPipeline

p = GuitarChordPipeline()
r = p.process('samples/Nutshell_guitar.wav')

fl = r.frame_labels
print(f'Total frames: {len(fl)}')

# Show unique labels in the first 300 frames (bar 1)
prev = None
for i, label in enumerate(fl[:300]):
    if label != prev:
        print(f'  frame[{i:4d}] (t={i*512/44100:.3f}s) = {label}')
        prev = label

print()
print(f'First 10 beat times: {[f"{t:.3f}" for t in r.analysis.beat_times[:10]]}')
