from src.pipeline.pipeline import GuitarChordPipeline

p = GuitarChordPipeline()
r = p.process('samples/Nutshell_guitar.wav')

fl = r.frame_labels
print('Frame labels (first 300 frames):')
prev = None
for i, label in enumerate(fl[:300]):
    if label != prev:
        print(f'  frame[{i:4d}] (t={i*512/44100:.3f}s) = {label}')
        prev = label

print()
print('Chord events (first 10):')
for ev in r.chord_events[:10]:
    print(f'  {ev.start_time:.3f}s - {ev.end_time:.3f}s : {ev.symbol}')
