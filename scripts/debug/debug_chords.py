from src.pipeline.pipeline import GuitarChordPipeline

p = GuitarChordPipeline()
r = p.process('samples/Nutshell_guitar.wav')

print('First 20 chord events:')
for ev in r.chord_events[:20]:
    print(f'  {ev.start_time:.3f}s - {ev.end_time:.3f}s : {ev.symbol}')

bt = r.analysis.beat_times
print(f'BPM: {r.analysis.bpm:.1f}')
print(f'Beat duration: {60/r.analysis.bpm:.3f}s')
print('First 10 beat times:', [f'{t:.3f}' for t in bt[:10]])
