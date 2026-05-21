from src.pipeline.pipeline import GuitarChordPipeline
from src.features.tab_exporter import _subdivide
import numpy as np

p = GuitarChordPipeline()
r = p.process('samples/Nutshell_guitar.wav')

bt = r.analysis.beat_times
sub_times = _subdivide(bt, 2)

print(f'beat_times[0] = {bt[0]:.8f}')
print(f'sub_times[0]  = {sub_times[0]:.8f}')
print(f'sub_times[1]  = {sub_times[1]:.8f}')
print()

# Show first chord event
ev = r.chord_events[0]
print(f'First chord_event: {ev.symbol} [{ev.start_time:.8f}, {ev.end_time:.8f})')
print()

# Check _assign_chords condition for first 3 sub_times
for i, t in enumerate(sub_times[:4]):
    label = "N.C."
    for ev2 in r.chord_events[:5]:
        if ev2.start_time <= t < ev2.end_time:
            label = ev2.symbol
            break
    print(f'  sub_times[{i}] = {t:.8f}  -> {label}')
