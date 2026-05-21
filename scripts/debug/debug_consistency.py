from src.pipeline.pipeline import GuitarChordPipeline
from src.pipeline.output_formatter import OutputFormatter

p = GuitarChordPipeline()
r = p.process('samples/Nutshell_guitar.wav')

fl = r.frame_labels

# Manually re-run frames_to_chord_events on r.frame_labels to see if it produces different events
fmt = OutputFormatter()
events = fmt.frames_to_chord_events(fl, hop_length=512, sample_rate=44100)

print('Events from r.frame_labels:')
for ev in events[:10]:
    print(f'  {ev.start_time:.3f}s - {ev.end_time:.3f}s : {ev.symbol}')

print()
print('Events from r.chord_events:')
for ev in r.chord_events[:10]:
    print(f'  {ev.start_time:.3f}s - {ev.end_time:.3f}s : {ev.symbol}')

# Check that frame_labels around the transition is correct
print()
print(f'r.frame_labels[35:42] = {fl[35:42]}')
print(f'r.frame_labels[106:112] = {fl[106:112]}')
print(f'r.frame_labels[171:177] = {fl[171:177]}')
print(f'r.frame_labels[237:243] = {fl[237:243]}')
