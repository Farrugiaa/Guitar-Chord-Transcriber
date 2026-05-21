import librosa
import numpy as np
from src.pipeline.pipeline import GuitarChordPipeline
from src.features.tab_exporter import PitchTabExporter, _subdivide
from pathlib import Path

p = GuitarChordPipeline()
r = p.process('samples/Nutshell_guitar.wav')

ckpt = Path('guitar_only/checkpoints/tab_cnn_best.pt')
exp = PitchTabExporter(sample_rate=r.sample_rate, tuning=r.tuning, tab_cnn_checkpoint=str(ckpt))

audio = r.guitar_waveform.squeeze().numpy()
audio_isolated = exp._isolate_rhythm_guitar(audio, r.analysis.bpm)
fmin = librosa.midi_to_hz(exp._open_midi[0])
beat_times = r.analysis.beat_times
sub_times = _subdivide(beat_times, 2)
sub_frets_all = exp._tab_cnn_frets(audio_isolated, sub_times, fmin)

from src.theory.chord_builder import ChordBuilder
from src.theory.scales import ScaleAnalyser

sa = ScaleAnalyser()
try:
    scale_pcs = set(sa.get_scale_pitch_classes(r.analysis.key, r.analysis.scale_type))
except:
    scale_pcs = None

chord_builder = ChordBuilder()

# Full detect_chords reimplementation WITH the frame alignment fix
total_frames = int(np.ceil(len(audio_isolated) / exp.hop_length))
beat_labels = []
for bi in range(max(len(beat_times)-1, 1)):
    string_obs = [[] for _ in range(6)]
    for si in range(2):
        idx = bi * 2 + si
        if idx < len(sub_frets_all):
            for s_idx, fret in enumerate(sub_frets_all[idx]):
                if fret >= 0:
                    string_obs[s_idx].append(fret)
    pcs = []
    for s_idx in range(6):
        obs = string_obs[s_idx]
        if not obs:
            continue
        mode_fret = min(obs, key=lambda f: (-obs.count(f), f))
        pc = (exp._open_midi[s_idx] + mode_fret) % 12
        if pc not in pcs:
            pcs.append(pc)
    label = chord_builder.identify_chord_from_pitches(pcs, key_context=r.analysis.key, scale_pcs=scale_pcs) if pcs else 'N.C.'
    beat_labels.append(label)

# Frame expansion WITH the fix
frame_labels = []
# Prepend N.C. for frames before first beat
first_beat_frame = int(librosa.time_to_frames(float(beat_times[0]), sr=exp.sample_rate, hop_length=exp.hop_length))
frame_labels.extend(['N.C.'] * first_beat_frame)
print(f'Prepended {first_beat_frame} N.C. frames before beat 0')

for bi, label in enumerate(beat_labels[:12]):
    t_start = float(beat_times[bi])
    t_end = float(beat_times[bi+1]) if bi+1 < len(beat_times) else t_start + 60/68
    fs = int(librosa.time_to_frames(t_start, sr=exp.sample_rate, hop_length=exp.hop_length))
    fe = int(librosa.time_to_frames(t_end, sr=exp.sample_rate, hop_length=exp.hop_length))
    fe = min(fe, total_frames)
    n = max(0, fe - fs)
    print(f'  bi={bi} label={label:15s} list[{len(frame_labels)}:{len(frame_labels)+n}] audio[{fs}:{fe}]')
    frame_labels.extend([label] * n)

print()
print(f'tab_labels[0:40] = {list(set(frame_labels[0:40]))}')
print(f'tab_labels[37:45] = {frame_labels[37:45]}')
print(f'tab_labels[105:115] = {frame_labels[105:115]}')
print(f'tab_labels[170:180] = {frame_labels[170:180]}')
