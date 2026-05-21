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
audio = exp._isolate_rhythm_guitar(audio, r.analysis.bpm)
fmin = librosa.midi_to_hz(exp._open_midi[0])
beat_times = r.analysis.beat_times
sub_times = _subdivide(beat_times, 2)
sub_frets_all = exp._tab_cnn_frets(audio, sub_times, fmin)

from src.theory.chord_builder import ChordBuilder
chord_builder = ChordBuilder()

beat_labels = []
for bi in range(min(len(beat_times)-1, 12)):
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
    label = chord_builder.identify_chord_from_pitches(pcs, key_context=r.analysis.key) if pcs else 'N.C.'
    beat_labels.append(label)

total_frames = int(np.ceil(len(audio) / exp.hop_length))
print("Frame expansion (list indices vs audio frames):")
running = 0
for bi, label in enumerate(beat_labels):
    t_start = float(beat_times[bi])
    t_end = float(beat_times[bi+1]) if bi+1 < len(beat_times) else t_start + 60/68
    fs = int(librosa.time_to_frames(t_start, sr=exp.sample_rate, hop_length=exp.hop_length))
    fe = int(librosa.time_to_frames(t_end, sr=exp.sample_rate, hop_length=exp.hop_length))
    fe = min(fe, total_frames)
    n = max(0, fe - fs)
    print("  bi=%d label=%-12s list[%d:%d] audio_frames[%d:%d] t=[%.3f,%.3f]" % (
        bi, label, running, running+n, fs, fe, t_start, t_end))
    running += n

first_beat_frame = int(librosa.time_to_frames(beat_times[0], sr=44100, hop_length=512))
print()
print("First beat at frame:", first_beat_frame, "(%.3fs)" % beat_times[0])
print("Frames before first beat (offset):", first_beat_frame)
print("These frames are NOT prepended -> all beat labels are shifted by %d list indices" % first_beat_frame)
