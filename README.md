# Guitar Chord Extractor

A desktop application that takes a guitar audio file and outputs:
- Chord progression with section labels (Intro, Verse, Chorus…)
- BPM, key, and tuning detection
- Strumming pattern diagram
- Full-song ASCII guitar tab
- MIDI export for DAW verification

---

## Requirements

- Windows 10 or 11
- [Python 3.12](https://www.python.org/downloads/) — during installation, tick **"Add Python to PATH"**
- ~2 GB free disk space (dependencies + Demucs model)

No GPU required. The app runs on CPU automatically.

---

## Setup (first time only)

1. Download or clone this folder to your PC.
2. Double-click **`setup.bat`** and wait for it to finish (~5 minutes depending on internet speed).
3. The Demucs source-separation model (~300 MB) downloads automatically on the **first run**.

---

## Running the app

Double-click **`run.bat`**

Or from a terminal:
```
venv\Scripts\python app.py
```

---

## Using the app

1. Click **Open Audio File** and select a `.wav` or `.mp3` guitar recording.
2. Click **Analyse** and wait (~30–60 seconds for a 4-minute song).
3. The chord progression, key, BPM, and strumming pattern appear automatically.
4. Use the **Export** buttons in the sidebar to save:
   - **Chord sheet (.txt)** — plain-text chord progression
   - **Tab notation (.txt)** — full-song guitar tab
   - **MIDI (.mid)** — for import into Logic, GarageBand, Reaper, etc.

### Tip — pre-separated guitar stems
If you have already extracted the guitar track (e.g. using Demucs yourself), name
the file `<songname>_guitar.wav` and the app will skip the separation step,
saving ~30 seconds of processing time.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Re-install Python 3.12 and tick "Add Python to PATH" |
| App opens but crashes immediately | Re-run `setup.bat` to repair dependencies |
| First run very slow | Demucs model is downloading (~300 MB) — wait for it to finish |
| Black / empty chord output | Try a cleaner recording; heavy reverb or very low volume can confuse the detector |
