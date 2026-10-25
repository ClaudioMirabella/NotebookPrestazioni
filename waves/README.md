# The Mathematics of Waves: an animated series made with Manim

A series of commentary videos (≤ 10 minutes each), animated with
[Manim Community](https://docs.manim.community) and following
D. Fleisch & L. Kinnaman, *A Student's Guide to Waves* (Cambridge University Press, 2015).
Informal tone, but thorough.

| # | Episode | Book | Status |
|---|---|---|---|
| 1 | Vectors, complex numbers and wave functions | Ch. 1 | **done**: `ep01_vectors_complex_wavefunctions.py` |
| 2 | Phasors, describing waves with phasors, partial derivatives | Ch. 1-2 | planned |
| 3 | The classical wave equation (a partial differential equation) | Ch. 2 | planned |
| 4 | General solutions, boundary conditions; heat and diffusion equations | Ch. 2-3 | planned |
| 5 | Fourier analysis, wave packets, dispersion | Ch. 3 | planned |
| 6 | Mechanical waves: strings, pressure waves, ocean waves | Ch. 4 | planned |
| 7 | Electromagnetic waves, time-varying fields | Ch. 5 | planned |
| 8 | Quantum waves and examples | Ch. 6 | planned |

## Files

| File | What it is |
|---|---|
| `primer.py` | **Start here.** Five tiny scenes, one Manim idea each, commented line by line |
| `wavekit.py` | Shared toolkit: colour palette, font sizes, `WaveScene` base class (`narrate`, `title_card`, ...) |
| `ep01_vectors_complex_wavefunctions.py` | Episode 1, about 9 minutes |
| `render.py` | Renders an episode and copies the video and the subtitles to `videos/` |
| `tests/` | Checks that need no rendering (`pytest`) |
| `environment.yml`, `requirements.txt` | Environment (conda, recommended / pip) |
| `pyproject.toml` | Settings of the linter (ruff) and of pytest |

## Learning path

1. Read and render the primer, one scene at a time:
   ```bash
   manim -pql primer.py P1_MobjectsAndAnimations
   ```
   * **P1**: mobjects vs animations, the `play`/`wait` timeline, positioning;
   * **P2**: formulas with `MathTex`, colouring parts, `TransformMatchingTex`;
   * **P3**: `Axes`, `plot`, and `c2p` (math coordinates to screen);
   * **P4**: `ValueTracker` + `always_redraw` + updaters. This is how every wave moves;
   * **P5**: `.animate`, rate functions, `LaggedStart`, `AnimationGroup`.
2. Read `wavekit.py`, a small base class that every episode reuses.
3. Read episode 1 section by section. Each section is a method of `Episode01` and can be
   rendered on its own, which is much faster while editing:
   ```bash
   manim -pql ep01_vectors_complex_wavefunctions.py Ep01Euler
   ```
   Sections: `Ep01Intro`, `Ep01WhatIsAWave`, `Ep01Vectors`, `Ep01ComplexNumbers`,
   `Ep01Euler`, `Ep01WaveFunctions`, `Ep01Recap`.

## Rendering

```bash
python render.py ep01                 # whole episode, 1080p60 (a few minutes per minute of video)
python render.py ep01 -q l            # quick 480p15 preview of the whole episode
```

`-ql` (480p15) is about 15 times faster than `-qh` (1080p60): use it while developing.
Manim caches every animation (`media/`), so after a small edit only the changed
animations are re-rendered.

All commands are run from inside this folder.

## Setup (Anaconda + TeX Live)

```bash
conda env create -f environment.yml
conda activate waves
manim checkhealth            # latex and dvisvgm must be PASSED
pytest                       # quick checks, no rendering
ruff check .                 # code style and likely bugs
```

conda-forge provides Manim with all its compiled parts (ManimPango, Cairo, Pango, FFmpeg).
Use only the conda-forge channel, as `environment.yml` does: mixing in Anaconda's `defaults`
Cairo causes `ImportError ... symbol not found` (fix: `conda install -c conda-forge pycairo`).
Formulas (`MathTex`) use your system TeX Live. If one fails to compile, install the
LaTeX packages Manim needs:

```bash
tlmgr install amsmath babel-english cbfonts-fd cm-super count1to doublestroke dvisvgm \
  everysel fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin mathastext \
  microtype multitoc physics preview prelim2e ragged2e relsize rsfs setspace standalone \
  tipa wasy wasysym xcolor xetex xkeyval
```

In VS Code, select the `waves` interpreter (*Python: Select Interpreter*).

**Manim Community vs ManimGL.** github.com/3b1b/manim is *ManimGL* (`from manimlib import *`),
a different library from Manim Community (`from manim import *`, docs.manim.community),
which is the one used here. Code written for one does not run with the other.

## How the commentary works

Each line of the script is written in the code with `self.narrate("...", *animations)`:

* the line goes into the subtitle file `Episode01.srt`, which Manim writes next to the video;
* the scene keeps going for the time needed to say the line aloud (2.5 words per second,
  `WORDS_PER_SECOND` in `wavekit.py`), playing the given animations meanwhile.

So the video's pacing follows the script. You can:

* publish the video with the `.srt` subtitles;
* record a voice-over by reading the `.srt` (each entry has its start time); or
* re-time everything by changing `WORDS_PER_SECOND` (e.g. 2.2 for a slower pace).

## Coding conventions (and why)

* **One base class, one palette.** Every scene derives from `WaveScene`, and colours have
  *meanings* (amber = real part / x component, pink = imaginary part / y component,
  light blue = waves). Viewers learn to read them across episodes.
* **Sections as methods.** `Episode01.construct` calls one method per section. Thin
  subclasses (`Ep01Euler`, ...) render a single section without duplicating code.
* **Continuous change = `ValueTracker` + `always_redraw`.** Geometry is computed from the
  tracker at every frame, so labels and shapes can never disagree.
* **Physics first.** Drawings follow the formulas exactly. For example, the phasor's real
  part has the same length as the wave's displacement because both use the same scale
  (`axes.c2p`).
* **Named constants, no magic numbers** for the parameters of the example wave
  (`WAVELENGTH`, `PERIOD`, ...).
* **Closures in loops.** Updaters created in a loop get their own copy of the loop
  variable through a small factory function (see `particle_updater` in section 1). This
  avoids a classic Python bug.
* **Linting.** The code passes `ruff check .` (rules in `pyproject.toml`).
