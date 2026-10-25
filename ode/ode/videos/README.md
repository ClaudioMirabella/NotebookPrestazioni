# MATH 331 videos (Manim Community)

Three commentary videos (at most 10 minutes each) on the models of the notebooks. They follow
the design of the `waves/` series: shared toolkit, commentary written in the code and exported as
timed subtitles (`.srt`), sections that can be rendered one at a time.

| Episode | File | Content | Length |
|---|---|---|---|
| 1 | `ep01_growth_decay.py` | P' = kP: model, slope field, uniqueness, separation of variables, Lebl Example 0.2.1, decay, half-life, radiocarbon dating, the sign of k | ~5.3 min |
| 2 | `ep02_logistic.py` | x' = kx(M - x): per-capita rate, phase line, stability, S-curve and inflection, exact solution and blow-up, harvesting and saddle-node bifurcation | ~4.7 min |
| 3 | `ep03_euler.py` | Euler's method on Lebl's y' = y^2/3, Table 1.7.1, first order (Taylor), Heun and RK4, instability of y' = -5y, absolute stability | ~4.4 min |

* `odekit.py`: palette (each colour has one meaning: blue = exact solution, amber = numerical
  approximation, green = stable, pink = unstable, violet = parameter), `OdeScene` base class with
  `narrate`, `title_card`, `section_title`, `key_idea`, `clear_screen`, and the drawing helpers
  `slope_field` and `solution_curve`.
* `render.py`: renders whole episodes into `output/` (video + subtitles).
* `tests/`: checks that need no rendering (`python -m pytest`).

## Rendering

```bash
manim -pql ep02_logistic.py Ep02Harvesting   # one section, quick 480p preview
python render.py ep01                        # one episode, 1080p60
python render.py all -q l                    # all episodes, quick previews
```

Parallel renders must use separate media folders (`render.py` does this): two Manim processes
compiling the same formula in a shared cache can collide.

## Setup

Use `../environment.yml` (conda-forge Manim, no compiler needed). `manim checkhealth` must report
`latex` and `dvisvgm` as PASSED (system TeX Live). For the Manim basics, see `waves/primer.py`
and `waves/README.md`.
