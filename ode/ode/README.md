# MATH 331 - Differential Equations: course material

| Folder / file | Content |
|---|---|
| `MATH331_Lecture_Plan.pdf` | the lecture plan: 30 sessions, 15 units, projects, open textbook |
| `FullText_Diff_Eq_For_Engineers.pdf` | **course text**: J. Lebl, *Notes on Diffy Qs* (CC BY-SA, free for students) |
| other PDFs | reference library (numerical methods, modelling, analysis), used to cross-check the material |
| `notebooks/` | Jupyter notebooks: problem, analytical and numerical solutions, sliders |
| `videos/` | commentary videos made with Manim (code, subtitles; rendered videos in `videos/output/`) |
| `environment.yml`, `requirements.txt` | Python environment (conda recommended / pip) |

## First block: first-order models (units U1, U2, U4, preview of U14)

| Topic | Notebook | Video | Lebl |
|---|---|---|---|
| exponential growth | `notebooks/01_Exponential_Growth.ipynb` | episode 1 | 0.2, 1.1-1.3 |
| radioactive decay, decay chains | `notebooks/02_Radioactive_Decay.ipynb` | episode 1 | 1.3, 1.4 |
| logistic equation, harvesting | `notebooks/03_Logistic_Equation.ipynb` | episode 2 | 1.6 |
| Euler's method and beyond | `notebooks/04_Euler_Method.ipynb` | episode 3 | 1.7 |

Start from `notebooks/00_Index.ipynb`.

## Setup

```bash
conda env create -f environment.yml
conda activate ode
cd notebooks && python -m pytest        # 28 tests: odelib against the references
jupyter lab                              # or open the folder in VS Code (kernel: ode)
```

## Rigour and cross-checks

Every reference value used in the notebooks is reproduced by the code and checked with
`assert` statements: Lebl's Example 0.2.1, Figures 1.6.2 and 1.6.5-1.6.7, Tables 1.7.1-1.7.2 and
Exercises 1.7.4 and 1.7.9; the radiocarbon example of Adkins and Davidson (Springer, 2012); the
absolute-stability analysis of Meseguer (Wiley, 2020), Section 8.2. Two small slips of the course
text found this way are pointed out in notebook 04.
