# MATH 331 notebooks

Open `00_Index.ipynb` first. Each notebook: problem statement, existence and uniqueness, analytical
solution (verified with SymPy), reproduction of reference examples (with `assert`), numerical
solutions, sliders (ipywidgets), interpretation, exercises.

* `odelib.py`: tested helper functions (analytic solutions, Euler/Heun/RK4, slope fields, phase lines,
  characteristic roots, free response of the mass-spring-damper, Wronskians).
* `tests/`: `python -m pytest` (from this folder) checks `odelib` against the reference values.
* Sliders need a running kernel (Jupyter, JupyterLab or VS Code). Every slider plot also has a static
  version, so the notebooks can be read on GitHub.
* Notebooks 05-07 (second-order linear equations) should be read in order.
* To re-execute all notebooks: `python ../../tools/run_notebooks.py --inplace *.ipynb`.
