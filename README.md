# NotebookPrestazioni

Jupyter notebooks and Python code for an aircraft performance course
(*Prestazioni del velivolo*).

| Folder | Content |
|---|---|
| `Capitolo1/` | International Standard Atmosphere and basic definitions (Italian) |
| `Capitolo8/` | Rate of climb and ceilings of a propeller aircraft (Italian) |
| `Capitolo9/` | Range and endurance, Breguet equations (Italian) |
| `PrestazioniRoskam/` | Complete performance calculations following Roskam & Lan (English) |
| `aircraft_performance/` | Python package used by the `PrestazioniRoskam` notebooks |
| `tests/` | Automated tests, including the worked examples of the references |
| `References/` | Roskam & Lan, *Airplane Aerodynamics and Performance*; Anderson, *Aircraft Performance and Design* |

## The `PrestazioniRoskam` notebooks

| Notebook | Topics | Roskam & Lan |
|---|---|---|
| `00_Overview` | conventions, SI units, error handling, example aircraft | - |
| `01_AerodynamicDatabase_LevelFlight` | parabolic drag polar, parasite-area estimate of CD0, configurations (flaps, gear), polar points E/P/A, stall speeds, thrust and power required, maximum/minimum level speed, flight envelope | Ch. 5, 8, 12.3 |
| `02_Range_Endurance_Loiter` | specific range and endurance, Breguet equations (propeller and jet), numerical cruise for three flight programs, best speeds, wind, loiter, payload-range diagram, fuel reserves | Ch. 11 |
| `03_RateOfClimb` | rate of climb and climb gradient, hodograph, jet and propeller optima, steep climbs, acceleration factor, ceilings, time/distance/fuel to climb, OEI second-segment gradient | Ch. 9 |
| `04_RateOfDescent` | gliding flight (glide angle, sink rate, hodograph, time aloft), speed to fly with wind, idle descent, OEI drift-down | Sec. 8.2, 9.2.3, 9.3.3, 9.4.2 |
| `05_Takeoff_Landing` | ground roll with ground effect, Roskam's analytical method, numerical integration, wind/slope/altitude/temperature effects, FAR 23/25 statistical methods, balanced field length, landing distance | Ch. 10 |
| `06_Maneuvering_FlightEnvelope` | level turns, instantaneous vs sustained load factor, turn radius and rate, FAR 23 V-n diagram | Ch. 12 |

The notebooks explain the theory (with the equation numbers of Roskam & Lan)
and call small, documented functions of the `aircraft_performance` package.
Four example aircraft are provided (`aircraft_performance.examples`): a light
single-engine piston aircraft (the one of `Capitolo9`), the light twin of
Roskam Examples 9.1/10.2/10.3, an ATR 72-like turboprop and the Gulfstream
IV-like business jet used by Anderson.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                     # ~100 tests, textbook examples included
jupyter notebook PrestazioniRoskam/  # open the notebooks
```

```python
import aircraft_performance as ap
from aircraft_performance import units as u, climb, range_endurance as rng

jet = ap.examples.business_jet_giv()
best = climb.max_rate_of_climb(jet, altitude=3000.0)
print(f"RC_max = {best.rate:.1f} m/s at {best.V / u.KT:.0f} kt")
```

## Design principles

* **SI units everywhere**; conversions happen once, with named constants
  (`u.FT`, `u.LBF`, `u.HP`, `u.KT`, `u.bsfc_to_si(...)`, `u.tsfc_to_si(...)`).
* **No silent failures**: invalid inputs raise `InputError`; physically
  impossible requests (flight above the ceiling, a take-off that never reaches
  lift-off speed...) raise `InfeasibleFlightConditionError`; results outside
  the validity of the model (e.g. Mach number above drag divergence) emit a
  `PerformanceWarning`. No function returns `NaN`.
* **Readability first**: one function per equation or method, docstrings citing
  the reference equation, simple numerical methods (grid search + Brent,
  trapezoidal integration, `solve_ivp`).
* **Verified**: the test suite reproduces, among others, Anderson's Gulfstream
  IV examples (minimum thrust, climb-rate table from sea level to 60,000 ft,
  glide, endurance) and Roskam Examples 9.2, 10.2, 10.3, 11.1-11.3.
  The continuous-integration workflow runs the tests and executes every
  notebook (`python tools/run_notebooks.py`).

## References

* J. Roskam, C.T. Lan, *Airplane Aerodynamics and Performance*, DARcorporation, 1997.
* J.D. Anderson Jr., *Aircraft Performance and Design*, McGraw-Hill, 1999.
