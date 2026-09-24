"""
aircraft_performance -- teaching-oriented aircraft performance library.

Methods follow J. Roskam & C.T. Lan, *Airplane Aerodynamics and Performance*
(DARcorporation, 1997), cross-checked with J.D. Anderson, *Aircraft
Performance and Design* (McGraw-Hill, 1999). Both are in ``References/``.

All quantities are in SI units (m, s, kg, N, W, Pa, K); use
:mod:`aircraft_performance.units` to convert data given in other units.

Modules
-------
units            conversion factors (ft, lb, hp, kt, SFC ...)
atmosphere       International Standard Atmosphere, airspeed conversions
aerodynamics     parabolic drag polar, characteristic points, stall speed, ...
propulsion       propeller and jet powerplant models (lapse with altitude, SFC)
aircraft         the Aircraft data container
examples         ready-to-use example aircraft
level_flight     thrust/power required, minimum and maximum speeds, envelope
climb            rate of climb, climb angle, ceilings, time to climb
descent          gliding flight, powered descent, drift-down
range_endurance  Breguet equations, numerical cruise, loiter, payload-range
takeoff_landing  take-off and landing distances
maneuvering      turns, load factors, V-n diagram
errors           exception and warning classes
plotting         notebook plot style (imported explicitly, needs matplotlib)
reporting        plain-text result tables
"""

from . import (aerodynamics, aircraft, atmosphere, climb, descent, errors, examples,
               level_flight, maneuvering, propulsion, range_endurance, takeoff_landing, units)
from .aerodynamics import ParabolicDragPolar
from .aircraft import Aircraft, Configuration
from .errors import (ConvergenceError, InfeasibleFlightConditionError, InputError,
                     PerformanceError, PerformanceWarning)
from .propulsion import JetPropulsion, PropellerPropulsion

__version__ = "1.0.0"

__all__ = [
    "aerodynamics", "aircraft", "atmosphere", "climb", "descent", "errors", "examples",
    "level_flight", "maneuvering", "propulsion", "range_endurance", "takeoff_landing", "units",
    "ParabolicDragPolar", "Aircraft", "Configuration", "JetPropulsion", "PropellerPropulsion",
    "PerformanceError", "InputError", "InfeasibleFlightConditionError", "ConvergenceError",
    "PerformanceWarning", "__version__",
]
