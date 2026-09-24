"""
Unit conversion factors and helpers.

**Every function of this package works in SI units** (m, s, kg, N, W, Pa, K).
The references (Roskam & Lan, Anderson) mostly use British units and some
"inconsistent" engineering units such as lb/(hp h) for the specific fuel
consumption. Converting once, at the boundary, with the named constants below
is the safest way to avoid the classic factor-of-3600 or factor-of-g mistakes.

Usage example::

    from aircraft_performance import units as u
    S = 175.0 * u.FT2            # wing area in m^2
    W = 4600.0 * u.LBF           # weight in N
    c = u.bsfc_to_si(0.45)       # 0.45 lb/(hp h)  ->  1/m

Multiply by a constant to go *to* SI, divide by it to go *back*.
"""

from __future__ import annotations

from .validation import ensure_non_negative

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------
G0 = 9.80665                 #: standard gravitational acceleration [m/s^2]

# ---------------------------------------------------------------------------
# Length, area, speed
# ---------------------------------------------------------------------------
FT = 0.3048                  #: 1 ft in m
FT2 = FT ** 2                #: 1 ft^2 in m^2
NM = 1852.0                  #: 1 nautical mile in m
MI = 1609.344                #: 1 statute mile in m
KM = 1000.0                  #: 1 km in m
KT = NM / 3600.0             #: 1 knot in m/s
KMH = 1000.0 / 3600.0        #: 1 km/h in m/s
FPM = FT / 60.0              #: 1 ft/min in m/s

# ---------------------------------------------------------------------------
# Mass, force, power, time
# ---------------------------------------------------------------------------
LB = 0.45359237              #: 1 lb (mass) in kg
LBF = LB * G0                #: 1 lbf in N
SLUG = LBF / FT              #: 1 slug in kg
HP = 550.0 * FT * LBF        #: 1 horsepower (550 ft lbf/s) in W  (= 745.7 W)
KW = 1000.0                  #: 1 kW in W
HOUR = 3600.0                #: 1 hour in s
MINUTE = 60.0                #: 1 minute in s
PSF = LBF / FT2              #: 1 lbf/ft^2 in Pa
SLUG_FT3 = SLUG / FT ** 3    #: 1 slug/ft^3 in kg/m^3

# ---------------------------------------------------------------------------
# Specific fuel consumption
# ---------------------------------------------------------------------------
# Propeller aircraft (piston / turboprop): the engine is characterised by the
# *brake* (shaft) specific fuel consumption, fuel WEIGHT per unit shaft ENERGY.
#     c [N / (W s)] = c [N / J] = c [1/m]
# Jet aircraft: thrust specific fuel consumption, fuel WEIGHT per unit THRUST
# per unit time.
#     c_t [N / (N s)] = c_t [1/s]
# Both definitions use the fuel *weight* flow, as in Roskam Eqns (11.1), (11.47).


def bsfc_to_si(c_lb_per_hp_h: float) -> float:
    """Convert a brake SFC from lb/(hp h) to the SI value in 1/m (= N/J)."""
    c = ensure_non_negative("c_lb_per_hp_h", c_lb_per_hp_h)
    return c * LBF / (HP * HOUR)


def bsfc_from_si(c_per_m: float) -> float:
    """Convert a brake SFC from 1/m back to lb/(hp h)."""
    c = ensure_non_negative("c_per_m", c_per_m)
    return c * HP * HOUR / LBF


def bsfc_metric_to_si(c_kg_per_kw_h: float) -> float:
    """Convert a brake SFC from kg/(kW h) (fuel mass) to the SI weight-based value in 1/m."""
    c = ensure_non_negative("c_kg_per_kw_h", c_kg_per_kw_h)
    return c * G0 / (KW * HOUR)


def tsfc_to_si(c_per_hour: float) -> float:
    """Convert a thrust SFC from lb/(lbf h) (i.e. 1/h) to 1/s."""
    c = ensure_non_negative("c_per_hour", c_per_hour)
    return c / HOUR


def tsfc_from_si(c_per_s: float) -> float:
    """Convert a thrust SFC from 1/s back to 1/h (lb/(lbf h))."""
    c = ensure_non_negative("c_per_s", c_per_s)
    return c * HOUR


def tsfc_metric_to_si(c_kg_per_N_h: float) -> float:
    """Convert a thrust SFC from kg/(N h) (fuel mass per newton per hour) to 1/s."""
    c = ensure_non_negative("c_kg_per_N_h", c_kg_per_N_h)
    return c * G0 / HOUR


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
def celsius_to_kelvin(t_c: float) -> float:
    """Convert a temperature from degrees Celsius to kelvin."""
    return t_c + 273.15


def kelvin_to_celsius(t_k: float) -> float:
    """Convert a temperature from kelvin to degrees Celsius."""
    return t_k - 273.15
