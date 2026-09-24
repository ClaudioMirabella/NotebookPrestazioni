"""
Steady, level flight: thrust and power required, minimum and maximum speeds.

Equations (Roskam & Lan, Section 8.4)
-------------------------------------
In steady level flight ``L = W`` and ``T = D``. With a parabolic polar::

    T_req = D = q S CD0 + K W^2 / (q S)          (Eqn 8.59)
    P_req = D V                                   (Eqn 8.69)

* The drag is minimum at point E of the polar: ``D_min = W / E_max``
  (Eqns 8.64-8.68), where zero-lift and induced drag are equal.
* The power required is minimum at point P (Eqns 8.72-8.74).
* ``V_min_drag / V_min_power = 3**0.25 = 1.32`` (Eqn 8.75).

The maximum level speed is the high-speed intersection of the available and
required curves. For a jet with speed-independent thrust and a parabolic
polar it has the closed form of Roskam Eqn (12.14) / Anderson Eqn (5.50)::

    V_max^2 = [ (T/W)(W/S) + (W/S) sqrt((T/W)^2 - 4 CD0 K) ] / (rho CD0)

For the general case (propeller, static-thrust cap, any lapse model) the
intersection is found numerically by :func:`level_flight_speeds`.
The minimum level speed is the larger of the stall speed and the low-speed
intersection of the available and required curves (Roskam Fig. 8.18).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import aerodynamics as aero
from ._numerics import maximize_on_interval, root_in_bracket
from .aerodynamics import ParabolicDragPolar
from .aircraft import Aircraft
from .atmosphere import isa
from .errors import ConvergenceError, InfeasibleFlightConditionError
from .validation import ensure_positive, ensure_result_finite

#: Upper bound [m/s] of the speed search interval (well beyond any subsonic aircraft).
V_SEARCH_MAX = 1200.0


# ---------------------------------------------------------------------------
# Required thrust and power (primitive formulas)
# ---------------------------------------------------------------------------
def thrust_required(weight, wing_area, rho, V, polar: ParabolicDragPolar):
    """Thrust required for steady level flight, T = D [N] (Roskam Eqn 8.59)."""
    return aero.drag(weight, wing_area, rho, V, polar)


def power_required(weight, wing_area, rho, V, polar: ParabolicDragPolar):
    """Power required for steady level flight, P = D V [W] (Roskam Eqn 8.69)."""
    return thrust_required(weight, wing_area, rho, V, polar) * np.asarray(V)


def minimum_drag(weight, polar: ParabolicDragPolar) -> float:
    """Minimum drag in level flight D_min = W / E_max [N] (Roskam Eqn 8.68)."""
    W = ensure_positive("weight", weight)
    return W / polar.max_lift_to_drag


def speed_minimum_drag(weight, wing_area, rho, polar: ParabolicDragPolar):
    """Speed for minimum drag (point E) [m/s] (Roskam Eqns 8.20, 8.66)."""
    return aero.speed_at_polar_point(weight, wing_area, rho, polar.point_E)


def speed_minimum_power(weight, wing_area, rho, polar: ParabolicDragPolar):
    """Speed for minimum power required (point P) [m/s] (Roskam Eqn 8.73)."""
    return aero.speed_at_polar_point(weight, wing_area, rho, polar.point_P)


def minimum_power_required(weight, wing_area, rho, polar: ParabolicDragPolar):
    """Minimum power required for level flight [W] (Roskam Eqn 8.74)."""
    V = speed_minimum_power(weight, wing_area, rho, polar)
    return power_required(weight, wing_area, rho, V, polar)


# ---------------------------------------------------------------------------
# Closed-form speeds for a jet with parabolic polar and constant thrust
# ---------------------------------------------------------------------------
def _jet_level_speed_roots(weight, wing_area, rho, polar: ParabolicDragPolar, thrust):
    """Both solutions V_min, V_max of T = D for a constant thrust (quadratic in V^2)."""
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    T = ensure_positive("thrust", thrust)
    TW = T / W
    WS = W / S
    disc = TW ** 2 - 4.0 * polar.CD0 * polar.K
    if disc < 0.0:
        raise InfeasibleFlightConditionError(
            f"T/W = {TW:.4f} is smaller than the minimum drag-to-weight ratio "
            f"1/E_max = {1.0 / polar.max_lift_to_drag:.4f}: level flight is impossible.")
    V2_max = (TW * WS + WS * np.sqrt(disc)) / (rho * polar.CD0)
    V2_min = (TW * WS - WS * np.sqrt(disc)) / (rho * polar.CD0)
    return float(np.sqrt(V2_min)), float(np.sqrt(V2_max))


def max_level_speed_jet_parabolic(weight, wing_area, rho, polar: ParabolicDragPolar, thrust) -> float:
    """Maximum level speed of a jet, closed form (Roskam Eqn 12.14, Anderson Eqn 5.50) [m/s]."""
    return _jet_level_speed_roots(weight, wing_area, rho, polar, thrust)[1]


def min_level_speed_jet_parabolic(weight, wing_area, rho, polar: ParabolicDragPolar, thrust) -> float:
    """Low-speed (thrust-limited) solution of T = D for a jet [m/s].

    The actual minimum speed is the larger of this value and the stall speed.
    """
    return _jet_level_speed_roots(weight, wing_area, rho, polar, thrust)[0]


# ---------------------------------------------------------------------------
# General numerical solution for an Aircraft object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelFlightSpeeds:
    """Characteristic speeds of steady level flight at one altitude and weight [m/s]."""

    altitude: float
    weight: float
    V_stall: float            #: 1-g stall speed
    V_min_propulsive: float   #: low-speed intersection T_av = D (may be below V_stall)
    V_min: float              #: actual minimum speed = max(V_stall, V_min_propulsive)
    V_max: float              #: maximum level speed
    V_min_drag: float         #: speed for minimum drag (point E)
    V_min_power: float        #: speed for minimum power (point P)
    min_speed_limited_by: str  #: "stall" or "thrust/power"

    @property
    def mach_max(self) -> float:
        """Mach number at the maximum level speed."""
        return self.V_max / isa(self.altitude).speed_of_sound


def excess_thrust(aircraft: Aircraft, V, altitude, weight=None, config: str = "clean",
                  throttle: float = 1.0, n_operative=None, delta_T: float = 0.0,
                  load_factor: float = 1.0):
    """Available thrust minus drag [N] at speed V (level flight or n-g manoeuvre)."""
    W = aircraft.weight_or_default(weight)
    rho = isa(altitude, delta_T).density
    polar = aircraft.polar(config)
    T = aircraft.propulsion.thrust_available(V, altitude, throttle, n_operative, delta_T)
    D = aero.drag(W, aircraft.wing_area, rho, V, polar, load_factor)
    return T - D


def level_flight_speeds(aircraft: Aircraft, altitude: float, weight: Optional[float] = None,
                        config: str = "clean", throttle: float = 1.0, n_operative=None,
                        delta_T: float = 0.0) -> LevelFlightSpeeds:
    """Minimum and maximum level-flight speeds by numerical intersection of T_av and D.

    Raises
    ------
    InfeasibleFlightConditionError
        If the available thrust is lower than the drag at every speed, i.e.
        the altitude is above the absolute ceiling for this weight/throttle.
    """
    W = aircraft.weight_or_default(weight)
    atm = isa(altitude, delta_T)
    conf = aircraft.config(config)
    S = aircraft.wing_area
    V_s = float(aero.stall_speed(W, S, atm.density, conf.CL_max))

    def f(V):
        return float(excess_thrust(aircraft, V, altitude, W, config, throttle, n_operative, delta_T))

    V_lo = 0.2 * V_s
    V_peak, f_peak = maximize_on_interval(f, V_lo, V_SEARCH_MAX, n_grid=400, log_spacing=True)
    if f_peak < 0.0:
        raise InfeasibleFlightConditionError(
            f"Level flight is impossible at h = {altitude:.0f} m, W = {W:.0f} N: the maximum "
            f"excess thrust is {f_peak:.1f} N < 0 (above the absolute ceiling).")
    if f(V_SEARCH_MAX) >= 0.0:
        raise ConvergenceError(f"No maximum speed found below {V_SEARCH_MAX} m/s; check the inputs.")
    V_max = root_in_bracket(f, V_peak, V_SEARCH_MAX, "maximum level speed")
    if f(V_lo) < 0.0:
        V_min_prop = root_in_bracket(f, V_lo, V_peak, "minimum level speed")
    else:
        V_min_prop = V_lo   # thrust exceeds drag even far below the stall speed
    V_min = max(V_s, V_min_prop)
    if V_min > V_max:
        raise InfeasibleFlightConditionError(
            f"At h = {altitude:.0f} m the stall speed ({V_s:.1f} m/s) exceeds the maximum "
            f"level speed ({V_max:.1f} m/s): no steady level flight is possible.")
    conf.polar.check_mach(V_max / atm.speed_of_sound, context="Maximum level speed: ")
    return LevelFlightSpeeds(
        altitude=float(altitude), weight=W, V_stall=V_s, V_min_propulsive=V_min_prop,
        V_min=V_min, V_max=V_max,
        V_min_drag=float(speed_minimum_drag(W, S, atm.density, conf.polar)),
        V_min_power=float(speed_minimum_power(W, S, atm.density, conf.polar)),
        min_speed_limited_by="stall" if V_s >= V_min_prop else "thrust/power",
    )


def flight_envelope(aircraft: Aircraft, altitudes, weight: Optional[float] = None,
                    config: str = "clean", throttle: float = 1.0, n_operative=None,
                    delta_T: float = 0.0) -> dict:
    """Level-flight envelope (V_min, V_max, V_stall versus altitude).

    The altitudes are processed in increasing order; the computation stops at
    the first altitude where level flight is impossible (the absolute ceiling
    has been crossed). Nothing is silently filled with NaN: the returned arrays
    only contain the feasible altitudes, and ``"first_infeasible_altitude"``
    tells where the envelope was closed (``None`` if all were feasible).
    """
    hs = np.sort(np.atleast_1d(np.asarray(altitudes, dtype=float)))
    rows = []
    first_infeasible = None
    for h in hs:
        try:
            rows.append(level_flight_speeds(aircraft, h, weight, config, throttle, n_operative, delta_T))
        except InfeasibleFlightConditionError:
            first_infeasible = float(h)
            break
    if not rows:
        raise InfeasibleFlightConditionError("Level flight is impossible at every requested altitude.")
    return {
        "altitude": np.array([r.altitude for r in rows]),
        "V_stall": np.array([r.V_stall for r in rows]),
        "V_min": np.array([r.V_min for r in rows]),
        "V_max": np.array([r.V_max for r in rows]),
        "V_min_drag": np.array([r.V_min_drag for r in rows]),
        "V_min_power": np.array([r.V_min_power for r in rows]),
        "first_infeasible_altitude": first_infeasible,
    }


def performance_curves(aircraft: Aircraft, altitude: float, V, weight: Optional[float] = None,
                       config: str = "clean", throttle: float = 1.0, n_operative=None,
                       delta_T: float = 0.0) -> dict:
    """Arrays of required/available thrust and power versus speed, for plotting.

    Returns a dictionary with keys ``V, CL, CD, T_req, T_av, P_req, P_av, D0, Di``.
    Speeds below the stall speed are *kept* (they are useful to see the
    whole curve) but flagged by ``below_stall`` (boolean array).
    """
    W = aircraft.weight_or_default(weight)
    V = np.asarray(ensure_positive("V", V), dtype=float)
    atm = isa(altitude, delta_T)
    conf = aircraft.config(config)
    S = aircraft.wing_area
    CL = aero.lift_coefficient(W, S, atm.density, V)
    D0, Di = aero.drag_breakdown(W, S, atm.density, V, conf.polar)
    T_req = D0 + Di
    T_av = aircraft.propulsion.thrust_available(V, altitude, throttle, n_operative, delta_T)
    return {
        "V": V, "CL": CL, "CD": conf.polar.drag_coefficient(CL),
        "T_req": ensure_result_finite("T_req", T_req), "T_av": T_av,
        "P_req": T_req * V, "P_av": T_av * V, "D0": D0, "Di": Di,
        "below_stall": CL > conf.CL_max,
    }
