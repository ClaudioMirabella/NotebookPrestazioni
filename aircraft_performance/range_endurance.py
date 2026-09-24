"""
Range, endurance (loiter) and payload-range.

Fundamental relations (Roskam & Lan, Chapter 11)
------------------------------------------------
The aircraft weight decreases only because fuel is burnt:
``dW/dt = -W_dot_f`` (fuel *weight* flow). Therefore::

    dt = -dW / W_dot_f            -> specific endurance  S.E. = 1 / W_dot_f      [s/N]
    ds = -V_ground dW / W_dot_f   -> specific range      S.R. = V_g / W_dot_f    [m/N]

with ``W_dot_f = c P_shaft = c D V / eta_p`` for propeller aircraft (Eqn 11.1)
and ``W_dot_f = c_t T = c_t D`` for jets (Eqn 11.47). Range and endurance are
the integrals of S.R. and S.E. over the fuel burnt (Eqns 11.29, 11.31); they
are evaluated numerically by :func:`cruise` for three classic flight programs:

1. ``"constant_altitude_constant_CL"`` -- the speed is reduced as fuel burns;
2. ``"constant_altitude_constant_speed"`` -- CL (and L/D) change during cruise;
3. ``"constant_speed_constant_CL"`` -- the *cruise-climb*: altitude increases
   so that ``rho / W`` stays constant.

Breguet closed forms (valid under the stated assumptions)
---------------------------------------------------------
Propeller (Eqns 11.7-11.8; the endurance needs constant altitude and CL)::

    R = (eta_p / c) (CL/CD) ln(W0/W1)
    E = (eta_p / c) (CL^1.5/CD) sqrt(2 rho S) (W1^-0.5 - W0^-0.5)

Jet (Eqns 11.55, 11.57, 11.60)::

    R_h,CL = (2 / c_t) sqrt(2 / (rho S)) (CL^0.5/CD) (sqrt(W0) - sqrt(W1))
    R_V,CL = (V / c_t) (CL/CD) ln(W0/W1)                      (cruise-climb)
    E      = (1 / c_t) (CL/CD) ln(W0/W1)

and, for constant altitude *and* constant speed with a parabolic polar,
``R = (2 E_max V / c_t) [atan(W0 a) - atan(W1 a)]`` with
``a = sqrt(K/CD0) / (q S)`` (the propeller version replaces V/c_t by eta_p/c).

Optimum conditions for a parabolic polar: propeller range at point E,
propeller endurance at point P; jet endurance at point E, jet range at point A
(constant altitude) or E (cruise-climb at a given speed).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import aerodynamics as aero
from ._numerics import maximize_on_interval
from .aerodynamics import ParabolicDragPolar
from .aircraft import Aircraft
from .atmosphere import altitude_from_density, isa
from .errors import InfeasibleFlightConditionError, InputError, PerformanceWarning
from .level_flight import level_flight_speeds
from .units import G0
from .validation import (ensure_fraction, ensure_greater, ensure_positive,
                         ensure_result_finite)

CRUISE_PROGRAMS = (
    "constant_altitude_constant_CL",
    "constant_altitude_constant_speed",
    "constant_speed_constant_CL",
)


def _check_weights(W0, W1):
    W0 = ensure_positive("W_start", W0)
    W1 = ensure_positive("W_end", W1)
    ensure_greater("W_start", W0, "W_end", W1)
    return W0, W1


# ---------------------------------------------------------------------------
# Breguet closed forms
# ---------------------------------------------------------------------------
def breguet_range_propeller(eta_p, bsfc, lift_to_drag, W_start, W_end) -> float:
    """Propeller Breguet range [m] (Roskam Eqn 11.7, SI units: bsfc in 1/m)."""
    eta = ensure_fraction("eta_p", eta_p)
    c = ensure_positive("bsfc", bsfc)
    E = ensure_positive("lift_to_drag", lift_to_drag)
    W0, W1 = _check_weights(W_start, W_end)
    return eta / c * E * np.log(W0 / W1)


def breguet_endurance_propeller(eta_p, bsfc, CL, CD, rho, wing_area, W_start, W_end) -> float:
    """Propeller endurance at constant altitude and CL [s] (Roskam Eqn 11.8)."""
    eta = ensure_fraction("eta_p", eta_p)
    c = ensure_positive("bsfc", bsfc)
    CL = ensure_positive("CL", CL)
    CD = ensure_positive("CD", CD)
    rho = ensure_positive("rho", rho)
    S = ensure_positive("wing_area", wing_area)
    W0, W1 = _check_weights(W_start, W_end)
    return eta / c * CL ** 1.5 / CD * np.sqrt(2.0 * rho * S) * (W1 ** -0.5 - W0 ** -0.5)


def breguet_range_jet_constant_altitude(tsfc, rho, wing_area, CL, CD, W_start, W_end) -> float:
    """Jet range at constant altitude and CL (speed decreasing) [m] (Roskam Eqn 11.55)."""
    c = ensure_positive("tsfc", tsfc)
    rho = ensure_positive("rho", rho)
    S = ensure_positive("wing_area", wing_area)
    CL = ensure_positive("CL", CL)
    CD = ensure_positive("CD", CD)
    W0, W1 = _check_weights(W_start, W_end)
    return 2.0 / c * np.sqrt(2.0 / (rho * S)) * np.sqrt(CL) / CD * (np.sqrt(W0) - np.sqrt(W1))


def breguet_range_jet_constant_speed(V, tsfc, lift_to_drag, W_start, W_end) -> float:
    """Jet range at constant speed and CL, i.e. cruise-climb [m] (Roskam Eqn 11.60)."""
    V = ensure_positive("V", V)
    c = ensure_positive("tsfc", tsfc)
    E = ensure_positive("lift_to_drag", lift_to_drag)
    W0, W1 = _check_weights(W_start, W_end)
    return V / c * E * np.log(W0 / W1)


def breguet_endurance_jet(tsfc, lift_to_drag, W_start, W_end) -> float:
    """Jet endurance at constant CL (any altitude program) [s] (Roskam Eqns 11.57, 11.63)."""
    c = ensure_positive("tsfc", tsfc)
    E = ensure_positive("lift_to_drag", lift_to_drag)
    W0, W1 = _check_weights(W_start, W_end)
    return E / c * np.log(W0 / W1)


def _arctan_term(polar: ParabolicDragPolar, q, S, W0, W1):
    a = np.sqrt(polar.K / polar.CD0) / (q * S)
    return 2.0 * polar.max_lift_to_drag * (np.arctan(W0 * a) - np.arctan(W1 * a))


def range_constant_altitude_speed_jet(V, rho, wing_area, polar: ParabolicDragPolar, tsfc,
                                      W_start, W_end) -> float:
    """Jet range at constant altitude AND constant speed, parabolic polar [m] (closed form)."""
    V = ensure_positive("V", V)
    rho = ensure_positive("rho", rho)
    c = ensure_positive("tsfc", tsfc)
    W0, W1 = _check_weights(W_start, W_end)
    q = 0.5 * rho * V ** 2
    return V / c * _arctan_term(polar, q, wing_area, W0, W1)


def range_constant_altitude_speed_propeller(V, rho, wing_area, polar: ParabolicDragPolar,
                                            eta_p, bsfc, W_start, W_end) -> float:
    """Propeller range at constant altitude AND constant speed, parabolic polar [m] (closed form)."""
    V = ensure_positive("V", V)
    rho = ensure_positive("rho", rho)
    eta = ensure_fraction("eta_p", eta_p)
    c = ensure_positive("bsfc", bsfc)
    W0, W1 = _check_weights(W_start, W_end)
    q = 0.5 * rho * V ** 2
    return eta / c * _arctan_term(polar, q, wing_area, W0, W1)


def cruise_climb_final_altitude(h_start: float, W_start, W_end, delta_T: float = 0.0) -> float:
    """Final altitude of a cruise-climb (constant V and CL): rho1 = rho0 W1/W0 [m]."""
    W0, W1 = _check_weights(W_start, W_end)
    rho0 = isa(h_start, delta_T).density
    return altitude_from_density(rho0 * W1 / W0, delta_T)


# ---------------------------------------------------------------------------
# Specific range and endurance
# ---------------------------------------------------------------------------
def fuel_flow_level_flight(aircraft: Aircraft, V, altitude: float, weight: Optional[float] = None,
                           config: str = "clean", delta_T: float = 0.0, check_thrust: bool = True):
    """Fuel weight flow [N/s] needed for steady level flight at speed V.

    Raises
    ------
    InfeasibleFlightConditionError
        If ``check_thrust`` and the drag exceeds the maximum available thrust.
    """
    W = aircraft.weight_or_default(weight)
    V = np.asarray(ensure_positive("V", V), dtype=float)
    rho = isa(altitude, delta_T).density
    D = aero.drag(W, aircraft.wing_area, rho, V, aircraft.polar(config))
    if check_thrust:
        T_max = aircraft.propulsion.thrust_available(V, altitude, 1.0, None, delta_T)
        if np.any(D > T_max * (1.0 + 1e-9)):
            raise InfeasibleFlightConditionError(
                f"Drag exceeds the maximum available thrust at h = {altitude:.0f} m: "
                "this cruise condition cannot be flown.")
    return aircraft.propulsion.fuel_weight_flow_for_thrust(D, V)


def specific_range(aircraft: Aircraft, V, altitude: float, weight: Optional[float] = None,
                   config: str = "clean", headwind: float = 0.0, delta_T: float = 0.0):
    """Specific range S.R. = V_ground / W_dot_f [m of ground distance per N of fuel] (Eqn 11.28).

    Multiply by g0 to obtain metres per kilogram of fuel.
    """
    ff = fuel_flow_level_flight(aircraft, V, altitude, weight, config, delta_T, check_thrust=False)
    return (np.asarray(V) - float(headwind)) / ff


def specific_endurance(aircraft: Aircraft, V, altitude: float, weight: Optional[float] = None,
                       config: str = "clean", delta_T: float = 0.0):
    """Specific endurance S.E. = 1 / W_dot_f [s per N of fuel] (Roskam Eqn 11.30)."""
    return 1.0 / fuel_flow_level_flight(aircraft, V, altitude, weight, config, delta_T, check_thrust=False)


def _speed_limits(aircraft, altitude, weight, config, delta_T):
    speeds = level_flight_speeds(aircraft, altitude, weight, config, delta_T=delta_T)
    return speeds.V_min, speeds.V_max


def best_range_speed(aircraft: Aircraft, altitude: float, weight: Optional[float] = None,
                     config: str = "clean", headwind: float = 0.0, delta_T: float = 0.0) -> dict:
    """Speed of maximum specific range (with wind), within the level-flight envelope (Roskam Fig. 11.7)."""
    W = aircraft.weight_or_default(weight)
    lo, hi = _speed_limits(aircraft, altitude, W, config, delta_T)
    V, sr = maximize_on_interval(
        lambda v: float(specific_range(aircraft, v, altitude, W, config, headwind, delta_T)), lo, hi)
    if sr <= 0.0:
        raise InfeasibleFlightConditionError("The headwind exceeds the airspeed: no progress over the ground.")
    return {"V": V, "specific_range": sr, "CL": float(aero.lift_coefficient(W, aircraft.wing_area,
                                                                           isa(altitude, delta_T).density, V))}


def best_endurance_speed(aircraft: Aircraft, altitude: float, weight: Optional[float] = None,
                         config: str = "clean", delta_T: float = 0.0) -> dict:
    """Speed of maximum specific endurance (minimum fuel flow), within the envelope."""
    W = aircraft.weight_or_default(weight)
    lo, hi = _speed_limits(aircraft, altitude, W, config, delta_T)
    V, se = maximize_on_interval(
        lambda v: float(specific_endurance(aircraft, v, altitude, W, config, delta_T)), lo, hi)
    return {"V": V, "specific_endurance": se, "CL": float(aero.lift_coefficient(W, aircraft.wing_area,
                                                                               isa(altitude, delta_T).density, V))}


# ---------------------------------------------------------------------------
# Numerical cruise integration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CruiseResult:
    """Time histories of a cruise (arrays at the weight nodes, from start to end)."""

    program: str
    weight: np.ndarray       #: [N]
    distance: np.ndarray     #: cumulated ground distance [m]
    time: np.ndarray         #: cumulated time [s]
    altitude: np.ndarray     #: [m]
    speed: np.ndarray        #: true airspeed [m/s]
    CL: np.ndarray
    lift_to_drag: np.ndarray
    fuel_flow: np.ndarray    #: fuel weight flow [N/s]

    @property
    def range(self) -> float:
        """Total ground distance [m]."""
        return float(self.distance[-1])

    @property
    def endurance(self) -> float:
        """Total time [s]."""
        return float(self.time[-1])

    @property
    def fuel_weight(self) -> float:
        """Fuel weight burnt [N]."""
        return float(self.weight[0] - self.weight[-1])


def cruise(aircraft: Aircraft, W_start: float, W_end: float, program: str, altitude: float,
           speed: Optional[float] = None, CL: Optional[float] = None, config: str = "clean",
           headwind: float = 0.0, delta_T: float = 0.0, n_steps: int = 200,
           check_thrust: bool = True) -> CruiseResult:
    """Integrate a cruise (or loiter) numerically between two weights (Roskam Secs. 11.1.5, 11.2.4).

    Parameters
    ----------
    program : str
        One of :data:`CRUISE_PROGRAMS`.
    altitude : float
        Cruise altitude [m] (initial altitude for the cruise-climb).
    speed, CL : float, optional
        Initial condition: give exactly one of them. For constant-CL programs
        ``speed`` is the *initial* speed; for the constant-speed program ``CL``
        is the *initial* lift coefficient.
    headwind : float
        Constant headwind [m/s] (negative for tailwind); affects range only.
    """
    if program not in CRUISE_PROGRAMS:
        raise InputError(f"program must be one of {CRUISE_PROGRAMS}, got {program!r}.")
    if (speed is None) == (CL is None):
        raise InputError("Specify exactly one of 'speed' or 'CL' as initial condition.")
    W0, W1 = _check_weights(W_start, W_end)
    if W0 > aircraft.weight * 1.001:
        raise InputError("W_start exceeds the aircraft reference (maximum take-off) weight.")
    conf = aircraft.config(config)
    S = aircraft.wing_area
    atm0 = isa(altitude, delta_T)
    if speed is not None:
        V0 = ensure_positive("speed", speed)
        CL0 = float(aero.lift_coefficient(W0, S, atm0.density, V0))
    else:
        CL0 = ensure_positive("CL", CL)
        V0 = float(aero.speed_for_lift_coefficient(W0, S, atm0.density, CL0))

    W = np.linspace(W0, W1, int(n_steps) + 1)
    if program == "constant_altitude_constant_CL":
        h = np.full_like(W, float(altitude))
        CLs = np.full_like(W, CL0)
        rho = np.full_like(W, atm0.density)
        V = np.sqrt(2.0 * W / (rho * S * CLs))
    elif program == "constant_altitude_constant_speed":
        h = np.full_like(W, float(altitude))
        V = np.full_like(W, V0)
        rho = np.full_like(W, atm0.density)
        CLs = 2.0 * W / (rho * V ** 2 * S)
    else:  # cruise-climb
        V = np.full_like(W, V0)
        CLs = np.full_like(W, CL0)
        rho = atm0.density * W / W0
        h = np.array([altitude_from_density(r, delta_T) for r in rho])

    if np.any(CLs > conf.CL_max):
        raise InfeasibleFlightConditionError(
            f"The cruise requires CL up to {CLs.max():.2f} > CL_max = {conf.CL_max:.2f} (stall).")
    CD = conf.polar.drag_coefficient(CLs)
    D = W * CD / CLs
    if check_thrust:
        T_max = np.array([aircraft.propulsion.thrust_available(v, hh, 1.0, None, delta_T)
                          for v, hh in zip(V, h)])
        if np.any(D > T_max * (1.0 + 1e-9)):
            i = int(np.argmax(D - T_max))
            raise InfeasibleFlightConditionError(
                f"Drag ({D[i]:.0f} N) exceeds maximum thrust ({T_max[i]:.0f} N) at "
                f"W = {W[i]:.0f} N, h = {h[i]:.0f} m: this cruise cannot be flown.")
    a = np.array([isa(hh, delta_T).speed_of_sound for hh in h])
    conf.polar.check_mach(V / a, context="Cruise: ")
    ff = aircraft.propulsion.fuel_weight_flow_for_thrust(D, V)
    Vg = V - float(headwind)
    if np.any(Vg <= 0.0):
        raise InfeasibleFlightConditionError("The headwind exceeds the airspeed during the cruise.")
    # W decreases along the arrays: ds = -Vg/ff dW, dt = -1/ff dW
    dist = cumulative_trapezoid(Vg / ff, -W, initial=0.0)
    time = cumulative_trapezoid(1.0 / ff, -W, initial=0.0)
    ensure_result_finite("cruise distance", dist)
    return CruiseResult(program=program, weight=W, distance=dist, time=time, altitude=h,
                        speed=V, CL=CLs, lift_to_drag=CLs / CD, fuel_flow=ff)


def loiter(aircraft: Aircraft, W_start: float, W_end: float, altitude: float,
           config: str = "clean", delta_T: float = 0.0, n_steps: int = 200) -> CruiseResult:
    """Maximum-endurance loiter at constant altitude (Roskam Secs. 11.1.2.4, 11.2.2.5).

    The lift coefficient is kept at its optimum for endurance: point P
    (max CL^1.5/CD) for propeller aircraft, point E (max CL/CD) for jets.
    If that CL exceeds CL_max / 1.2^2 (a 20 % margin above the stall speed)
    the limited value is used and a warning is issued.
    """
    conf = aircraft.config(config)
    CL_opt = conf.polar.point_E.CL if aircraft.is_jet else conf.polar.point_P.CL
    CL_limit = conf.CL_max / 1.2 ** 2
    if CL_opt > CL_limit:
        warnings.warn(f"Optimum loiter CL = {CL_opt:.2f} is too close to the stall; "
                      f"using CL = {CL_limit:.2f} (V = 1.2 V_S).", PerformanceWarning, stacklevel=2)
        CL_opt = CL_limit
    return cruise(aircraft, W_start, W_end, "constant_altitude_constant_CL", altitude, CL=CL_opt,
                  config=config, delta_T=delta_T, n_steps=n_steps)


# ---------------------------------------------------------------------------
# Payload-range diagram
# ---------------------------------------------------------------------------
def payload_range(aircraft: Aircraft, altitude: float, program: str = "constant_altitude_constant_CL",
                  speed: Optional[float] = None, CL: Optional[float] = None,
                  reserve_fraction: float = 0.10, config: str = "clean",
                  delta_T: float = 0.0) -> dict:
    """Payload-range diagram corner points (Roskam Sec. 11.3.1, Fig. 11.28).

    Simplifying assumptions (clearly stated so they can be refined):

    * the cruise starts at the take-off weight (fuel for taxi, take-off and
      climb is included in the reserve fraction);
    * a fraction ``reserve_fraction`` of the loaded fuel is not used for
      cruise (reserves + non-cruise phases);
    * the cruise follows the chosen ``program`` from the same initial
      condition (``speed`` or ``CL``).

    Returns a dictionary with the points ``A`` (max payload, zero range),
    ``B`` (max payload, harmonic range), ``C`` (max fuel) and ``D`` (ferry),
    each as a dict with payload mass, fuel mass, take-off mass and range, plus
    arrays ``range`` and ``payload`` for plotting.
    """
    for name in ("operating_empty_mass", "max_payload_mass", "max_fuel_mass"):
        if getattr(aircraft, name) is None:
            raise InputError(f"payload_range needs aircraft.{name}.")
    r = ensure_fraction("reserve_fraction", reserve_fraction, allow_zero=True)
    if r >= 1.0:
        raise InputError("reserve_fraction must be smaller than 1.")
    m_oe = aircraft.operating_empty_mass
    m_to_max = aircraft.mass
    pl_max = aircraft.max_payload_mass
    fuel_max = aircraft.max_fuel_mass
    if m_oe + pl_max > m_to_max:
        raise InputError("OEM + max payload exceeds the maximum take-off mass.")

    def range_for(payload, fuel):
        if fuel <= 0.0:
            return 0.0
        m0 = m_oe + payload + fuel
        W0 = m0 * G0
        W1 = (m0 - (1.0 - r) * fuel) * G0
        res = cruise(aircraft, W0, W1, program, altitude, speed=speed, CL=CL,
                     config=config, delta_T=delta_T, n_steps=100)
        return res.range

    # B: maximum payload, as much fuel as the MTOM (and the tanks) allow
    fuel_B = min(m_to_max - m_oe - pl_max, fuel_max)
    # C: full tanks (if the MTOM allows it), payload reduced to respect the MTOM
    fuel_C = min(fuel_max, m_to_max - m_oe)
    payload_C = min(pl_max, m_to_max - m_oe - fuel_C)
    # D: ferry range, no payload; the fuel can never exceed MTOM - OEM
    points = {
        "A": {"payload": pl_max, "fuel": 0.0},
        "B": {"payload": pl_max, "fuel": fuel_B},
        "C": {"payload": payload_C, "fuel": fuel_C},
        "D": {"payload": 0.0, "fuel": fuel_C},
    }
    for p in points.values():
        p["takeoff_mass"] = m_oe + p["payload"] + p["fuel"]
        p["range"] = range_for(p["payload"], p["fuel"])
    order = ["A", "B", "C", "D"]
    return {
        **points,
        "range": np.array([points[k]["range"] for k in order]),
        "payload": np.array([points[k]["payload"] for k in order]),
    }


__all__ = [
    "CRUISE_PROGRAMS", "breguet_range_propeller", "breguet_endurance_propeller",
    "breguet_range_jet_constant_altitude", "breguet_range_jet_constant_speed", "breguet_endurance_jet",
    "range_constant_altitude_speed_jet", "range_constant_altitude_speed_propeller",
    "cruise_climb_final_altitude", "fuel_flow_level_flight", "specific_range", "specific_endurance",
    "best_range_speed", "best_endurance_speed", "CruiseResult", "cruise", "loiter", "payload_range",
]
