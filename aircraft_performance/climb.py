"""
Climb performance: rate of climb, climb angle, ceilings and time to climb.

Equations (Roskam & Lan, Chapter 9)
-----------------------------------
Quasi-steady, small flight-path angles (``L = W``, Eqns 9.6-9.12)::

    RC = V sin(gamma) = (T - D) V / W = (P_av - P_req) / W
    sin(gamma) = (T - D) / W                           (climb gradient, Eqn 9.11a)

*Jet, parabolic polar, constant thrust* (Sec. 9.2.1.2)
    best-RC lift coefficient (Eqn 9.30):
    ``CL = -(T/W) pi A e / 2 + sqrt(((T/W) pi A e / 2)^2 + 3 CD0 pi A e)``;
    steepest climb at point E: ``sin(gamma_max) = T/W - 1/E_max`` (Eqns 9.32-9.34).

*Propeller, parabolic polar, constant power* (Sec. 9.3.1.2)
    best RC at point P of the polar, ``CL = sqrt(3 CD0 pi A e)`` (Eqn 9.57).

*Steep climbs* (Sec. 9.2.2, 9.3.2): ``L = W cos(gamma)``, solved exactly here.

*Accelerated climb* (Sec. 9.5): if the true airspeed changes with altitude
(constant EAS or constant Mach schedule) part of the excess power goes into
kinetic energy::

    RC = (T - D) V / W / (1 + (V/g) dV/dh)             (Eqn 9.10)

*Ceilings* (Sec. 9.4.3): absolute (RC_max = 0), service (RC_max = 100 ft/min
for propeller aircraft, 500 ft/min for jets at maximum continuous thrust),
cruise ceiling (300 ft/min).

*Time to climb* (Sec. 9.4.1): ``t = integral dh / RC_max`` evaluated
numerically in altitude steps (Roskam Table 9.6), together with the horizontal
distance and the fuel burnt. The linearised closed form of Eqn (9.74) is also
provided for comparison.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import aerodynamics as aero
from ._numerics import maximize_on_interval, root_in_bracket
from .aerodynamics import ParabolicDragPolar
from .aircraft import Aircraft
from .atmosphere import H_MAX, isa
from .errors import (InfeasibleFlightConditionError, InputError, PerformanceError,
                     PerformanceWarning)
from .level_flight import V_SEARCH_MAX
from .units import FPM, G0
from .validation import (ensure_in_range, ensure_non_negative, ensure_positive,
                         ensure_result_finite)

#: Rate of climb defining the service ceiling (Roskam Sec. 9.4.3) [m/s].
SERVICE_CEILING_RC = {"propeller": 100.0 * FPM, "jet": 500.0 * FPM}
#: Rate of climb defining the cruise ceiling (military, M < 1) [m/s].
CRUISE_CEILING_RC = 300.0 * FPM


# ---------------------------------------------------------------------------
# Primitive formulas (small angles)
# ---------------------------------------------------------------------------
def rate_of_climb_from_forces(thrust, drag_force, weight, V):
    """RC = (T - D) V / W [m/s] (Roskam Eqn 9.11)."""
    W = ensure_positive("weight", weight)
    return (np.asarray(thrust) - np.asarray(drag_force)) * np.asarray(V) / W


def rate_of_climb_from_power(power_available, power_required, weight):
    """RC = (P_av - P_req) / W [m/s] (Roskam Eqn 9.12)."""
    W = ensure_positive("weight", weight)
    return (np.asarray(power_available) - np.asarray(power_required)) / W


def jet_best_climb_lift_coefficient(thrust_to_weight, polar: ParabolicDragPolar) -> float:
    """Lift coefficient for maximum RC of a jet, parabolic polar (Roskam Eqn 9.30)."""
    TW = ensure_positive("thrust_to_weight", thrust_to_weight)
    piAe = 1.0 / polar.K
    return float(-TW * piAe / 2.0 + np.sqrt((TW * piAe / 2.0) ** 2 + 3.0 * polar.CD0 * piAe))


@dataclass(frozen=True)
class ClimbPoint:
    """An optimum climb condition."""

    V: float          #: true airspeed [m/s]
    rate: float       #: rate of climb [m/s]
    gamma: float      #: flight-path angle [rad]
    CL: float         #: lift coefficient [-]

    @property
    def gamma_deg(self) -> float:
        """Flight-path angle in degrees."""
        return float(np.degrees(self.gamma))

    @property
    def rate_fpm(self) -> float:
        """Rate of climb in ft/min."""
        return self.rate / FPM


def jet_max_rate_of_climb_parabolic(weight, wing_area, rho, polar: ParabolicDragPolar, thrust) -> ClimbPoint:
    """Maximum RC of a jet with constant thrust and parabolic polar (small angles)."""
    W = ensure_positive("weight", weight)
    T = ensure_positive("thrust", thrust)
    CL = jet_best_climb_lift_coefficient(T / W, polar)
    V = float(aero.speed_for_lift_coefficient(W, wing_area, rho, CL))
    D = W * polar.drag_coefficient(CL) / CL
    RC = (T - D) * V / W
    return ClimbPoint(V=V, rate=float(RC), gamma=float(np.arcsin(np.clip(RC / V, -1, 1))), CL=CL)


def jet_max_climb_angle_parabolic(weight, wing_area, rho, polar: ParabolicDragPolar, thrust) -> ClimbPoint:
    """Steepest climb of a jet: point E, ``sin(gamma) = T/W - 1/E_max`` (Roskam Eqn 9.32-9.34).

    The speed accounts for ``L = W cos(gamma)`` as in Anderson Eqn (5.98).
    """
    W = ensure_positive("weight", weight)
    T = ensure_positive("thrust", thrust)
    sin_g = T / W - 1.0 / polar.max_lift_to_drag
    if sin_g > 1.0:
        raise InfeasibleFlightConditionError("T/W is large enough for a vertical climb.")
    gamma = float(np.arcsin(sin_g))
    CL = polar.point_E.CL
    V = float(np.sqrt(2.0 * W * np.cos(gamma) / (rho * wing_area * CL)))
    return ClimbPoint(V=V, rate=V * sin_g, gamma=gamma, CL=CL)


def propeller_max_rate_of_climb_parabolic(weight, wing_area, rho, polar: ParabolicDragPolar,
                                          power_available) -> ClimbPoint:
    """Maximum RC of a propeller aircraft with constant power: Roskam Steps 1-6, Sec. 9.3.1.2."""
    W = ensure_positive("weight", weight)
    P_av = ensure_positive("power_available", power_available)
    point = polar.point_P                                          # Step 1
    V = float(aero.speed_for_lift_coefficient(W, wing_area, rho, point.CL))  # Step 2
    D = W * point.CD / point.CL                                     # Steps 3-4
    RC = (P_av - D * V) / W                                         # Steps 5-6
    return ClimbPoint(V=V, rate=float(RC), gamma=float(np.arcsin(np.clip(RC / V, -1, 1))), CL=point.CL)


def propeller_steep_climb_sin_gamma(power_available, weight, V, CL, CD) -> float:
    """Exact climb angle for a propeller aircraft at large angles (Roskam Eqn 9.64).

    Solves the quadratic Eqn (9.63) in sin(gamma) with ``x = P_av / (W V)`` and
    ``r = CL^2 / (CL^2 + CD^2)`` (minus sign in front of the square root).
    Roskam then iterates on CL because ``L = W cos(gamma)``; see
    :func:`steep_climb` for the fully consistent solution.
    """
    x = ensure_positive("power_available", power_available) / (ensure_positive("weight", weight)
                                                                * ensure_positive("V", V))
    r = CL ** 2 / (CL ** 2 + CD ** 2)
    disc = (r * x) ** 2 - r * x ** 2 + CD ** 2 / (CL ** 2 + CD ** 2)
    if disc < 0.0:
        raise InfeasibleFlightConditionError("No real solution for the steep climb angle.")
    return float(r * x - np.sqrt(disc))


# ---------------------------------------------------------------------------
# General numerical methods for an Aircraft
# ---------------------------------------------------------------------------
def rate_of_climb(aircraft: Aircraft, V, altitude: float, weight: Optional[float] = None,
                  config: str = "clean", throttle: float = 1.0, n_operative=None,
                  delta_T: float = 0.0, extra_CD0: float = 0.0):
    """Quasi-steady rate of climb (small angles) at speed(s) V [m/s].

    A negative value is a rate of descent. ``extra_CD0`` adds drag, e.g. the
    windmilling/trim drag of an inoperative engine (Roskam Sec. 9.6.2).
    """
    W = aircraft.weight_or_default(weight)
    rho = isa(altitude, delta_T).density
    polar = aircraft.polar(config).with_increments(delta_CD0=ensure_non_negative("extra_CD0", extra_CD0))
    V = ensure_positive("V", V)
    T = aircraft.propulsion.thrust_available(V, altitude, throttle, n_operative, delta_T)
    D = aero.drag(W, aircraft.wing_area, rho, V, polar)
    return ensure_result_finite("rate of climb", (T - D) * np.asarray(V) / W)


def steep_climb(aircraft: Aircraft, V: float, altitude: float, weight: Optional[float] = None,
                config: str = "clean", throttle: float = 1.0, n_operative=None,
                delta_T: float = 0.0) -> ClimbPoint:
    """Exact steady climb at speed V without small-angle approximations.

    Solves ``T - D(CL) - W sin(gamma) = 0`` with ``CL = 2 W cos(gamma) / (rho V^2 S)``
    for gamma in [-90 deg, +90 deg] (Roskam Eqns 9.35-9.38 and 9.58-9.64).
    """
    W = aircraft.weight_or_default(weight)
    V = ensure_positive("V", V)
    rho = isa(altitude, delta_T).density
    polar = aircraft.polar(config)
    q = 0.5 * rho * V ** 2
    S = aircraft.wing_area
    T = float(aircraft.propulsion.thrust_available(V, altitude, throttle, n_operative, delta_T))

    def residual(gamma):
        CL = W * np.cos(gamma) / (q * S)
        return T - q * S * polar.drag_coefficient(CL) - W * np.sin(gamma)

    lo, hi = -np.pi / 2 + 1e-9, np.pi / 2 - 1e-9
    if residual(hi) > 0.0:
        raise InfeasibleFlightConditionError(
            "Thrust exceeds weight plus drag even in a vertical climb: no steady solution.")
    gamma = root_in_bracket(residual, lo, hi, "steep climb angle")
    CL = W * np.cos(gamma) / (q * S)
    if CL > aircraft.config(config).CL_max:
        warnings.warn(f"Steep climb at V = {V:.1f} m/s requires CL = {CL:.2f} > CL_max.",
                      PerformanceWarning, stacklevel=2)
    return ClimbPoint(V=float(V), rate=float(V * np.sin(gamma)), gamma=float(gamma), CL=float(CL))


def _speed_bounds(aircraft: Aircraft, altitude, W, config, delta_T, margin: float):
    rho = isa(altitude, delta_T).density
    V_s = float(aero.stall_speed(W, aircraft.wing_area, rho, aircraft.config(config).CL_max))
    return margin * V_s, V_SEARCH_MAX


def max_rate_of_climb(aircraft: Aircraft, altitude: float, weight: Optional[float] = None,
                      config: str = "clean", throttle: float = 1.0, n_operative=None,
                      delta_T: float = 0.0, extra_CD0: float = 0.0,
                      min_speed_margin: float = 1.0) -> ClimbPoint:
    """Maximum rate of climb and best-RC speed, found numerically (small angles).

    The search starts at ``min_speed_margin * V_stall`` (use e.g. 1.2 to
    respect a regulatory speed margin). The result may be negative: it is
    then the *minimum rate of descent* (drift-down).
    """
    W = aircraft.weight_or_default(weight)
    lo, hi = _speed_bounds(aircraft, altitude, W, config, delta_T,
                           ensure_in_range("min_speed_margin", min_speed_margin, 1.0, 3.0))

    def rc(V):
        return float(rate_of_climb(aircraft, V, altitude, W, config, throttle, n_operative,
                                   delta_T, extra_CD0))

    V_opt, RC = maximize_on_interval(rc, lo, hi, n_grid=200, log_spacing=True)
    rho = isa(altitude, delta_T).density
    CL = float(aero.lift_coefficient(W, aircraft.wing_area, rho, V_opt))
    gamma = float(np.arcsin(np.clip(RC / V_opt, -1.0, 1.0)))
    if abs(np.degrees(gamma)) > 15.0:
        warnings.warn(f"Best climb angle is {np.degrees(gamma):.1f} deg > 15 deg: the small-angle "
                      "result is approximate, see steep_climb().", PerformanceWarning, stacklevel=2)
    aircraft.polar(config).check_mach(V_opt / isa(altitude, delta_T).speed_of_sound,
                                      context="Best rate-of-climb speed: ")
    return ClimbPoint(V=V_opt, rate=RC, gamma=gamma, CL=CL)


def max_climb_angle(aircraft: Aircraft, altitude: float, weight: Optional[float] = None,
                    config: str = "clean", throttle: float = 1.0, n_operative=None,
                    delta_T: float = 0.0, extra_CD0: float = 0.0,
                    min_speed_margin: float = 1.0) -> ClimbPoint:
    """Steepest climb (maximum gradient (T - D)/W), found numerically (small angles)."""
    W = aircraft.weight_or_default(weight)
    lo, hi = _speed_bounds(aircraft, altitude, W, config, delta_T,
                           ensure_in_range("min_speed_margin", min_speed_margin, 1.0, 3.0))

    def gradient(V):
        return float(rate_of_climb(aircraft, V, altitude, W, config, throttle, n_operative,
                                   delta_T, extra_CD0)) / V

    V_opt, grad = maximize_on_interval(gradient, lo, hi, n_grid=200, log_spacing=True)
    rho = isa(altitude, delta_T).density
    CL = float(aero.lift_coefficient(W, aircraft.wing_area, rho, V_opt))
    if grad > 1.0:
        raise InfeasibleFlightConditionError("Climb gradient > 1: use steep_climb().")
    return ClimbPoint(V=V_opt, rate=grad * V_opt, gamma=float(np.arcsin(grad)), CL=CL)


# ---------------------------------------------------------------------------
# Acceleration factor (Roskam Sec. 9.5)
# ---------------------------------------------------------------------------
def acceleration_factor(speed_schedule: Callable[[float], float], altitude: float,
                        dh: float = 10.0) -> float:
    """Acceleration factor ``(V/g) dV/dh`` for a given TAS schedule V(h) (Roskam Eqn 9.10).

    ``speed_schedule`` is a function returning the true airspeed [m/s] at
    altitude h; the derivative is computed with central finite differences.
    """
    h = float(altitude)
    dh = ensure_positive("dh", dh)
    h_lo, h_hi = max(h - dh, -1000.0), min(h + dh, H_MAX)
    V = speed_schedule(h)
    dVdh = (speed_schedule(h_hi) - speed_schedule(h_lo)) / (h_hi - h_lo)
    return float(V / G0 * dVdh)


def acceleration_factor_constant_eas(V_eas: float, altitude: float, delta_T: float = 0.0) -> float:
    """Acceleration factor for a climb at constant equivalent airspeed (numerical)."""
    V_eas = ensure_positive("V_eas", V_eas)

    def schedule(h):
        return V_eas / np.sqrt(isa(h, delta_T).sigma)
    return acceleration_factor(schedule, altitude)


def acceleration_factor_constant_mach(mach: float, altitude: float, delta_T: float = 0.0) -> float:
    """Acceleration factor for a climb at constant Mach number (numerical).

    Negative in the troposphere (the speed of sound decreases), zero in the
    isothermal stratosphere.
    """
    M = ensure_positive("mach", mach)

    def schedule(h):
        return M * isa(h, delta_T).speed_of_sound
    return acceleration_factor(schedule, altitude)


def acceleration_factor_roskam_troposphere(mach: float, schedule: str) -> float:
    """Roskam closed forms in the troposphere: 0.567 M^2 (const. EAS, Eqn 9.84), -0.133 M^2 (const. M, Eqn 9.87)."""
    M = ensure_non_negative("mach", mach)
    if schedule == "eas":
        return 0.567 * M ** 2
    if schedule == "mach":
        return -0.133 * M ** 2
    raise InputError("schedule must be 'eas' or 'mach'.")


def accelerated_rate_of_climb(steady_rate_of_climb, accel_factor):
    """RC corrected for flight-path acceleration: RC_steady / (1 + (V/g) dV/dh) (Roskam Eqn 9.80)."""
    af = np.asarray(accel_factor, dtype=float)
    if np.any(af <= -1.0):
        raise InputError("Acceleration factor must be greater than -1.")
    return np.asarray(steady_rate_of_climb) / (1.0 + af)


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ceilings:
    """Ceilings at a given weight and throttle setting [m]."""

    absolute: float          #: RC_max = 0
    service: float           #: RC_max = service threshold
    service_rate: float      #: threshold used for the service ceiling [m/s]
    cruise: Optional[float]  #: RC_max = 300 ft/min (None if not computed)


def ceiling_for_rate(aircraft: Aircraft, target_rate: float, weight: Optional[float] = None,
                     config: str = "clean", throttle: float = 1.0, n_operative=None,
                     delta_T: float = 0.0, extra_CD0: float = 0.0) -> float:
    """Altitude [m] at which the maximum rate of climb equals ``target_rate``.

    Raises
    ------
    InfeasibleFlightConditionError
        If even at sea level the aircraft cannot reach ``target_rate``.
    PerformanceError
        If the ceiling lies above the upper limit of the atmosphere model.
    """
    target = ensure_non_negative("target_rate", target_rate)
    W = aircraft.weight_or_default(weight)

    def f(h):
        # While searching, the trial altitudes may lie far above the ceiling, where the
        # "best" speed is meaningless (e.g. supersonic): silence the warnings there.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)
            return max_rate_of_climb(aircraft, h, W, config, throttle, n_operative, delta_T,
                                     extra_CD0).rate - target

    h_bottom = 0.0
    if f(h_bottom) < 0.0:
        raise InfeasibleFlightConditionError(
            f"RC_max at sea level is below {target:.3f} m/s: the ceiling does not exist.")
    # Bracket the ceiling by stepping upward (RC_max decreases with altitude)
    step = 1000.0
    h_top = h_bottom
    while f(h_top) > 0.0:
        h_bottom = h_top
        h_top = min(h_top + step, H_MAX)
        if h_top == h_bottom:
            raise PerformanceError(f"The ceiling is above {H_MAX:.0f} m (outside the atmosphere model).")
    h_ceiling = root_in_bracket(f, h_bottom, h_top, "ceiling", xtol=0.5)
    # Re-evaluate once at the ceiling, now with warnings enabled (e.g. Mach > M_DD)
    max_rate_of_climb(aircraft, h_ceiling, W, config, throttle, n_operative, delta_T, extra_CD0)
    return h_ceiling


def ceilings(aircraft: Aircraft, weight: Optional[float] = None, config: str = "clean",
             throttle: float = 1.0, n_operative=None, delta_T: float = 0.0,
             service_rate: Optional[float] = None, include_cruise: bool = False,
             extra_CD0: float = 0.0) -> Ceilings:
    """Absolute, service (and optionally cruise) ceilings (Roskam Sec. 9.4.3).

    ``extra_CD0`` adds drag, e.g. windmilling and trim drag with one engine inoperative.
    """
    if service_rate is None:
        service_rate = SERVICE_CEILING_RC[aircraft.propulsion.kind]
    args = (weight, config, throttle, n_operative, delta_T, extra_CD0)
    absolute = ceiling_for_rate(aircraft, 0.0, *args)
    service = ceiling_for_rate(aircraft, service_rate, *args)
    cruise = ceiling_for_rate(aircraft, CRUISE_CEILING_RC, *args) if include_cruise else None
    return Ceilings(absolute=absolute, service=service, service_rate=float(service_rate), cruise=cruise)


# ---------------------------------------------------------------------------
# Time to climb
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClimbProfile:
    """Result of a time-to-climb integration (arrays at the altitude nodes)."""

    altitude: np.ndarray     #: [m]
    time: np.ndarray         #: cumulated time [s]
    distance: np.ndarray     #: cumulated horizontal distance [m]
    fuel_weight: np.ndarray  #: cumulated fuel weight burnt [N]
    weight: np.ndarray       #: aircraft weight [N]
    rate_of_climb: np.ndarray  #: RC used at each node [m/s]
    speed: np.ndarray        #: TAS at each node [m/s]

    @property
    def total_time(self) -> float:
        return float(self.time[-1])

    @property
    def total_distance(self) -> float:
        return float(self.distance[-1])

    @property
    def total_fuel_weight(self) -> float:
        return float(self.fuel_weight[-1])


def time_to_climb(aircraft: Aircraft, h_start: float, h_end: float, weight: Optional[float] = None,
                  config: str = "clean", throttle: float = 1.0, n_operative=None,
                  delta_T: float = 0.0, n_steps: int = 40,
                  accelerated: bool = False, min_rate: float = 0.05) -> ClimbProfile:
    """Numerical time, distance and fuel to climb at maximum RC (Roskam Table 9.6).

    The climb is divided into ``n_steps`` altitude increments. RC_max, speed
    and fuel flow are evaluated at each altitude node with the weight
    *predicted* from the fuel burnt so far (Roskam Step 3: "the average weight
    may have to be iterated"), and averaged over each step (trapezoidal rule)
    to obtain the time, horizontal distance and fuel increments.

    If ``accelerated`` is True, the RC is corrected by the acceleration factor
    of the best-RC speed schedule (Roskam Sec. 9.5).

    Raises
    ------
    InfeasibleFlightConditionError
        If RC_max falls below ``min_rate`` [m/s] before ``h_end``: the target
        altitude is at (or above) the ceiling and would take an infinite time.
    """
    ensure_in_range("h_start", h_start, -500.0, H_MAX)
    ensure_in_range("h_end", h_end, -500.0, H_MAX)
    if not h_end > h_start:
        raise InputError("h_end must be greater than h_start.")
    if int(n_steps) != n_steps or n_steps < 2:
        raise InputError("n_steps must be an integer >= 2.")
    W0 = aircraft.weight_or_default(weight)
    hs = np.linspace(h_start, h_end, int(n_steps) + 1)
    prop = aircraft.propulsion

    def best(h, W):
        """RC_max (optionally acceleration-corrected), best-RC speed and fuel flow at (h, W)."""
        pt = max_rate_of_climb(aircraft, h, W, config, throttle, n_operative, delta_T)
        rc = pt.rate
        if accelerated:
            def sched(hh):
                return max_rate_of_climb(aircraft, hh, W, config, throttle, n_operative, delta_T).V
            rc = rc / (1.0 + acceleration_factor(sched, h, dh=50.0))
        if rc < min_rate:
            raise InfeasibleFlightConditionError(
                f"RC_max = {rc:.3f} m/s at h = {h:.0f} m is below {min_rate} m/s: "
                "the requested altitude is at or above the ceiling.")
        ff = float(prop.fuel_weight_flow_available(pt.V, h, throttle, n_operative, delta_T))
        return rc, pt.V, ff

    # Fail fast: the top of the climb is the most critical point (RC_max decreases with h)
    best(hs[-1], W0)

    rc_a, V_a, ff_a = best(hs[0], W0)
    t, s, fuel, Ws, rcs, Vs = [0.0], [0.0], [0.0], [W0], [rc_a], [V_a]
    W = W0
    for h1, h2 in zip(hs[:-1], hs[1:]):
        dh = h2 - h1
        W_pred = W - ff_a * dh / rc_a                  # predictor: weight at the end of the step
        rc_b, V_b, ff_b = best(h2, W_pred)
        dt = dh / (0.5 * (rc_a + rc_b))                # trapezoidal average of RC
        dWf = 0.5 * (ff_a + ff_b) * dt
        V_avg = 0.5 * (V_a + V_b)
        gamma = np.arcsin(np.clip(0.5 * (rc_a + rc_b) / V_avg, -1, 1))
        W -= dWf
        t.append(t[-1] + dt)
        s.append(s[-1] + V_avg * np.cos(gamma) * dt)
        fuel.append(fuel[-1] + dWf)
        Ws.append(W)
        rcs.append(rc_b)
        Vs.append(V_b)
        rc_a, V_a, ff_a = rc_b, V_b, ff_b
    return ClimbProfile(altitude=hs, time=np.array(t), distance=np.array(s),
                        fuel_weight=np.array(fuel), weight=np.array(Ws),
                        rate_of_climb=np.array(rcs), speed=np.array(Vs))


def time_to_climb_linear(rc_sea_level: float, absolute_ceiling: float, altitude) -> float:
    """Time to climb with a linear RC_max(h) law (Roskam Eqns 9.73-9.74) [s].

    ``t = (h_abs / RC_0) ln(1 / (1 - h / h_abs))``
    """
    rc0 = ensure_positive("rc_sea_level", rc_sea_level)
    h_abs = ensure_positive("absolute_ceiling", absolute_ceiling)
    h = np.asarray(ensure_non_negative("altitude", altitude))
    if np.any(h >= h_abs):
        raise InfeasibleFlightConditionError("The absolute ceiling cannot be reached in finite time.")
    return h_abs / rc0 * np.log(1.0 / (1.0 - h / h_abs))


def max_rate_of_climb_vs_altitude(aircraft: Aircraft, altitudes, weight: Optional[float] = None,
                                  config: str = "clean", throttle: float = 1.0, n_operative=None,
                                  delta_T: float = 0.0) -> dict:
    """RC_max, best-RC speed and best climb angle versus altitude (for plots)."""
    hs = np.atleast_1d(np.asarray(altitudes, dtype=float))
    rc, V, gam = [], [], []
    for h in hs:
        pt = max_rate_of_climb(aircraft, h, weight, config, throttle, n_operative, delta_T)
        rc.append(pt.rate)
        V.append(pt.V)
        gam.append(max_climb_angle(aircraft, h, weight, config, throttle, n_operative, delta_T).gamma)
    return {"altitude": hs, "RC_max": np.array(rc), "V_best_RC": np.array(V),
            "gamma_max": np.array(gam)}


__all__ = [
    "SERVICE_CEILING_RC", "CRUISE_CEILING_RC", "rate_of_climb_from_forces", "rate_of_climb_from_power",
    "jet_best_climb_lift_coefficient", "ClimbPoint", "jet_max_rate_of_climb_parabolic",
    "jet_max_climb_angle_parabolic", "propeller_max_rate_of_climb_parabolic",
    "propeller_steep_climb_sin_gamma", "rate_of_climb", "steep_climb", "max_rate_of_climb",
    "max_climb_angle", "acceleration_factor", "acceleration_factor_constant_eas",
    "acceleration_factor_constant_mach", "acceleration_factor_roskam_troposphere",
    "accelerated_rate_of_climb", "Ceilings", "ceiling_for_rate", "ceilings", "ClimbProfile",
    "time_to_climb", "time_to_climb_linear", "max_rate_of_climb_vs_altitude",
]
