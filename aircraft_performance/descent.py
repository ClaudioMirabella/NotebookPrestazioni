"""
Descent performance: unpowered glide, powered descent and drift-down.

Unpowered glide (Roskam & Lan, Section 8.2)
-------------------------------------------
With T = 0 the equilibrium along and normal to the flight path gives::

    tan(gamma) = CD / CL                        (Eqn 8.15)  glide angle (positive down)
    V = sqrt(2 W / (rho S sqrt(CL^2 + CD^2)))   (Eqn 8.16)  exact
    RD = V sin(gamma)                           (Eqn 8.21)  rate of descent (sink rate)

* Minimum glide angle (longest distance) at point E:
  ``tan(gamma_min) = 1/E_max`` and ``R_max = h E_max`` (Eqns 8.27-8.28).
  The glide *distance* does not depend on weight or altitude.
* Minimum sink rate (longest time aloft) at point P, maximum CL^3/CD^2
  (Eqns 8.24-8.31). Since density increases while descending, the true time
  aloft must be integrated in altitude (Roskam Problem 8.5): see
  :func:`glide_time`.
* Wind shifts the origin of the hodograph (Roskam Sec. 8.2.4.4):
  :func:`best_glide_with_wind` finds the speed-to-fly that maximises the
  ground distance with head/tail wind and vertical air motion.

Powered descent and drift-down (Roskam Sections 9.2.3, 9.3.3, 9.4.2)
---------------------------------------------------------------------
``RD = (D - T) V / W``: with one engine inoperative (OEI) a multi-engine
aircraft cruising above its OEI ceiling must *drift down*. The drift-down
time, distance and fuel are integrated numerically down to the OEI ceiling
(where RC = 100 ft/min), with a minimum practical descent rate of 100 ft/min
(Roskam Table 9.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad

from ._numerics import maximize_on_interval
from .aerodynamics import ParabolicDragPolar
from .aircraft import Aircraft
from .atmosphere import H_MAX, density, isa
from .climb import ceiling_for_rate, max_rate_of_climb
from .errors import ConvergenceError, InfeasibleFlightConditionError, InputError
from .units import FPM
from .validation import ensure_in_range, ensure_positive, ensure_result_finite


# ---------------------------------------------------------------------------
# Glide at a given lift coefficient
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GlideState:
    """Steady glide condition at one lift coefficient."""

    CL: float
    CD: float
    gamma: float      #: glide angle, positive downwards [rad]
    V: float          #: true airspeed [m/s]
    sink_rate: float  #: rate of descent RD > 0 [m/s]

    @property
    def gamma_deg(self) -> float:
        return float(np.degrees(self.gamma))

    @property
    def horizontal_speed(self) -> float:
        """V cos(gamma) [m/s]."""
        return self.V * np.cos(self.gamma)

    @property
    def glide_ratio(self) -> float:
        """Horizontal distance per unit height lost = 1/tan(gamma) = CL/CD."""
        return 1.0 / np.tan(self.gamma)


def glide_state(weight, wing_area, rho, CL, polar: ParabolicDragPolar) -> GlideState:
    """Exact steady glide at lift coefficient CL (Roskam Eqns 8.15-8.16, 8.21)."""
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    CL = ensure_positive("CL", CL)
    CD = polar.drag_coefficient(CL)
    gamma = float(np.arctan(CD / CL))
    CR = np.hypot(CL, CD)                    # resultant force coefficient (Eqn 8.9)
    V = float(np.sqrt(2.0 * W / (rho * S * CR)))
    return GlideState(CL=float(CL), CD=float(CD), gamma=gamma, V=V, sink_rate=V * np.sin(gamma))


def glide_polar(weight, wing_area, rho, polar: ParabolicDragPolar, CL_values) -> dict:
    """Hodograph (speed polar) data: horizontal speed vs rate of descent (Roskam Fig. 8.7-8.8)."""
    CLs = np.atleast_1d(ensure_positive("CL_values", CL_values))
    states = [glide_state(weight, wing_area, rho, c, polar) for c in CLs]
    return {
        "CL": CLs,
        "V": np.array([s.V for s in states]),
        "V_horizontal": np.array([s.horizontal_speed for s in states]),
        "sink_rate": np.array([s.sink_rate for s in states]),
        "gamma": np.array([s.gamma for s in states]),
    }


def best_glide(weight, wing_area, rho, polar: ParabolicDragPolar) -> GlideState:
    """Minimum glide angle condition (point E of the polar, Roskam Eqns 8.27, 8.29)."""
    return glide_state(weight, wing_area, rho, polar.point_E.CL, polar)


def minimum_sink(weight, wing_area, rho, polar: ParabolicDragPolar) -> GlideState:
    """Minimum rate-of-descent condition (point P of the polar, Roskam Eqn 8.30)."""
    return glide_state(weight, wing_area, rho, polar.point_P.CL, polar)


def max_glide_distance(height_loss, polar: ParabolicDragPolar) -> float:
    """Maximum still-air glide distance R = h E_max [m] (Roskam Eqn 8.28, exact for constant CL)."""
    h = ensure_positive("height_loss", height_loss)
    return h * polar.max_lift_to_drag


def glide_time(weight, wing_area, polar: ParabolicDragPolar, h_start: float, h_end: float = 0.0,
               CL: Optional[float] = None, delta_T: float = 0.0) -> float:
    """Time [s] to glide from h_start to h_end at constant CL, integrating density with altitude.

    ``t = integral dh / RD(h)`` with ``RD(h) = RD_ref sqrt(rho_ref / rho(h))``.
    By default CL is that of minimum sink (point P), giving the maximum time
    aloft (Roskam Problem 8.5). Because the sink rate scales with
    ``1/sqrt(rho)``, it *decreases* while descending into denser air: the
    integrated time is therefore *longer* than the estimate of Roskam Eqn
    (8.31) evaluated with the density of the starting altitude.
    """
    ensure_in_range("h_start", h_start, -500.0, H_MAX)
    ensure_in_range("h_end", h_end, -500.0, H_MAX)
    if not h_start > h_end:
        raise InputError("h_start must be above h_end.")
    CL = polar.point_P.CL if CL is None else ensure_positive("CL", CL)

    def inv_sink(h):
        return 1.0 / glide_state(weight, wing_area, float(density(h, delta_T)), CL, polar).sink_rate

    t, err = quad(inv_sink, h_end, h_start, limit=200)
    return ensure_result_finite("glide time", t)


def glide_time_constant_density(weight, wing_area, rho, polar: ParabolicDragPolar, height_loss) -> float:
    """Roskam Eqn (8.31): maximum time aloft using a single (constant) density [s]."""
    h = ensure_positive("height_loss", height_loss)
    return h / minimum_sink(weight, wing_area, rho, polar).sink_rate


def best_glide_with_wind(weight, wing_area, rho, polar: ParabolicDragPolar, CL_max: float,
                         headwind: float = 0.0, updraft: float = 0.0) -> dict:
    """Speed-to-fly for maximum ground distance with wind (Roskam Sec. 8.2.4.4).

    Parameters
    ----------
    headwind : float
        Horizontal wind component against the direction of flight [m/s]
        (negative = tailwind).
    updraft : float
        Vertical air velocity [m/s] (positive = rising air, negative = sink).

    Returns a dictionary with the optimal ``state`` (GlideState, air-relative),
    the ground-relative glide ratio and the ground speed.

    Raises
    ------
    InfeasibleFlightConditionError
        If the aircraft cannot make progress over the ground (headwind too
        strong), or the updraft is so strong that the aircraft climbs
        (a glide *ratio* is then meaningless).
    """
    hw = float(headwind)
    wz = float(updraft)

    def ground_ratio(CL):
        s = glide_state(weight, wing_area, rho, CL, polar)
        vg = s.horizontal_speed - hw
        net_sink = s.sink_rate - wz
        if vg <= 0.0 or net_sink <= 0.0:
            return -np.inf
        return vg / net_sink

    CL_lo = 0.05 * polar.point_E.CL
    try:
        CL_opt, ratio = maximize_on_interval(ground_ratio, CL_lo, CL_max, n_grid=400, log_spacing=True)
    except ConvergenceError:
        ratio = -np.inf     # no speed gives progress over the ground
    if not np.isfinite(ratio):
        raise InfeasibleFlightConditionError(
            "With this wind the aircraft either cannot progress over the ground "
            "or does not descend at all.")
    s = glide_state(weight, wing_area, rho, CL_opt, polar)
    return {"state": s, "ground_glide_ratio": ratio, "ground_speed": s.horizontal_speed - hw}


# ---------------------------------------------------------------------------
# Powered descent and drift-down
# ---------------------------------------------------------------------------
def rate_of_descent(aircraft: Aircraft, V, altitude: float, weight: Optional[float] = None,
                    config: str = "clean", throttle: float = 0.0, n_operative=None,
                    delta_T: float = 0.0, extra_CD0: float = 0.0):
    """Rate of descent RD = (D - T) V / W [m/s] (Roskam Eqns 9.18-9.20). Negative means climbing."""
    from .climb import rate_of_climb
    return -rate_of_climb(aircraft, V, altitude, weight, config, throttle, n_operative, delta_T, extra_CD0)


def minimum_rate_of_descent(aircraft: Aircraft, altitude: float, weight: Optional[float] = None,
                            config: str = "clean", throttle: float = 0.0, n_operative=None,
                            delta_T: float = 0.0, extra_CD0: float = 0.0) -> dict:
    """Minimum RD and the corresponding speed, i.e. the maximum of (T - D) V / W."""
    pt = max_rate_of_climb(aircraft, altitude, weight, config, throttle, n_operative, delta_T, extra_CD0)
    return {"V": pt.V, "rate_of_descent": -pt.rate, "gamma": -pt.gamma}


@dataclass(frozen=True)
class DescentProfile:
    """Result of a descent (drift-down) integration."""

    altitude: np.ndarray       #: [m], decreasing
    time: np.ndarray           #: cumulated time [s]
    distance: np.ndarray       #: cumulated horizontal distance [m]
    fuel_weight: np.ndarray    #: cumulated fuel burnt [N]
    rate_of_descent: np.ndarray  #: RD actually flown [m/s]
    speed: np.ndarray          #: TAS [m/s]
    final_altitude: float      #: altitude where the descent ends [m]

    @property
    def total_time(self) -> float:
        return float(self.time[-1])

    @property
    def total_distance(self) -> float:
        return float(self.distance[-1])

    @property
    def total_fuel_weight(self) -> float:
        return float(self.fuel_weight[-1])


def drift_down(aircraft: Aircraft, h_start: float, weight: Optional[float] = None,
               n_operative: Optional[int] = None, config: str = "clean", throttle: float = 1.0,
               extra_CD0: float = 0.0, delta_T: float = 0.0,
               oei_ceiling_rate: float = 100.0 * FPM, min_descent_rate: float = 100.0 * FPM,
               n_steps: int = 30) -> DescentProfile:
    """One-engine-inoperative drift-down from ``h_start`` to the OEI ceiling (Roskam Sec. 9.4.2).

    At each altitude the aircraft flies at the speed of minimum rate of
    descent with the remaining engines at ``throttle``; where the minimum
    RD is smaller than ``min_descent_rate`` (near the OEI ceiling) the latter
    is used, as done in Roskam Table 9.8. ``extra_CD0`` models the drag of the
    windmilling engine and the trim drag (Roskam Fig. 9.27).

    Raises
    ------
    InputError
        For a single-engine aircraft (use the glide functions instead).
    InfeasibleFlightConditionError
        If ``h_start`` is already below the OEI ceiling (no drift-down needed).
    """
    n_eng = aircraft.propulsion.n_engines
    if n_eng < 2:
        raise InputError("Drift-down with engines operating requires a multi-engine aircraft.")
    n_op = n_eng - 1 if n_operative is None else n_operative
    W0 = aircraft.weight_or_default(weight)
    h_ceiling = ceiling_for_rate(aircraft, oei_ceiling_rate, W0, config, throttle, n_op, delta_T, extra_CD0)
    if h_start <= h_ceiling:
        raise InfeasibleFlightConditionError(
            f"h_start = {h_start:.0f} m is below the OEI ceiling ({h_ceiling:.0f} m): no drift-down.")
    hs = np.linspace(h_start, h_ceiling, int(n_steps) + 1)
    prop = aircraft.propulsion
    W = W0
    t, s, fuel, rds, Vs = [0.0], [0.0], [0.0], [], []

    def node(h, W):
        pt = max_rate_of_climb(aircraft, h, W, config, throttle, n_op, delta_T, extra_CD0)
        rd = max(-pt.rate, min_descent_rate)
        ff = float(prop.fuel_weight_flow_available(pt.V, h, throttle, n_op, delta_T))
        return rd, pt.V, ff

    rd_a, V_a, ff_a = node(hs[0], W)
    rds.append(rd_a); Vs.append(V_a)
    for h2 in hs[1:]:
        rd_b, V_b, ff_b = node(h2, W)
        dh = hs[0] - hs[1]
        dt = dh / (0.5 * (rd_a + rd_b))
        dWf = 0.5 * (ff_a + ff_b) * dt
        W -= dWf
        t.append(t[-1] + dt)
        s.append(s[-1] + 0.5 * (V_a + V_b) * dt)
        fuel.append(fuel[-1] + dWf)
        rds.append(rd_b); Vs.append(V_b)
        rd_a, V_a, ff_a = rd_b, V_b, ff_b
    return DescentProfile(altitude=hs, time=np.array(t), distance=np.array(s),
                          fuel_weight=np.array(fuel), rate_of_descent=np.array(rds),
                          speed=np.array(Vs), final_altitude=float(h_ceiling))


def descent_at_constant_eas(aircraft: Aircraft, h_start: float, h_end: float, V_eas: float,
                            weight: Optional[float] = None, config: str = "clean",
                            throttle: float = 0.0, delta_T: float = 0.0, n_steps: int = 40) -> DescentProfile:
    """Descent at constant equivalent airspeed and fixed throttle (e.g. idle, throttle = 0).

    Integrates ``dt = -dh / RD`` with ``RD = (D - T) V / W``. The kinetic
    energy change due to the varying TAS is neglected (it is small in a descent).

    Raises
    ------
    InfeasibleFlightConditionError
        If at some altitude the aircraft does not descend with this throttle
        setting and speed (RD <= 0).
    """
    ensure_in_range("h_start", h_start, -500.0, H_MAX)
    ensure_in_range("h_end", h_end, -500.0, H_MAX)
    if not h_start > h_end:
        raise InputError("h_start must be above h_end.")
    V_eas = ensure_positive("V_eas", V_eas)
    W = aircraft.weight_or_default(weight)
    hs = np.linspace(h_start, h_end, int(n_steps) + 1)
    prop = aircraft.propulsion
    t, s, fuel, rds, Vs = [0.0], [0.0], [0.0], [], []
    for i, h in enumerate(hs):
        V = V_eas / np.sqrt(isa(h, delta_T).sigma)
        rd = float(rate_of_descent(aircraft, V, h, W, config, throttle, None, delta_T))
        if rd <= 0.0:
            raise InfeasibleFlightConditionError(
                f"At h = {h:.0f} m and EAS = {V_eas:.1f} m/s the aircraft does not descend "
                f"(RD = {rd:.2f} m/s): reduce throttle or increase drag.")
        rds.append(rd)
        Vs.append(V)
        if i > 0:
            dt = (hs[i - 1] - h) / (0.5 * (rds[-2] + rd))
            ff = 0.5 * (float(prop.fuel_weight_flow_available(Vs[-2], hs[i - 1], throttle, None, delta_T))
                        + float(prop.fuel_weight_flow_available(V, h, throttle, None, delta_T)))
            t.append(t[-1] + dt)
            s.append(s[-1] + 0.5 * (Vs[-2] + V) * dt)
            fuel.append(fuel[-1] + ff * dt)
    return DescentProfile(altitude=hs, time=np.array(t), distance=np.array(s),
                          fuel_weight=np.array(fuel), rate_of_descent=np.array(rds),
                          speed=np.array(Vs), final_altitude=float(h_end))


__all__ = [
    "GlideState", "glide_state", "glide_polar", "best_glide", "minimum_sink", "max_glide_distance",
    "glide_time", "glide_time_constant_density", "best_glide_with_wind", "rate_of_descent",
    "minimum_rate_of_descent", "DescentProfile", "drift_down", "descent_at_constant_eas",
]
