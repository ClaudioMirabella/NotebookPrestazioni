"""
Take-off and landing distances (Roskam & Lan, Chapter 10).

Take-off (Fig. 10.6): ``S_TO = S_NGR + S_R + S_TR + S_CL``
    ground roll with the nose wheel on the ground (0 -> V_R), rotation
    (V_R -> V_LOF), transition (circular arc) and climb to the screen height
    (35 ft for FAR 25, 50 ft for FAR 23 and military aircraft).

Landing (Fig. 10.29): ``S_L = S_LA + S_LR + S_LNGR``
    air distance from the 50 ft screen (approach + flare), free roll while
    the nose wheel is lowered, braked ground roll.

Ground-roll equation of motion (Eqns 10.5 and 10.72)::

    dV/dt = g [ (T/W - mu) - (CD_g - mu CL_g) q / (W/S) - phi ]

where ``CD_g, CL_g`` are the coefficients in ground effect, ``mu`` the rolling
(or braking) friction coefficient and ``phi`` the runway slope (positive
uphill). Three families of methods are implemented:

* :func:`takeoff_roskam_approximate` and :func:`landing_roskam_approximate`:
  the *approximate analytical* methods of Secs. 10.3.3 and 10.6.3, which
  reproduce Roskam Examples 10.2 and 10.3 step by step;
* :func:`takeoff_ground_roll_numerical` / :func:`landing_ground_roll_numerical`
  and the aircraft-level :func:`takeoff_distance`: numerical integration of
  the equation of motion (the "accurate" method of Secs. 10.3.1, 10.6.1),
  including wind and runway slope;
* statistical correlations (Secs. 10.3.2, 10.6.2) for FAR 23 / FAR 25
  aircraft, and Torenbeek's balanced field length (Eqn 10.57).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from . import aerodynamics as aero
from . import units as u
from .aerodynamics import ParabolicDragPolar
from .aircraft import Aircraft
from .atmosphere import isa
from .errors import InfeasibleFlightConditionError, InputError, PerformanceWarning
from .units import G0
from .validation import (ensure_in_range, ensure_non_negative, ensure_positive,
                         ensure_result_finite)

# ---------------------------------------------------------------------------
# Regulatory and empirical data
# ---------------------------------------------------------------------------
#: Screen (obstacle) heights [m] (Roskam Sec. 10.1 and 10.4).
SCREEN_HEIGHT = {
    "FAR25_takeoff": 35.0 * u.FT,
    "FAR23_takeoff": 50.0 * u.FT,
    "military_takeoff": 50.0 * u.FT,
    "landing": 50.0 * u.FT,
}

#: Typical rolling friction coefficients (Roskam Table 10.1b).
ROLLING_FRICTION = {
    "concrete / asphalt": 0.025,
    "hard turf": 0.05,
    "short grass": 0.05,
    "long grass": 0.10,
    "soft ground": 0.20,
}

#: Typical braking friction coefficients, (min, max) (Roskam Table 10.3).
#: Design calculations often use 0.3-0.5 to account for real braking efficiency.
BRAKING_FRICTION = {
    "dry concrete / asphalt": (0.65, 0.80),
    "wet concrete / asphalt": (0.25, 0.75),
    "slush": (0.15, 0.40),
    "dry ice / packed snow": (0.05, 0.05),
}

#: Minimum second-segment OEI climb gradients (FAR 25.121, Roskam p. 474).
SECOND_SEGMENT_MIN_GRADIENT = {2: 0.024, 3: 0.027, 4: 0.030}

ThrustFunction = Callable[[float], float]


def linear_thrust_model(T_static: float, T_ref: float, V_ref: float) -> ThrustFunction:
    """Thrust varying linearly with speed, T(V) = T0 + (T_ref - T0) V / V_ref (Roskam Fig. 10.9)."""
    T0 = ensure_positive("T_static", T_static)
    T1 = ensure_non_negative("T_ref", T_ref)
    V1 = ensure_positive("V_ref", V_ref)

    def thrust(V):
        return T0 + (T1 - T0) * V / V1
    return thrust


def quadratic_thrust_model(k0: float, k1: float, k2: float) -> ThrustFunction:
    """Thrust T(V) = k0 - k1 V + k2 V^2 (Anderson Eqn 6.74) [N, V in m/s]."""
    ensure_positive("k0", k0)

    def thrust(V):
        return k0 - k1 * V + k2 * V ** 2
    return thrust


def optimum_ground_run_lift_coefficient(polar: ParabolicDragPolar, mu: float) -> float:
    """CL_g that minimises (CD_g - mu CL_g), i.e. the ground run: CL = mu / (2K) (Roskam Problem 10.4)."""
    mu = ensure_in_range("mu", mu, 0.0, 1.0)
    return mu / (2.0 * polar.K)


def ground_effect_induced_drag_reduction(CL, aspect_ratio, sigma_prime):
    """Decrease of induced drag in ground effect, (1 - sigma') CL^2 / (pi A) (Roskam Eqn 10.12, e = 1)."""
    AR = ensure_positive("aspect_ratio", aspect_ratio)
    sp = ensure_in_range("sigma_prime", sigma_prime, 0.0, 1.0)
    return (1.0 - sp) * np.asarray(CL) ** 2 / (np.pi * AR)


def ground_acceleration(V, weight, wing_area, rho, thrust, CL_ground, CD_ground, mu,
                        runway_slope: float = 0.0):
    """Acceleration along the runway [m/s^2] (Roskam Eqns 10.5, 10.72 with N_n = 0)."""
    W = ensure_positive("weight", weight)
    WS = W / ensure_positive("wing_area", wing_area)
    q = 0.5 * rho * np.asarray(V) ** 2
    return G0 * ((np.asarray(thrust) / W - mu) - (CD_ground - mu * CL_ground) * q / WS - runway_slope)


# ---------------------------------------------------------------------------
# Air distance after lift-off (shared by the take-off methods)
# ---------------------------------------------------------------------------
def transition_delta_CL(CL_max_TO: float, V_LOF_over_VS: float) -> float:
    """Incremental CL in a 'normal effort' transition (Roskam Eqn 10.42, V_mean = V_LOF)."""
    r = ensure_in_range("V_LOF_over_VS", V_LOF_over_VS, 1.0, 2.0)
    CLm = ensure_positive("CL_max_TO", CL_max_TO)
    return 0.5 * (r ** 2 - 1.0) * (CLm * (1.0 / r ** 2 - 0.53) + 0.38)


def _air_distance(W, S, rho, V_LOF, V_S, V_screen, CL_max_TO, thrust_LOF, drag_LOF, screen_height):
    """Transition and climb to screen, Roskam Eqns 10.38-10.47 and 10.52-10.53."""
    dCL = transition_delta_CL(CL_max_TO, V_LOF / V_S)
    R_TR = 2.0 * (W / S) / (rho * G0 * dCL)                     # Eqn 10.41
    sin_theta = (thrust_LOF - drag_LOF) / W                     # Eqn 10.44
    if sin_theta <= 0.0:
        raise InfeasibleFlightConditionError(
            "Drag exceeds thrust at lift-off: the aircraft cannot climb out (check weight/thrust).")
    theta = float(np.arcsin(min(sin_theta, 1.0)))
    h_TR = R_TR * (1.0 - np.cos(theta))                          # Eqn 10.47
    if h_TR < screen_height:
        S_TR = R_TR * np.sin(theta)                             # Eqn 10.45
        S_CL = (screen_height - h_TR) / np.tan(theta)           # Eqn 10.46
    else:
        # The screen is cleared during the transition: stop the arc at h = screen
        # (Roskam: "if h_TR > h_screen then S_CL = 0").
        S_TR = float(np.sqrt(R_TR ** 2 - (R_TR - screen_height) ** 2))
        S_CL = 0.0
    t_TR = S_TR / V_LOF                                           # Eqn 10.52
    t_CL = S_CL / (0.5 * (V_LOF + V_screen))                      # Eqn 10.53
    return dict(delta_CL=dCL, R_TR=R_TR, theta_climb=theta, h_TR=h_TR,
                S_TR=S_TR, S_CL=S_CL, t_TR=t_TR, t_CL=t_CL)


@dataclass(frozen=True)
class TakeoffResult:
    """Take-off distance components [m], speeds [m/s] and times [s]."""

    V_S: float
    V_R: float
    V_LOF: float
    S_NGR: float     #: ground roll, nose wheel on the ground
    S_R: float       #: rotation
    S_TR: float      #: transition
    S_CL: float      #: climb to the screen height
    t_NGR: float
    t_R: float
    t_TR: float
    t_CL: float
    delta_CL: float
    R_TR: float      #: transition radius
    theta_climb: float  #: climb-out angle [rad]
    h_TR: float      #: height at the end of the transition
    screen_height: float
    method: str

    @property
    def ground_distance(self) -> float:
        """S_G = S_NGR + S_R."""
        return self.S_NGR + self.S_R

    @property
    def air_distance(self) -> float:
        """S_A = S_TR + S_CL."""
        return self.S_TR + self.S_CL

    @property
    def total_distance(self) -> float:
        """Take-off distance to the screen height, S_TO (Roskam Eqn 10.1)."""
        return self.ground_distance + self.air_distance

    @property
    def total_time(self) -> float:
        """Time to take off, t_TO (Roskam Eqn 10.49)."""
        return self.t_NGR + self.t_R + self.t_TR + self.t_CL


def takeoff_roskam_approximate(weight, wing_area, rho, CL_max_TO: float, polar_TO: ParabolicDragPolar,
                               CL_ground: float, CD_ground: float, mu: float, thrust: ThrustFunction,
                               screen_height: float = SCREEN_HEIGHT["FAR23_takeoff"],
                               V_R_factor: float = 1.10, V_LOF_factor: float = 1.15,
                               V_screen_factor: float = 1.20, t_rotate: float = 1.0,
                               runway_slope: float = 0.0) -> TakeoffResult:
    """Roskam's approximate analytical take-off method, zero wind (Sec. 10.3.3, Example 10.2).

    Parameters
    ----------
    polar_TO : ParabolicDragPolar
        Take-off configuration polar *out of ground effect* (used at lift-off).
    CL_ground, CD_ground : float
        Lift and drag coefficients during the ground roll, *in ground effect*.
    thrust : callable
        Take-off thrust as a function of airspeed, T(V) [N].
    t_rotate : float
        Rotation time: ~1 s light aircraft, 2 s fighters, 3 s transports (p. 463).

    Notes
    -----
    Wind is not supported by this closed-form method (Roskam's Eqn 10.36 needs
    an iteration); use :func:`takeoff_distance` for take-offs with wind.
    """
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    mu = ensure_in_range("mu", mu, 0.0, 0.5)
    ensure_in_range("V_R_factor", V_R_factor, 1.0, 1.5)
    ensure_in_range("V_LOF_factor", V_LOF_factor, V_R_factor, 1.6)
    ensure_non_negative("t_rotate", t_rotate)
    V_S = float(aero.stall_speed(W, S, rho, CL_max_TO))
    V_R = V_R_factor * V_S
    V_LOF = V_LOF_factor * V_S
    V_2 = V_screen_factor * V_S

    def acc(V):
        return float(ground_acceleration(V, W, S, rho, thrust(V), CL_ground, CD_ground, mu, runway_slope))

    a0 = acc(0.0)                                                 # Eqn 10.27
    aR = acc(V_R)                                                 # Eqn 10.28
    if a0 <= 0.0 or aR <= 0.0:
        raise InfeasibleFlightConditionError(
            f"The aircraft does not accelerate up to V_R (a(0) = {a0:.2f}, a(V_R) = {aR:.2f} m/s^2).")
    ratio = aR / a0
    k = 1.0 if abs(ratio - 1.0) < 1e-9 else (1.0 - ratio) / np.log(1.0 / ratio)  # Eqn 10.30
    a_avg = k * a0
    S_NGR = V_R ** 2 / (2.0 * a_avg)                               # Eqn 10.29
    t_NGR = V_R / acc(V_R / np.sqrt(2.0))                          # Eqn 10.51
    S_R = 0.5 * (V_R + V_LOF) * t_rotate                           # Eqn 10.37
    CL_LOF = CL_max_TO / V_LOF_factor ** 2
    D_LOF = 0.5 * rho * V_LOF ** 2 * S * polar_TO.drag_coefficient(CL_LOF)
    air = _air_distance(W, S, rho, V_LOF, V_S, V_2, CL_max_TO, thrust(V_LOF), D_LOF, screen_height)
    return TakeoffResult(V_S=V_S, V_R=V_R, V_LOF=V_LOF, S_NGR=S_NGR, S_R=S_R, S_TR=air["S_TR"],
                         S_CL=air["S_CL"], t_NGR=t_NGR, t_R=float(t_rotate), t_TR=air["t_TR"],
                         t_CL=air["t_CL"], delta_CL=air["delta_CL"], R_TR=air["R_TR"],
                         theta_climb=air["theta_climb"], h_TR=air["h_TR"],
                         screen_height=float(screen_height), method="Roskam approximate analytical")


def takeoff_ground_roll_numerical(weight, wing_area, rho, CL_ground: float, CD_ground: float,
                                  mu: float, thrust: ThrustFunction, V_end: float,
                                  headwind: float = 0.0, runway_slope: float = 0.0,
                                  t_max: float = 300.0) -> dict:
    """Integrate the ground run from rest to the airspeed ``V_end`` (Roskam Eqn 10.8).

    The state is (airspeed V, ground distance s); at brake release the ground
    speed is zero, i.e. the airspeed equals the headwind. The ground distance
    grows with the ground speed ``V - headwind``.

    Returns a dictionary with ``distance``, ``time`` and the time histories
    ``t, V, s``.

    Raises
    ------
    InfeasibleFlightConditionError
        If the acceleration vanishes before ``V_end`` (not enough thrust) or
        the run lasts more than ``t_max``.
    """
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    V_end = ensure_positive("V_end", V_end)
    hw = float(headwind)
    if hw >= V_end:
        raise InputError("The headwind is larger than the target airspeed.")

    def rhs(t, y):
        V = y[0]
        a = ground_acceleration(V, W, S, rho, thrust(V), CL_ground, CD_ground, mu, runway_slope)
        return [float(a), V - hw]

    def reached(t, y):
        return y[0] - V_end
    reached.terminal = True
    reached.direction = 1

    def stalled(t, y):
        return float(ground_acceleration(y[0], W, S, rho, thrust(y[0]), CL_ground, CD_ground, mu,
                                         runway_slope)) - 1e-6
    stalled.terminal = True
    stalled.direction = -1

    if stalled(0.0, [max(hw, 0.0)]) <= 0.0:
        raise InfeasibleFlightConditionError("The aircraft does not accelerate from rest.")
    sol = solve_ivp(rhs, (0.0, t_max), [max(hw, 0.0), 0.0], events=(reached, stalled),
                    max_step=0.5, rtol=1e-8, atol=1e-8, dense_output=False)
    if sol.t_events[0].size == 0:
        raise InfeasibleFlightConditionError(
            f"The airspeed {V_end:.1f} m/s is never reached (max {sol.y[0].max():.1f} m/s): "
            "thrust too low or runway too steep.")
    return {"distance": float(sol.y_events[0][0][1]), "time": float(sol.t_events[0][0]),
            "t": sol.t, "V": sol.y[0], "s": sol.y[1]}


def takeoff_distance(aircraft: Aircraft, altitude: float = 0.0, weight: Optional[float] = None,
                     mu: float = ROLLING_FRICTION["concrete / asphalt"],
                     CL_ground: Optional[float] = None, headwind: float = 0.0,
                     runway_slope: float = 0.0, delta_T: float = 0.0,
                     V_R_factor: float = 1.10, V_LOF_factor: float = 1.15,
                     V_screen_factor: float = 1.20, t_rotate: Optional[float] = None,
                     screen_height: Optional[float] = None, thrust: Optional[ThrustFunction] = None,
                     config: str = "takeoff") -> TakeoffResult:
    """Take-off distance of an :class:`Aircraft` (numerical ground roll + Roskam air distance).

    * Ground roll to V_R integrated numerically with wind and slope (Sec. 10.3.1.1),
      with the induced drag reduced by ground effect (Eqns 10.12-10.13) when
      ``aircraft.wing_height_above_ground`` is known.
    * Rotation at the average of V_R and V_LOF for ``t_rotate`` (Eqn 10.37).
    * Transition and climb to the screen with Roskam's geometric model.

    Defaults: ``CL_ground`` = optimum ground-run CL (Roskam Problem 10.4),
    limited to 80 % of CL_max; ``t_rotate`` = 1 s for propeller aircraft and 3 s
    for jets; ``screen_height`` = 35 ft for jets (FAR 25), 50 ft otherwise.
    """
    W = aircraft.weight_or_default(weight)
    conf = aircraft.config(config)
    atm = isa(altitude, delta_T)
    rho = atm.density
    S = aircraft.wing_area
    mu = ensure_in_range("mu", mu, 0.0, 0.5)
    if thrust is None:
        def thrust(V):
            return float(aircraft.propulsion.thrust_available(V, altitude, 1.0, None, delta_T))
    if t_rotate is None:
        t_rotate = 3.0 if aircraft.is_jet else 1.0
    if screen_height is None:
        screen_height = SCREEN_HEIGHT["FAR25_takeoff"] if aircraft.is_jet else SCREEN_HEIGHT["FAR23_takeoff"]
    polar = conf.polar
    if CL_ground is None:
        CL_ground = min(optimum_ground_run_lift_coefficient(polar, mu), 0.8 * conf.CL_max)
    CL_ground = ensure_in_range("CL_ground", CL_ground, 0.0, conf.CL_max)
    sigma_p = 1.0
    if aircraft.wing_height_above_ground is not None:
        hb = aircraft.wing_height_above_ground / aircraft.wing_span
        sigma_p = float(aero.ground_effect_factor(np.clip(hb, 0.033, 0.25)))
        if not 0.033 <= hb <= 0.25:
            warnings.warn(f"h/b = {hb:.3f} outside the validity range of Roskam Eqn 10.13; clipped.",
                          PerformanceWarning, stacklevel=2)
    CD_ground = polar.CD0 + sigma_p * polar.K * CL_ground ** 2

    V_S = float(aero.stall_speed(W, S, rho, conf.CL_max))
    V_R, V_LOF, V_2 = V_R_factor * V_S, V_LOF_factor * V_S, V_screen_factor * V_S
    roll = takeoff_ground_roll_numerical(W, S, rho, CL_ground, CD_ground, mu, thrust, V_R,
                                         headwind, runway_slope)
    S_R = (0.5 * (V_R + V_LOF) - float(headwind)) * t_rotate
    D_LOF = 0.5 * rho * V_LOF ** 2 * S * polar.drag_coefficient(conf.CL_max / V_LOF_factor ** 2)
    air = _air_distance(W, S, rho, V_LOF, V_S, V_2, conf.CL_max, thrust(V_LOF), D_LOF, screen_height)
    # With wind the air distance over the ground scales with ground speed / airspeed
    wind_factor = (V_LOF - float(headwind)) / V_LOF
    return TakeoffResult(V_S=V_S, V_R=V_R, V_LOF=V_LOF, S_NGR=roll["distance"], S_R=S_R,
                         S_TR=air["S_TR"] * wind_factor, S_CL=air["S_CL"] * wind_factor,
                         t_NGR=roll["time"], t_R=float(t_rotate), t_TR=air["t_TR"], t_CL=air["t_CL"],
                         delta_CL=air["delta_CL"], R_TR=air["R_TR"], theta_climb=air["theta_climb"],
                         h_TR=air["h_TR"], screen_height=float(screen_height),
                         method="numerical ground roll + Roskam air distance")


# ---------------------------------------------------------------------------
# Statistical methods (Roskam Sec. 10.3.2 and 10.6.2) -- SI in, SI out
# ---------------------------------------------------------------------------
def takeoff_parameter_far23(wing_loading, power_loading, sigma, CL_max_TO) -> float:
    """TOP23 = (W/S)(W/P) / (sigma CL_max_TO) in lb^2/(ft^2 hp) (Roskam Eqn 10.22).

    ``wing_loading`` in N/m^2, ``power_loading`` in N/W (SI inputs).
    """
    WS = ensure_positive("wing_loading", wing_loading) / u.PSF
    WP = ensure_positive("power_loading", power_loading) * u.HP / u.LBF
    return WS * WP / (ensure_positive("sigma", sigma) * ensure_positive("CL_max_TO", CL_max_TO))


def takeoff_distance_far23_statistical(wing_loading, power_loading, sigma, CL_max_TO) -> dict:
    """FAR 23 take-off ground run and distance over 50 ft [m] (Roskam Eqns 10.20-10.23)."""
    top = takeoff_parameter_far23(wing_loading, power_loading, sigma, CL_max_TO)
    S_G = (4.9 * top + 0.009 * top ** 2) * u.FT
    return {"TOP23": top, "ground_run": S_G, "takeoff_distance": 1.66 * S_G}


def takeoff_parameter_far25(wing_loading, thrust_to_weight, sigma, CL_max_TO) -> float:
    """TOP25 = (W/S) / (sigma CL_max_TO (T/W)) in lb/ft^2 (Roskam Eqn 10.25)."""
    WS = ensure_positive("wing_loading", wing_loading) / u.PSF
    TW = ensure_positive("thrust_to_weight", thrust_to_weight)
    return WS / (ensure_positive("sigma", sigma) * ensure_positive("CL_max_TO", CL_max_TO) * TW)


def takeoff_field_length_far25_statistical(wing_loading, thrust_to_weight, sigma, CL_max_TO) -> dict:
    """FAR 25 take-off field length [m], S_TOFL = 37.5 TOP25 ft (Roskam Eqn 10.24)."""
    top = takeoff_parameter_far25(wing_loading, thrust_to_weight, sigma, CL_max_TO)
    return {"TOP25": top, "field_length": 37.5 * top * u.FT}


def landing_distance_far23_statistical(stall_speed_landing) -> dict:
    """FAR 23 landing ground roll and distance from 50 ft [m] (Roskam Eqns 10.100-10.101)."""
    Vs_kt = ensure_positive("stall_speed_landing", stall_speed_landing) / u.KT
    S_LG = 0.265 * Vs_kt ** 2 * u.FT
    return {"ground_roll": S_LG, "landing_distance": 1.938 * S_LG}


def landing_field_length_far25_statistical(approach_speed) -> dict:
    """FAR 25 landing field length [m], S_FL = 0.3 V_A^2 (V_A in kt, S in ft) (Roskam Fig. 10.38)."""
    Va_kt = ensure_positive("approach_speed", approach_speed) / u.KT
    S_FL = 0.3 * Va_kt ** 2 * u.FT
    return {"field_length": S_FL, "landing_distance": 0.6 * S_FL}


# ---------------------------------------------------------------------------
# Balanced field length (Torenbeek, Roskam Eqn 10.57)
# ---------------------------------------------------------------------------
def mean_takeoff_thrust_jet(static_thrust, bypass_ratio) -> float:
    """Mean take-off thrust of a jet, 0.75 T_TO (5 + BPR)/(4 + BPR) (Roskam Eqn 10.58) [N]."""
    T = ensure_positive("static_thrust", static_thrust)
    lam = ensure_non_negative("bypass_ratio", bypass_ratio)
    return 0.75 * (5.0 + lam) / (4.0 + lam) * T


def mean_takeoff_thrust_propeller(takeoff_power, sigma, n_engines: int, propeller_diameter) -> float:
    """Mean take-off thrust of a propeller aircraft (Roskam Eqn 10.59) [N].

    ``T = 5.75 P (sigma N_e D_p^2 / P)^(1/3)`` with P in hp, D_p in ft and T in lb
    (the constant 5.75 is dimensional); inputs and output are in SI units.
    """
    P_hp = ensure_positive("takeoff_power", takeoff_power) / u.HP
    D_ft = ensure_positive("propeller_diameter", propeller_diameter) / u.FT
    s = ensure_positive("sigma", sigma)
    T_lb = 5.75 * P_hp * (s * n_engines * D_ft ** 2 / P_hp) ** (1.0 / 3.0)
    return T_lb * u.LBF


def balanced_field_length_torenbeek(weight, wing_area, rho, CL_max_TO, mean_thrust,
                                    n_engines: int, second_segment_gradient,
                                    screen_height: float = SCREEN_HEIGHT["FAR25_takeoff"],
                                    CL_2: Optional[float] = None, flaps_deflected: bool = True) -> dict:
    """Torenbeek's balanced field length for FAR 25 aircraft [m] (Roskam Eqn 10.57).

    ``BFL = 0.863/(1 + 2.3 dgamma2) (W/S/(rho g CL2) + h_screen) (1/(T/W - mu') + 2.7) + 655 ft / sqrt(sigma)``

    with ``dgamma2 = gamma2 - gamma2_min``, ``mu' = 0.01 CL_max_TO + 0.02``
    (flaps in take-off position), ``CL2 = 0.694 CL_max_TO`` (V2 = 1.2 V_S).
    """
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    if n_engines not in SECOND_SEGMENT_MIN_GRADIENT:
        raise InputError("Torenbeek's BFL applies to 2-, 3- and 4-engine aircraft.")
    g2 = ensure_in_range("second_segment_gradient", second_segment_gradient, -0.2, 0.5)
    dg2 = g2 - SECOND_SEGMENT_MIN_GRADIENT[n_engines]
    if dg2 < 0.0:
        warnings.warn(f"The OEI second-segment gradient {g2:.4f} is below the FAR 25.121 minimum "
                      f"{SECOND_SEGMENT_MIN_GRADIENT[n_engines]}: the aircraft is not certifiable "
                      "at this weight.", PerformanceWarning, stacklevel=2)
    CL2 = 0.694 * CL_max_TO if CL_2 is None else ensure_positive("CL_2", CL_2)
    mu_p = 0.01 * CL_max_TO + 0.02 if flaps_deflected else 0.0
    TW = ensure_positive("mean_thrust", mean_thrust) / W
    if TW <= mu_p:
        raise InfeasibleFlightConditionError("Mean thrust-to-weight ratio is below mu': no take-off possible.")
    sigma = rho / isa(0.0).density
    bfl = (0.863 / (1.0 + 2.3 * dg2) * (W / S / (rho * G0 * CL2) + screen_height)
           * (1.0 / (TW - mu_p) + 2.7) + 655.0 * u.FT / np.sqrt(sigma))
    return {"BFL": ensure_result_finite("BFL", bfl), "delta_gamma2": dg2, "mu_prime": mu_p, "CL2": CL2}


# ---------------------------------------------------------------------------
# Landing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LandingResult:
    """Landing distance components [m], speeds [m/s] and times [s]."""

    V_S: float
    V_A: float        #: approach speed
    V_FL: float       #: flare speed
    V_TD: float       #: touchdown speed
    gamma_A: float    #: approach angle (positive down) [rad]
    R_flare: float
    h_flare: float
    S_LA: float       #: air distance (descent from screen + flare)
    S_LR: float       #: free roll while lowering the nose wheel
    S_LNGR: float     #: braked ground roll
    t_LA: float
    t_LR: float
    t_LNGR: float
    method: str

    @property
    def ground_distance(self) -> float:
        return self.S_LR + self.S_LNGR

    @property
    def total_distance(self) -> float:
        """Landing distance from the 50 ft screen, S_L (Roskam Eqn 10.61)."""
        return self.S_LA + self.S_LR + self.S_LNGR

    @property
    def far25_field_length(self) -> float:
        """FAR 25 landing field length S_L / 0.6 (Roskam Eqn 10.103)."""
        return self.total_distance / 0.6

    @property
    def total_time(self) -> float:
        return self.t_LA + self.t_LR + self.t_LNGR


def approach_angle(CL_approach, CD_approach, thrust_to_weight) -> float:
    """Approach flight-path angle (positive down) gamma = CD/CL - T/W [rad] (Roskam Eqn 10.81)."""
    g = ensure_positive("CD_approach", CD_approach) / ensure_positive("CL_approach", CL_approach) \
        - ensure_non_negative("thrust_to_weight", thrust_to_weight)
    if g <= 0.0:
        raise InfeasibleFlightConditionError(
            "With this approach thrust the aircraft does not descend (T/W >= CD/CL).")
    return float(g)


def landing_roskam_approximate(weight, wing_area, rho, CL_max_L: float, polar_L: ParabolicDragPolar,
                               mu_brake: float, CL_ground: float, CD_ground: float,
                               approach_thrust: Optional[float] = None,
                               approach_angle_deg: Optional[float] = None,
                               ground_thrust: float = 0.0, nose_load_ratio: float = 0.08,
                               mu_rolling: float = ROLLING_FRICTION["concrete / asphalt"],
                               V_A_factor: float = 1.3, V_FL_factor: float = 0.95,
                               V_TD_factor: float = 1.15, n_flare: float = 1.08,
                               t_free_roll: float = 1.0,
                               screen_height: float = SCREEN_HEIGHT["landing"]) -> LandingResult:
    """Roskam's approximate analytical landing method (Sec. 10.6.3, Example 10.3).

    Give either ``approach_thrust`` [N] (the approach angle follows from
    Eqn 10.81) or ``approach_angle_deg`` (typically 2.5-3 deg).
    ``ground_thrust`` is the idle thrust during the ground roll (negative for
    thrust reversers, not allowed for certification).
    """
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    mu_b = ensure_in_range("mu_brake", mu_brake, 0.0, 1.0)
    mu_g = ensure_in_range("mu_rolling", mu_rolling, 0.0, 0.5)
    Nn = ensure_in_range("nose_load_ratio", nose_load_ratio, 0.0, 0.3)
    n_fl = ensure_in_range("n_flare", n_flare, 1.01, 1.5)
    ensure_non_negative("t_free_roll", t_free_roll)
    V_S = float(aero.stall_speed(W, S, rho, CL_max_L))
    V_A = V_A_factor * V_S
    V_FL = V_FL_factor * V_A                                           # Eqn 10.108
    V_TD = V_TD_factor * V_S
    if (approach_thrust is None) == (approach_angle_deg is None):
        raise InputError("Give exactly one of 'approach_thrust' or 'approach_angle_deg'.")
    if approach_angle_deg is not None:
        gamma = float(np.radians(ensure_in_range("approach_angle_deg", approach_angle_deg, 0.5, 15.0)))
    else:
        CL_A = CL_max_L / V_A_factor ** 2
        gamma = approach_angle(CL_A, polar_L.drag_coefficient(CL_A), approach_thrust / W)
    R = V_FL ** 2 / (G0 * (n_fl - 1.0))                                # Eqn 10.107
    h_F = R * (1.0 - np.cos(gamma))                                     # Eqn 10.84
    if h_F < screen_height:
        S_LA = (screen_height - h_F) / np.tan(gamma) + R * np.sin(gamma)  # Eqn 10.106 (exact form)
    else:
        warnings.warn("The flare starts above the screen height: the approach is too steep "
                      "for this flare load factor.", PerformanceWarning, stacklevel=2)
        S_LA = float(np.sqrt(R ** 2 - (R - screen_height) ** 2))
    S_LR = V_TD * t_free_roll                                           # Eqn 10.110
    TW = ground_thrust / W
    A = 2.0 * G0 * ((mu_b - TW) - Nn * (mu_b - mu_g))                   # Eqn 10.113
    B = rho * G0 * (CD_ground - mu_b * CL_ground) / (W / S)             # Eqn 10.114
    if A <= 0.0:
        raise InfeasibleFlightConditionError("Braking cannot decelerate the aircraft (A <= 0).")
    arg = 1.0 + B / A * V_TD ** 2
    if arg <= 0.0:
        raise InfeasibleFlightConditionError("Invalid landing ground-roll parameters (1 + B V^2/A <= 0).")
    S_LNGR = float(np.log(arg) / B) if abs(B) > 1e-12 else V_TD ** 2 / A   # Eqn 10.112 / 10.115
    C, D = A / 2.0, B / 2.0                                             # Eqns 10.121-10.122
    if D > 0:
        t_LNGR = float(np.arctan(V_TD * np.sqrt(D / C)) / np.sqrt(C * D))   # Eqn 10.123
    elif D < 0:
        t_LNGR = float(np.arctanh(V_TD * np.sqrt(-D / C)) / np.sqrt(-C * D))  # Eqn 10.124
    else:
        t_LNGR = V_TD / C
    return LandingResult(V_S=V_S, V_A=V_A, V_FL=V_FL, V_TD=V_TD, gamma_A=gamma, R_flare=R, h_flare=h_F,
                         S_LA=float(S_LA), S_LR=S_LR, S_LNGR=S_LNGR, t_LA=float(S_LA / V_A),
                         t_LR=float(t_free_roll), t_LNGR=t_LNGR, method="Roskam approximate analytical")


def landing_ground_roll_numerical(weight, wing_area, rho, V_touchdown, mu_brake: float,
                                  CL_ground: float, CD_ground: float, ground_thrust: ThrustFunction = None,
                                  headwind: float = 0.0, runway_slope: float = 0.0,
                                  t_free_roll: float = 1.0, nose_load_ratio: float = 0.08,
                                  mu_rolling: float = ROLLING_FRICTION["concrete / asphalt"]) -> dict:
    """Integrate the landing ground roll from touchdown to rest (Roskam Eqns 10.72-10.75).

    During the first ``t_free_roll`` seconds the aircraft rolls freely
    (rolling friction, nose wheel off the ground); then brakes are applied
    on the main gear, with a fraction ``nose_load_ratio`` of the weight on the
    un-braked nose wheel. The ground speed is ``V - headwind``; the roll ends
    when the ground speed is zero.
    """
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    V_TD = ensure_positive("V_touchdown", V_touchdown)
    hw = float(headwind)
    if ground_thrust is None:
        def ground_thrust(V):
            return 0.0
    WS = W / S

    def acc(t, V):
        q = 0.5 * rho * V ** 2
        T = ground_thrust(V)
        if t < t_free_roll:
            mu_eff, nose = mu_rolling, 0.0
        else:
            mu_eff, nose = mu_brake, nose_load_ratio * (mu_brake - mu_rolling)
        # Eqn 10.72: the braking friction acts on (W - L - N_n) on the main gear
        return G0 * ((T / W - mu_eff) - (CD_ground - mu_eff * CL_ground) * q / WS
                     + nose - runway_slope)

    def rhs(t, y):
        return [acc(t, y[0]), y[0] - hw]

    def stopped(t, y):
        return y[0] - hw
    stopped.terminal = True
    stopped.direction = -1

    if acc(t_free_roll + 1.0, V_TD) >= 0.0:
        raise InfeasibleFlightConditionError("The aircraft does not decelerate on the ground.")
    sol = solve_ivp(rhs, (0.0, 600.0), [V_TD, 0.0], events=stopped, max_step=0.25,
                    rtol=1e-8, atol=1e-8)
    if sol.t_events[0].size == 0:
        raise InfeasibleFlightConditionError("The aircraft did not stop within 600 s.")
    return {"distance": float(sol.y_events[0][0][1]), "time": float(sol.t_events[0][0]),
            "t": sol.t, "V": sol.y[0], "s": sol.y[1]}


def landing_distance(aircraft: Aircraft, altitude: float = 0.0, weight: Optional[float] = None,
                     mu_brake: float = 0.4, approach_angle_deg: float = 3.0,
                     CL_ground: Optional[float] = None, CD_ground: Optional[float] = None,
                     headwind: float = 0.0, runway_slope: float = 0.0, delta_T: float = 0.0,
                     V_A_factor: float = 1.3, V_TD_factor: float = 1.15, n_flare: float = 1.08,
                     t_free_roll: Optional[float] = None, config: str = "landing") -> LandingResult:
    """Landing distance of an :class:`Aircraft`: Roskam air distance + numerical ground roll.

    Defaults: ground-roll CL = 0.3 CL_max (spoilers not modelled),
    ground-roll CD from the landing polar with ground effect; ``t_free_roll`` =
    1 s for propeller aircraft and 3 s for jets; zero ground thrust (idle).
    """
    W = aircraft.weight_or_default(weight)
    conf = aircraft.config(config)
    rho = isa(altitude, delta_T).density
    S = aircraft.wing_area
    if t_free_roll is None:
        t_free_roll = 3.0 if aircraft.is_jet else 1.0
    if CL_ground is None:
        CL_ground = 0.3 * conf.CL_max
    if CD_ground is None:
        sigma_p = 1.0
        if aircraft.wing_height_above_ground is not None:
            hb = np.clip(aircraft.wing_height_above_ground / aircraft.wing_span, 0.033, 0.25)
            sigma_p = float(aero.ground_effect_factor(hb))
        CD_ground = conf.polar.CD0 + sigma_p * conf.polar.K * CL_ground ** 2
    air = landing_roskam_approximate(W, S, rho, conf.CL_max, conf.polar, mu_brake, CL_ground, CD_ground,
                                     approach_angle_deg=approach_angle_deg, V_A_factor=V_A_factor,
                                     V_TD_factor=V_TD_factor, n_flare=n_flare, t_free_roll=t_free_roll)
    roll = landing_ground_roll_numerical(W, S, rho, air.V_TD, mu_brake, CL_ground, CD_ground,
                                         headwind=headwind, runway_slope=runway_slope,
                                         t_free_roll=t_free_roll)
    wind_factor = (air.V_A - float(headwind)) / air.V_A
    # the numerical roll already contains the free-roll phase: report it inside S_LNGR
    return LandingResult(V_S=air.V_S, V_A=air.V_A, V_FL=air.V_FL, V_TD=air.V_TD, gamma_A=air.gamma_A,
                         R_flare=air.R_flare, h_flare=air.h_flare, S_LA=air.S_LA * wind_factor,
                         S_LR=0.0, S_LNGR=roll["distance"], t_LA=air.t_LA, t_LR=0.0,
                         t_LNGR=roll["time"], method="Roskam air distance + numerical ground roll")


__all__ = [
    "SCREEN_HEIGHT", "ROLLING_FRICTION", "BRAKING_FRICTION", "SECOND_SEGMENT_MIN_GRADIENT",
    "linear_thrust_model", "quadratic_thrust_model", "optimum_ground_run_lift_coefficient",
    "ground_effect_induced_drag_reduction", "ground_acceleration", "transition_delta_CL",
    "TakeoffResult", "takeoff_roskam_approximate", "takeoff_ground_roll_numerical", "takeoff_distance",
    "takeoff_parameter_far23", "takeoff_distance_far23_statistical", "takeoff_parameter_far25",
    "takeoff_field_length_far25_statistical", "landing_distance_far23_statistical",
    "landing_field_length_far25_statistical", "mean_takeoff_thrust_jet", "mean_takeoff_thrust_propeller",
    "balanced_field_length_torenbeek", "LandingResult", "approach_angle", "landing_roskam_approximate",
    "landing_ground_roll_numerical", "landing_distance",
]
