"""
Manoeuvring performance and flight envelope (Roskam & Lan, Chapter 12).

Level coordinated turn (Sec. 12.6)::

    n = L / W = 1 / cos(phi)                            (Eqn 12.50)
    R = V^2 / (g sqrt(n^2 - 1))                         (Eqn 12.51)
    omega = V / R = g sqrt(n^2 - 1) / V                 (Eqn 12.52)

The load factor is limited by

* aerodynamics (stall): ``n <= q S CL_max / W`` -- *instantaneous* turn (Eqn 12.45);
* structure: ``n <= n_limit`` (V-n diagram);
* thrust/power: a *sustained* turn needs ``T_av >= D(n)`` with
  ``D(n) = q S (CD0 + K (n W / (q S))^2)`` (Eqns 12.46-12.47).

The FAR 23 V-n manoeuvre diagram (Sec. 12.4.1) is also provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import aerodynamics as aero
from . import units as u
from .aircraft import Aircraft
from .atmosphere import isa
from .errors import InputError
from .units import G0
from .validation import ensure_in_range, ensure_positive


def load_factor_from_bank_angle(bank_angle_deg):
    """n = 1 / cos(phi) for a level coordinated turn (Roskam Eqn 12.50)."""
    phi = np.radians(ensure_in_range("bank_angle_deg", bank_angle_deg, 0.0, 89.0))
    return 1.0 / np.cos(phi)


def bank_angle_from_load_factor(load_factor):
    """phi = arccos(1/n) [deg]."""
    n = ensure_in_range("load_factor", load_factor, 1.0, 50.0)
    return np.degrees(np.arccos(1.0 / np.asarray(n)))


def turn_radius(V, load_factor):
    """Radius of a level coordinated turn R = V^2 / (g sqrt(n^2 - 1)) [m] (Roskam Eqn 12.51)."""
    V = ensure_positive("V", V)
    n = np.asarray(ensure_in_range("load_factor", load_factor, 1.0, 50.0))
    if np.any(n <= 1.0):
        raise InputError("A turn requires a load factor n > 1 (n = 1 is straight flight, R = infinity).")
    return np.asarray(V) ** 2 / (G0 * np.sqrt(n ** 2 - 1.0))


def turn_rate(V, load_factor):
    """Turn rate omega = g sqrt(n^2 - 1) / V [rad/s] (Roskam Eqn 12.52)."""
    V = ensure_positive("V", V)
    n = np.asarray(ensure_in_range("load_factor", load_factor, 1.0, 50.0))
    return G0 * np.sqrt(n ** 2 - 1.0) / np.asarray(V)


def instantaneous_load_factor(weight, wing_area, rho, V, CL_max):
    """Aerodynamic (stall) limit on the load factor n = q S CL_max / W (Roskam Eqn 12.45)."""
    q = aero.dynamic_pressure(rho, V)
    return q * ensure_positive("wing_area", wing_area) * ensure_positive("CL_max", CL_max) \
        / ensure_positive("weight", weight)


def sustained_load_factor(aircraft: Aircraft, V, altitude: float, weight: Optional[float] = None,
                          config: str = "clean", throttle: float = 1.0, delta_T: float = 0.0):
    """Thrust-limited load factor of a sustained level turn at speed V (Eqns 12.46-12.47).

    From ``T_av = q S CD0 + K n^2 W^2 / (q S)``:
    ``n = sqrt((T_av - q S CD0) q S / K) / W``.
    Returns 0 where the thrust does not even cover the zero-lift drag.
    """
    W = aircraft.weight_or_default(weight)
    V = np.asarray(ensure_positive("V", V), dtype=float)
    rho = isa(altitude, delta_T).density
    polar = aircraft.polar(config)
    q = 0.5 * rho * V ** 2
    S = aircraft.wing_area
    T = aircraft.propulsion.thrust_available(V, altitude, throttle, None, delta_T)
    excess = np.clip(T - q * S * polar.CD0, 0.0, None)
    return np.sqrt(excess * q * S / polar.K) / W


@dataclass(frozen=True)
class TurnPerformance:
    """Turn performance versus speed at one altitude (arrays)."""

    V: np.ndarray
    n_aerodynamic: np.ndarray   #: stall limit
    n_sustained: np.ndarray     #: thrust/power limit
    n_structural: float         #: limit load factor
    n_instantaneous: np.ndarray  #: min(aerodynamic, structural)
    n_sustained_limited: np.ndarray  #: min(sustained, aerodynamic, structural)

    def radius(self, which: str = "sustained") -> np.ndarray:
        """Minimum turn radius for the chosen limit; infinite where n <= 1."""
        n = self.n_sustained_limited if which == "sustained" else self.n_instantaneous
        out = np.full_like(self.V, np.inf)
        ok = n > 1.0
        out[ok] = turn_radius(self.V[ok], n[ok])
        return out

    def rate(self, which: str = "sustained") -> np.ndarray:
        """Maximum turn rate [rad/s] for the chosen limit; zero where n <= 1."""
        n = self.n_sustained_limited if which == "sustained" else self.n_instantaneous
        out = np.zeros_like(self.V)
        ok = n > 1.0
        out[ok] = turn_rate(self.V[ok], n[ok])
        return out


def turn_performance(aircraft: Aircraft, altitude: float, V, n_limit: float,
                     weight: Optional[float] = None, config: str = "clean",
                     throttle: float = 1.0, delta_T: float = 0.0) -> TurnPerformance:
    """Instantaneous and sustained turn limits versus speed (Roskam Fig. 12.21)."""
    W = aircraft.weight_or_default(weight)
    V = np.asarray(ensure_positive("V", V), dtype=float)
    n_lim = ensure_in_range("n_limit", n_limit, 1.0, 15.0)
    rho = isa(altitude, delta_T).density
    n_aero = instantaneous_load_factor(W, aircraft.wing_area, rho, V, aircraft.config(config).CL_max)
    n_sus = sustained_load_factor(aircraft, V, altitude, W, config, throttle, delta_T)
    n_inst = np.minimum(n_aero, n_lim)
    return TurnPerformance(V=V, n_aerodynamic=n_aero, n_sustained=n_sus, n_structural=n_lim,
                           n_instantaneous=n_inst, n_sustained_limited=np.minimum(n_sus, n_inst))


def maneuvering_speed(stall_speed, n_limit):
    """Design manoeuvring speed V_A = V_S sqrt(n_limit) (Roskam Eqns 12.19, 12.37)."""
    return ensure_positive("stall_speed", stall_speed) * np.sqrt(ensure_positive("n_limit", n_limit))


def absolute_minimum_turn_radius(stall_speed):
    """Absolute minimum radius V_S^2 / g, approached for n -> infinity (Roskam Eqn 12.56)."""
    return ensure_positive("stall_speed", stall_speed) ** 2 / G0


def far23_limit_load_factors(weight, category: str = "normal") -> tuple:
    """FAR 23 limit load factors (n_pos, n_neg) (Roskam Eqns 12.26-12.28).

    ``weight`` in N. Normal category: ``n_pos = 2.1 + 24000/(W_lb + 10000)``,
    at most 3.8; utility 4.4; acrobatic 6.0; ``n_neg = -0.4 n_pos`` (normal,
    utility) or ``-0.5 n_pos`` (acrobatic).
    """
    W_lb = ensure_positive("weight", weight) / u.LBF
    if category == "normal":
        n_pos = min(2.1 + 24000.0 / (W_lb + 10000.0), 3.8)
        return n_pos, -0.4 * n_pos
    if category == "utility":
        return 4.4, -0.4 * 4.4
    if category == "acrobatic":
        return 6.0, -0.5 * 6.0
    raise InputError("category must be 'normal', 'utility' or 'acrobatic'.")


def vn_maneuver_diagram(weight, wing_area, CL_max_pos: float, CL_max_neg: float, n_pos: float,
                        n_neg: float, V_C: float, V_D: float, n_points: int = 200) -> dict:
    """FAR 23-style V-n manoeuvre envelope at sea level, in equivalent airspeed (Roskam Fig. 12.12).

    The positive/negative stall parabolas ``n = rho0 V^2 S CN_max / (2W)``
    (with CN_max ~ 1.1 CL_max, Eqn 12.17) are cut by the limit load factors;
    the envelope is closed at the dive speed V_D. The negative limit is
    linearly reduced to 0 between V_C and V_D (FAR 23 practice).
    """
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    if not n_pos > 0 > n_neg:
        raise InputError("n_pos must be positive and n_neg negative.")
    if not V_D > V_C > 0:
        raise InputError("Speeds must satisfy V_D > V_C > 0.")
    rho0 = isa(0.0).density
    CN_pos = 1.1 * ensure_positive("CL_max_pos", CL_max_pos)
    CN_neg = 1.1 * ensure_positive("CL_max_neg", CL_max_neg)
    V = np.linspace(0.0, V_D, n_points)
    n_stall_pos = 0.5 * rho0 * V ** 2 * S * CN_pos / W
    n_stall_neg = -0.5 * rho0 * V ** 2 * S * CN_neg / W
    upper = np.minimum(n_stall_pos, n_pos)
    neg_limit = np.where(V <= V_C, n_neg, n_neg * (V_D - V) / (V_D - V_C))
    lower = np.maximum(n_stall_neg, neg_limit)
    V_S = float(np.sqrt(2.0 * W / (rho0 * S * CN_pos)))
    return {"V": V, "n_upper": upper, "n_lower": lower, "V_S": V_S,
            "V_A": float(V_S * np.sqrt(n_pos)), "V_C": float(V_C), "V_D": float(V_D)}


__all__ = [
    "load_factor_from_bank_angle", "bank_angle_from_load_factor", "turn_radius", "turn_rate",
    "instantaneous_load_factor", "sustained_load_factor", "TurnPerformance", "turn_performance",
    "maneuvering_speed", "absolute_minimum_turn_radius", "far23_limit_load_factors",
    "vn_maneuver_diagram",
]
