"""
Minimum aerodynamic database for performance calculations.

Everything in flight performance starts from two ingredients:

1. the **drag polar** ``CD = CD(CL)`` of the aircraft in each configuration
   (clean, take-off flaps, landing flaps + gear...);
2. the **maximum lift coefficient** ``CL_max`` of each configuration, which
   fixes the stall speed.

This module provides a parabolic drag polar (Roskam & Lan Eqn 5.37, used in
all the closed-form results of Chapters 8-11), the characteristic points of
the polar, the classic relations between speed, lift coefficient and wing
loading, and a few quick estimation methods (equivalent parasite area,
ground effect, Oswald factor).

Characteristic points of a parabolic polar ``CD = CD0 + K CL^2``
-----------------------------------------------------------------
The naming follows the Italian teaching tradition already used in this
repository (points *E*, *P*, *A*):

======  =======================  ==================  =========  ================================
Point   Maximises                CL                  CD         Used for
======  =======================  ==================  =========  ================================
E       CL / CD (efficiency)     sqrt(CD0/K)         2 CD0      min drag, prop range, jet endurance, best glide
P       CL^(3/2) / CD            sqrt(3 CD0/K)       4 CD0      min power, prop endurance, min sink, prop best RC
A       CL^(1/2) / CD            sqrt(CD0/(3K))      4/3 CD0    jet range at constant altitude
======  =======================  ==================  =========  ================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .errors import InputError, PerformanceWarning
from .validation import (ensure_fraction, ensure_in_range, ensure_non_negative,
                         ensure_positive, ensure_result_finite)


# ---------------------------------------------------------------------------
# Parabolic drag polar
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolarPoint:
    """A characteristic point of the drag polar."""

    name: str                 #: short label ("E", "P", "A")
    description: str          #: what the point maximises
    CL: float                 #: lift coefficient [-]
    CD: float                 #: drag coefficient [-]

    @property
    def lift_to_drag(self) -> float:
        """Aerodynamic efficiency E = CL / CD at this point."""
        return self.CL / self.CD

    @property
    def endurance_parameter(self) -> float:
        """CL^(3/2) / CD at this point."""
        return self.CL ** 1.5 / self.CD

    @property
    def range_parameter_jet(self) -> float:
        """CL^(1/2) / CD at this point."""
        return self.CL ** 0.5 / self.CD


@dataclass(frozen=True)
class ParabolicDragPolar:
    """Parabolic drag polar ``CD = CD0 + K * CL**2`` (Roskam Eqn 5.37).

    Parameters
    ----------
    CD0 : float
        Zero-lift drag coefficient [-].
    K : float
        Induced drag factor ``K = 1 / (pi AR e)`` [-].
    name : str
        Label of the configuration (e.g. "clean", "take-off").
    mach_drag_divergence : float, optional
        Drag-divergence Mach number. The parabolic polar ignores compressibility;
        when a calculation exceeds this Mach number a
        :class:`~aircraft_performance.errors.PerformanceWarning` is issued.
    """

    CD0: float
    K: float
    name: str = "clean"
    mach_drag_divergence: Optional[float] = None

    def __post_init__(self):
        ensure_positive("CD0", self.CD0)
        ensure_positive("K", self.K)
        if self.CD0 > 0.5:
            raise InputError(f"CD0 = {self.CD0} is unrealistically large; check the input.")
        if self.mach_drag_divergence is not None:
            ensure_in_range("mach_drag_divergence", self.mach_drag_divergence, 0.1, 3.0)

    # --- alternative constructors -------------------------------------------
    @classmethod
    def from_aspect_ratio(cls, CD0: float, aspect_ratio: float, oswald_factor: float,
                          name: str = "clean", mach_drag_divergence: Optional[float] = None):
        """Build the polar from CD0, the aspect ratio AR and the Oswald factor e."""
        AR = ensure_positive("aspect_ratio", aspect_ratio)
        e = ensure_fraction("oswald_factor", oswald_factor)
        return cls(CD0=CD0, K=1.0 / (np.pi * AR * e), name=name,
                   mach_drag_divergence=mach_drag_divergence)

    def with_increments(self, delta_CD0: float = 0.0, K_factor: float = 1.0,
                        name: Optional[str] = None) -> "ParabolicDragPolar":
        """Return a new polar with a zero-lift drag increment and/or scaled K.

        Typical uses (Roskam Sec. 5.1.2 and Fig. 9.11): flaps and landing gear
        add ``delta_CD0``; an inoperative engine adds windmilling and trim drag;
        ground effect reduces the induced drag, i.e. ``K_factor < 1``.
        """
        dCD0 = ensure_non_negative("delta_CD0", delta_CD0)
        kf = ensure_positive("K_factor", K_factor)
        return replace(self, CD0=self.CD0 + dCD0, K=self.K * kf,
                       name=name if name is not None else self.name)

    # --- evaluation -----------------------------------------------------------
    def drag_coefficient(self, CL):
        """CD for the given lift coefficient(s)."""
        CL = np.asarray(CL, dtype=float)
        out = self.CD0 + self.K * CL ** 2
        return float(out) if out.ndim == 0 else out

    def lift_to_drag(self, CL):
        """Aerodynamic efficiency CL / CD."""
        return np.asarray(CL) / self.drag_coefficient(CL)

    # --- characteristic points -------------------------------------------------
    @property
    def max_lift_to_drag(self) -> float:
        """E_max = 1 / (2 sqrt(CD0 K))  (Roskam Eqn 8.18)."""
        return 1.0 / (2.0 * np.sqrt(self.CD0 * self.K))

    @property
    def point_E(self) -> PolarPoint:
        """Maximum efficiency CL/CD: minimum drag (Roskam Eqns 8.18-8.19)."""
        CL = np.sqrt(self.CD0 / self.K)
        return PolarPoint("E", "max CL/CD", float(CL), 2.0 * self.CD0)

    @property
    def point_P(self) -> PolarPoint:
        """Maximum CL^(3/2)/CD: minimum power required (Roskam Eqns 8.25-8.26)."""
        CL = np.sqrt(3.0 * self.CD0 / self.K)
        return PolarPoint("P", "max CL^1.5/CD", float(CL), 4.0 * self.CD0)

    @property
    def point_A(self) -> PolarPoint:
        """Maximum CL^(1/2)/CD: best jet range at constant altitude (Roskam Eqn 11.66)."""
        CL = np.sqrt(self.CD0 / (3.0 * self.K))
        return PolarPoint("A", "max CL^0.5/CD", float(CL), 4.0 / 3.0 * self.CD0)

    def characteristic_points(self) -> dict:
        """Dictionary {"A": ..., "E": ..., "P": ...} of the characteristic points."""
        return {"A": self.point_A, "E": self.point_E, "P": self.point_P}

    def check_mach(self, mach, context: str = "") -> None:
        """Warn if ``mach`` exceeds the drag-divergence Mach number (if known)."""
        if self.mach_drag_divergence is None:
            return
        m = np.max(np.asarray(mach, dtype=float))
        if m > self.mach_drag_divergence:
            warnings.warn(
                f"{context}Mach number {m:.3f} exceeds the drag-divergence Mach number "
                f"{self.mach_drag_divergence:.3f} of polar '{self.name}': the parabolic "
                "polar underestimates drag here (compressibility is not modelled).",
                PerformanceWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Speed - lift coefficient - wing loading relations
# ---------------------------------------------------------------------------
def dynamic_pressure(rho, V):
    """q = 0.5 rho V^2 [Pa]."""
    rho = ensure_positive("rho", rho)
    V = ensure_non_negative("V", V)
    return 0.5 * rho * np.asarray(V) ** 2


def lift_coefficient(weight, wing_area, rho, V, load_factor: float = 1.0):
    """Lift coefficient needed to produce ``L = n W`` at speed V: ``CL = 2 n W / (rho V^2 S)``."""
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    V = ensure_positive("V", V)
    n = ensure_positive("load_factor", load_factor)
    return 2.0 * n * W / (rho * np.asarray(V) ** 2 * S)


def speed_for_lift_coefficient(weight, wing_area, rho, CL, load_factor: float = 1.0):
    """True airspeed at which the lift coefficient ``CL`` produces ``L = n W``.

    ``V = sqrt(2 n W / (rho S CL))`` (Roskam Eqn 8.23 for n = 1).
    """
    W = ensure_positive("weight", weight)
    S = ensure_positive("wing_area", wing_area)
    rho = ensure_positive("rho", rho)
    CL = ensure_positive("CL", CL)
    n = ensure_positive("load_factor", load_factor)
    return np.sqrt(2.0 * n * W / (rho * S * np.asarray(CL)))


def stall_speed(weight, wing_area, rho, CL_max, load_factor: float = 1.0):
    """1-g (or n-g) stall speed ``V_S = sqrt(2 n W / (rho S CL_max))`` (Roskam Eqns 8.32, 12.54)."""
    CL_max = ensure_in_range("CL_max", CL_max, 0.1, 6.0)
    return speed_for_lift_coefficient(weight, wing_area, rho, CL_max, load_factor)


def speed_at_polar_point(weight, wing_area, rho, point: PolarPoint):
    """Level-flight speed at a characteristic point of the polar (E, P or A)."""
    return speed_for_lift_coefficient(weight, wing_area, rho, point.CL)


def drag(weight, wing_area, rho, V, polar: ParabolicDragPolar, load_factor: float = 1.0):
    """Total drag [N] in (level or n-g) flight at speed V, with ``L = n W``."""
    CL = lift_coefficient(weight, wing_area, rho, V, load_factor)
    CD = polar.drag_coefficient(CL)
    return ensure_result_finite("drag", 0.5 * rho * np.asarray(V) ** 2 * wing_area * CD)


def drag_breakdown(weight, wing_area, rho, V, polar: ParabolicDragPolar):
    """Split level-flight drag into zero-lift (parasite) and induced parts.

    Returns
    -------
    (D0, Di) : tuple of float or arrays [N]
        ``D0 = q S CD0`` grows with V^2, ``Di = K W^2 / (q S)`` decays with 1/V^2
        (Roskam Eqn 8.59). They are equal at the minimum-drag speed (point E).
    """
    q = dynamic_pressure(rho, V)
    S = ensure_positive("wing_area", wing_area)
    W = ensure_positive("weight", weight)
    D0 = q * S * polar.CD0
    Di = polar.K * W ** 2 / (q * S)
    return D0, Di


# ---------------------------------------------------------------------------
# Quick estimation methods
# ---------------------------------------------------------------------------
#: Representative equivalent skin-friction coefficients C_f read from Roskam
#: Figs. 5.41-5.43 (lines of constant C_f in the f - S_wet plots).
TYPICAL_SKIN_FRICTION_COEFFICIENTS = {
    "jet transport": 0.0030,
    "business jet": 0.0040,
    "jet fighter": 0.0040,
    "turboprop / multi-engine propeller": 0.0045,
    "single-engine propeller, retractable gear": 0.0060,
    "single-engine propeller, fixed gear": 0.0090,
    "sailplane": 0.0030,
}


def equivalent_parasite_area(wetted_area, skin_friction_coefficient):
    """Equivalent parasite (flat-plate) area ``f = C_f S_wet`` [m^2] (Roskam Figs. 5.41-5.43)."""
    S_wet = ensure_positive("wetted_area", wetted_area)
    cf = ensure_in_range("skin_friction_coefficient", skin_friction_coefficient, 1e-4, 0.05)
    return cf * S_wet


def cd0_from_parasite_area(parasite_area, wing_area):
    """Zero-lift drag coefficient ``CD0 = f / S`` (Roskam Eqn 5.38)."""
    f = ensure_positive("parasite_area", parasite_area)
    S = ensure_positive("wing_area", wing_area)
    return f / S


def oswald_factor_estimate(aspect_ratio, sweep_le_deg: float = 0.0):
    """Empirical Oswald efficiency factor (Raymer's correlation, not from Roskam).

    * straight wing (LE sweep < 30 deg): ``e = 1.78 (1 - 0.045 AR^0.68) - 0.64``
    * swept wing: ``e = 4.61 (1 - 0.045 AR^0.68) cos(sweep)^0.15 - 3.1``

    Use it only when no better data is available; Roskam Table 5.4 lists
    measured values between 0.67 and 0.93 for real aircraft.
    """
    AR = ensure_in_range("aspect_ratio", aspect_ratio, 2.0, 40.0)
    sweep = ensure_in_range("sweep_le_deg", sweep_le_deg, 0.0, 70.0)
    if sweep < 30.0:
        e = 1.78 * (1.0 - 0.045 * AR ** 0.68) - 0.64
    else:
        e = 4.61 * (1.0 - 0.045 * AR ** 0.68) * np.cos(np.radians(sweep)) ** 0.15 - 3.1
    if not 0.3 < e < 1.0:
        raise InputError(f"The correlation gives e = {e:.3f}, outside its validity range.")
    return float(e)


def ground_effect_factor(height_over_span):
    """Induced-drag ground influence coefficient sigma' (Roskam Eqn 10.13).

    ``sigma' = (1 - 1.32 h/b) / (1.05 + 7.4 h/b)`` for ``0.033 <= h/b <= 0.25``.
    The induced drag in ground effect is ``sigma'`` times the free-air value
    (Roskam Eqn 10.12), i.e. ``K_ground = sigma' K``.
    ``h`` is the height of the wing above the ground, ``b`` the span.
    Not valid for large flap deflections (> ~40 deg).
    """
    hb = ensure_in_range("height_over_span", height_over_span, 0.033, 0.25)
    return (1.0 - 1.32 * hb) / (1.05 + 7.4 * hb)


def minimum_turning_speed_factor(load_factor):
    """Stall speed ratio in an n-g manoeuvre: V_S(n) / V_S(1) = sqrt(n) (Roskam Eqn 12.54)."""
    n = ensure_positive("load_factor", load_factor)
    return np.sqrt(n)


__all__ = [
    "PolarPoint", "ParabolicDragPolar", "dynamic_pressure", "lift_coefficient",
    "speed_for_lift_coefficient", "stall_speed", "speed_at_polar_point", "drag",
    "drag_breakdown", "TYPICAL_SKIN_FRICTION_COEFFICIENTS", "equivalent_parasite_area",
    "cd0_from_parasite_area", "oswald_factor_estimate", "ground_effect_factor",
    "minimum_turning_speed_factor",
]
