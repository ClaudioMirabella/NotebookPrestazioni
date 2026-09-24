"""
Simple, transparent propulsion models for performance analysis.

Following Roskam & Lan (Chapters 6, 8, 9) two idealised engine families are
used throughout the performance literature:

* **Propeller aircraft** (piston or turboprop): the shaft power is roughly
  independent of speed, so the *power available* ``P_av = eta_p * P_shaft``
  is constant with V and the thrust is ``T = P_av / V`` (Roskam Sec. 9.3).
* **Jet aircraft** (turbojet / low-to-moderate bypass turbofan): the thrust is
  roughly independent of speed (Roskam Sec. 9.2.1.2), so ``T_av`` is constant
  with V and the power available ``P_av = T_av * V`` grows linearly.

Both vary with altitude through a *lapse* model based on the density ratio
sigma. Fuel consumption is described by a constant specific fuel consumption
(SI units, fuel **weight** flow per unit power or thrust, see ``units``).

Both classes expose the same interface, so that every performance routine can
work with either of them:

* ``thrust_available(V, altitude, throttle=1, n_operative=None, delta_T=0)``
* ``power_available(V, altitude, ...)``
* ``fuel_weight_flow_for_thrust(thrust, V)`` -- fuel weight flow [N/s] needed
  to produce a given thrust at speed V.
* ``fuel_weight_flow_available(V, altitude, ...)`` -- fuel flow at the given throttle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .atmosphere import density_ratio
from .errors import InputError
from .validation import (ensure_fraction, ensure_in_range, ensure_non_negative,
                         ensure_positive)

#: Lapse models available for propeller engines.
POWER_LAPSE_MODELS = ("gagg_ferrar", "density_power", "turbocharged")


def _check_operating_engines(n_operative, n_engines: int) -> int:
    """Validate the number of operating engines (default: all of them)."""
    if n_operative is None:
        return n_engines
    if int(n_operative) != n_operative or not 0 <= n_operative <= n_engines:
        raise InputError(f"n_operative must be an integer between 0 and {n_engines}, got {n_operative!r}.")
    return int(n_operative)


@dataclass(frozen=True)
class PropellerPropulsion:
    """Piston-propeller or turboprop powerplant.

    Parameters
    ----------
    max_shaft_power_sl : float
        Total maximum shaft (brake) power of **all** engines at sea level [W].
    propeller_efficiency : float
        Installed propulsive efficiency eta_p (0 < eta_p <= 1), assumed constant.
    bsfc : float
        Brake specific fuel consumption in SI units [1/m = N/J]
        (use :func:`aircraft_performance.units.bsfc_to_si` to convert from lb/(hp h)).
    n_engines : int
        Number of engines (used for one-engine-inoperative, OEI, cases).
    power_lapse : str
        Altitude lapse model for the shaft power:

        * ``"gagg_ferrar"``: normally aspirated piston engine,
          ``P/P_SL = 1.132 sigma - 0.132`` (Gagg & Ferrar).
        * ``"density_power"``: ``P/P_SL = sigma ** lapse_exponent`` (turboprops, n ~ 0.7-1).
        * ``"turbocharged"``: flat rated up to ``critical_altitude``, then Gagg-Ferrar
          referred to the critical altitude density.
    lapse_exponent : float
        Exponent for ``"density_power"``.
    critical_altitude : float
        Critical altitude [m] for ``"turbocharged"``.
    static_thrust : float, optional
        Total static thrust at sea level [N]. Needed for take-off calculations
        because ``T = P/V`` diverges at V -> 0. The thrust is taken as
        ``min(static_thrust * sigma-lapse, eta_p P / V)``.
    propeller_diameter : float, optional
        Propeller diameter [m] (only used by the Torenbeek BFL estimate).
    """

    max_shaft_power_sl: float
    propeller_efficiency: float
    bsfc: float
    n_engines: int = 1
    power_lapse: str = "gagg_ferrar"
    lapse_exponent: float = 1.0
    critical_altitude: float = 0.0
    static_thrust: Optional[float] = None
    propeller_diameter: Optional[float] = None

    kind = "propeller"

    def __post_init__(self):
        ensure_positive("max_shaft_power_sl", self.max_shaft_power_sl)
        ensure_fraction("propeller_efficiency", self.propeller_efficiency)
        ensure_positive("bsfc", self.bsfc)
        if self.bsfc > 1e-5:
            raise InputError(
                f"bsfc = {self.bsfc} 1/m is far too large: did you forget to convert from "
                "lb/(hp h)? Use units.bsfc_to_si().")
        if int(self.n_engines) != self.n_engines or self.n_engines < 1:
            raise InputError(f"n_engines must be a positive integer, got {self.n_engines!r}.")
        if self.power_lapse not in POWER_LAPSE_MODELS:
            raise InputError(f"power_lapse must be one of {POWER_LAPSE_MODELS}, got {self.power_lapse!r}.")
        ensure_in_range("lapse_exponent", self.lapse_exponent, 0.3, 2.0)
        ensure_non_negative("critical_altitude", self.critical_altitude)
        if self.static_thrust is not None:
            ensure_positive("static_thrust", self.static_thrust)
        if self.propeller_diameter is not None:
            ensure_positive("propeller_diameter", self.propeller_diameter)

    # --- altitude lapse --------------------------------------------------------
    def power_lapse_ratio(self, altitude, delta_T: float = 0.0):
        """Ratio P_shaft(h) / P_shaft(sea level) at full throttle."""
        sigma = np.asarray(density_ratio(altitude, delta_T))
        if self.power_lapse == "gagg_ferrar":
            ratio = 1.132 * sigma - 0.132
        elif self.power_lapse == "density_power":
            ratio = sigma ** self.lapse_exponent
        else:  # turbocharged
            sigma_c = float(density_ratio(self.critical_altitude, delta_T))
            ratio = np.where(sigma >= sigma_c, 1.0, 1.132 * (sigma / sigma_c) - 0.132)
        ratio = np.clip(ratio, 0.0, None)
        return float(ratio) if ratio.ndim == 0 else ratio

    # --- available power / thrust ----------------------------------------------
    def shaft_power_available(self, altitude, throttle: float = 1.0, n_operative=None,
                              delta_T: float = 0.0):
        """Shaft power [W] of the operating engines at the given throttle setting."""
        phi = ensure_fraction("throttle", throttle, allow_zero=True)
        n_op = _check_operating_engines(n_operative, self.n_engines)
        return (phi * self.max_shaft_power_sl * n_op / self.n_engines
                * self.power_lapse_ratio(altitude, delta_T))

    def power_available(self, V, altitude, throttle: float = 1.0, n_operative=None,
                        delta_T: float = 0.0):
        """Thrust power available ``eta_p P_shaft`` [W] (independent of V)."""
        ensure_non_negative("V", V)
        P = self.propeller_efficiency * self.shaft_power_available(altitude, throttle, n_operative, delta_T)
        return P * np.ones_like(np.asarray(V, dtype=float)) if np.ndim(V) else P

    def thrust_available(self, V, altitude, throttle: float = 1.0, n_operative=None,
                         delta_T: float = 0.0):
        """Thrust [N] = eta_p P / V, limited by the (lapsed) static thrust if given.

        Raises
        ------
        InputError
            If V = 0 is requested and no ``static_thrust`` was provided.
        """
        V = np.asarray(ensure_non_negative("V", V), dtype=float)
        P_av = self.propeller_efficiency * self.shaft_power_available(altitude, throttle, n_operative, delta_T)
        if self.static_thrust is None:
            if np.any(V <= 0.0):
                raise InputError("Propeller thrust at V = 0 requires 'static_thrust' to be specified.")
            T = P_av / V
        else:
            phi = ensure_fraction("throttle", throttle, allow_zero=True)
            n_op = _check_operating_engines(n_operative, self.n_engines)
            T_static = (self.static_thrust * phi * n_op / self.n_engines
                        * self.power_lapse_ratio(altitude, delta_T))
            with np.errstate(divide="ignore"):
                T = np.where(V > 0.0, np.minimum(T_static, P_av / np.where(V > 0, V, 1.0)), T_static)
        return float(T) if T.ndim == 0 else T

    # --- fuel -------------------------------------------------------------------
    def fuel_weight_flow_for_thrust(self, thrust, V):
        """Fuel weight flow [N/s] to produce ``thrust`` at speed V: ``c * T V / eta_p`` (Roskam Eqn 11.1)."""
        T = ensure_non_negative("thrust", thrust)
        V = ensure_positive("V", V)
        return self.bsfc * np.asarray(T) * np.asarray(V) / self.propeller_efficiency

    def fuel_weight_flow_available(self, V, altitude, throttle: float = 1.0, n_operative=None,
                                   delta_T: float = 0.0):
        """Fuel weight flow [N/s] of the engines at the given throttle: ``c * P_shaft``."""
        return self.bsfc * self.shaft_power_available(altitude, throttle, n_operative, delta_T)


@dataclass(frozen=True)
class JetPropulsion:
    """Turbojet / turbofan powerplant with speed-independent thrust.

    Parameters
    ----------
    max_thrust_sl : float
        Total maximum thrust of **all** engines at sea level [N].
    tsfc : float
        Thrust specific fuel consumption in SI units [1/s]
        (use :func:`aircraft_performance.units.tsfc_to_si` to convert from 1/h).
    n_engines : int
        Number of engines.
    lapse_exponent : float
        Thrust lapse ``T/T_SL = sigma ** m``. Anderson suggests m ~ 0.6-1.0
        depending on the engine; m = 1 is the classic simple turbojet model.
    bypass_ratio : float
        Engine bypass ratio (only used by the Torenbeek BFL estimate, Eqn 10.58).
    """

    max_thrust_sl: float
    tsfc: float
    n_engines: int = 2
    lapse_exponent: float = 1.0
    bypass_ratio: float = 0.0

    kind = "jet"

    def __post_init__(self):
        ensure_positive("max_thrust_sl", self.max_thrust_sl)
        ensure_positive("tsfc", self.tsfc)
        if self.tsfc > 1e-2:
            raise InputError(
                f"tsfc = {self.tsfc} 1/s is far too large: did you forget to convert from "
                "1/h? Use units.tsfc_to_si().")
        if int(self.n_engines) != self.n_engines or self.n_engines < 1:
            raise InputError(f"n_engines must be a positive integer, got {self.n_engines!r}.")
        ensure_in_range("lapse_exponent", self.lapse_exponent, 0.3, 2.0)
        ensure_non_negative("bypass_ratio", self.bypass_ratio)

    def thrust_lapse_ratio(self, altitude, delta_T: float = 0.0):
        """Ratio T(h) / T(sea level) at full throttle = sigma ** m."""
        return density_ratio(altitude, delta_T) ** self.lapse_exponent

    def thrust_available(self, V, altitude, throttle: float = 1.0, n_operative=None,
                         delta_T: float = 0.0):
        """Available thrust [N], constant with speed."""
        ensure_non_negative("V", V)
        phi = ensure_fraction("throttle", throttle, allow_zero=True)
        n_op = _check_operating_engines(n_operative, self.n_engines)
        T = phi * self.max_thrust_sl * n_op / self.n_engines * self.thrust_lapse_ratio(altitude, delta_T)
        return T * np.ones_like(np.asarray(V, dtype=float)) if np.ndim(V) else float(T)

    def power_available(self, V, altitude, throttle: float = 1.0, n_operative=None,
                        delta_T: float = 0.0):
        """Available power T_av * V [W]."""
        return self.thrust_available(V, altitude, throttle, n_operative, delta_T) * np.asarray(V)

    def fuel_weight_flow_for_thrust(self, thrust, V=None):
        """Fuel weight flow [N/s] to produce ``thrust``: ``c_t * T`` (Roskam Eqn 11.47)."""
        T = ensure_non_negative("thrust", thrust)
        return self.tsfc * np.asarray(T)

    def fuel_weight_flow_available(self, V, altitude, throttle: float = 1.0, n_operative=None,
                                   delta_T: float = 0.0):
        """Fuel weight flow [N/s] at the given throttle setting."""
        return self.tsfc * self.thrust_available(V, altitude, throttle, n_operative, delta_T)


def static_thrust_momentum_theory(shaft_power, propeller_diameter, rho: float = 1.225,
                                  figure_of_merit: float = 0.6):
    """Rough static thrust of a propeller from actuator-disk theory [N].

    ``T0 = (FM * P)^(2/3) * (2 rho A)^(1/3)``, with A the disk area and FM the
    figure of merit (0.5-0.75 for real propellers). Use only as a first guess
    when manufacturer data are not available.
    """
    P = ensure_positive("shaft_power", shaft_power)
    D = ensure_positive("propeller_diameter", propeller_diameter)
    rho = ensure_positive("rho", rho)
    FM = ensure_fraction("figure_of_merit", figure_of_merit)
    A = np.pi * D ** 2 / 4.0
    return float((FM * P) ** (2.0 / 3.0) * (2.0 * rho * A) ** (1.0 / 3.0))
