"""
International Standard Atmosphere (ISA), 0 - 32 km geopotential altitude.

This is a transparent re-implementation of the model described in the
``Capitolo1/AtmosferaStandard.ipynb`` notebook (and Roskam & Lan, Appendix A),
so that every performance routine can compute density, pressure and speed of
sound without hidden dependencies. The results agree with the ``ambiance``
package used in the earlier notebooks (this is verified in the test suite).

Model
-----
The atmosphere is divided into layers with a linear temperature profile
``T = T_b + L (h - h_b)`` (``L`` is the lapse rate). Hydrostatic equilibrium
``dp = -rho g dh`` combined with the perfect gas law ``p = rho R T`` gives

* for ``L != 0``:  ``p = p_b (T / T_b) ** (-g / (L R))``
* for ``L == 0``:  ``p = p_b exp(-g (h - h_b) / (R T_b))``

A temperature offset ``delta_T`` (hot or cold day, e.g. ISA+15) can be added:
the pressure profile is kept standard (pressure altitude) while the
temperature is shifted, and the density follows from the perfect gas law.
This is the usual convention in flight manuals.

Altitudes are *geopotential* altitudes in metres; the difference with the
geometric altitude is below 0.3 % up to 20 km and is neglected here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .errors import InputError
from .units import G0
from .validation import ensure_finite, ensure_in_range, ensure_positive

# ---------------------------------------------------------------------------
# Constants of the ISA model
# ---------------------------------------------------------------------------
R_AIR = 287.05287          #: specific gas constant of dry air [J/(kg K)]
GAMMA_AIR = 1.4            #: ratio of specific heats [-]
T0 = 288.15                #: sea-level temperature [K]
P0 = 101325.0              #: sea-level pressure [Pa]
RHO0 = P0 / (R_AIR * T0)   #: sea-level density [kg/m^3]  (= 1.2250)
A0 = float(np.sqrt(GAMMA_AIR * R_AIR * T0))  #: sea-level speed of sound [m/s]

#: Minimum and maximum altitude supported by the model [m].
H_MIN = -1000.0
H_MAX = 32000.0

# Layer table: (base altitude [m], base temperature [K], lapse rate [K/m])
_LAYERS = (
    (0.0, 288.15, -0.0065),      # troposphere
    (11000.0, 216.65, 0.0),      # tropopause / lower stratosphere (isothermal)
    (20000.0, 216.65, 0.0010),   # stratosphere
)


def _layer_base_pressures() -> tuple:
    """Pre-compute the pressure at the base of each layer."""
    pressures = [P0]
    for i in range(1, len(_LAYERS)):
        h_b, T_b, L = _LAYERS[i - 1]
        h_top = _LAYERS[i][0]
        pressures.append(_pressure_in_layer(h_top, h_b, T_b, L, pressures[-1]))
    return tuple(pressures)


def _pressure_in_layer(h, h_b, T_b, L, p_b):
    """Hydrostatic pressure inside one layer (see module docstring)."""
    if L == 0.0:
        return p_b * np.exp(-G0 * (h - h_b) / (R_AIR * T_b))
    T = T_b + L * (h - h_b)
    return p_b * (T / T_b) ** (-G0 / (L * R_AIR))


_P_BASES = None  # filled lazily below (needs _pressure_in_layer defined)


def _standard_T_p(h: np.ndarray):
    """Standard temperature and pressure at geopotential altitude(s) ``h``."""
    global _P_BASES
    if _P_BASES is None:
        _P_BASES = _layer_base_pressures()
    T = np.empty_like(h)
    p = np.empty_like(h)
    for i, (h_b, T_b, L) in enumerate(_LAYERS):
        h_top = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else np.inf
        # The first layer also covers the (small) negative altitudes.
        mask = (h < h_top) & ((h >= h_b) | (i == 0))
        T[mask] = T_b + L * (h[mask] - h_b)
        p[mask] = _pressure_in_layer(h[mask], h_b, T_b, L, _P_BASES[i])
    return T, p


@dataclass(frozen=True)
class AtmosphereState:
    """Properties of the atmosphere at one (or several) altitude(s).

    All attributes are floats for a scalar altitude, NumPy arrays otherwise.
    """

    altitude: float      #: geopotential altitude h [m]
    temperature: float   #: static temperature T [K]
    pressure: float      #: static pressure p [Pa]
    density: float       #: density rho [kg/m^3]
    speed_of_sound: float  #: a = sqrt(gamma R T) [m/s]
    dynamic_viscosity: float  #: mu, Sutherland's law [Pa s]

    @property
    def theta(self):
        """Temperature ratio T / T0 [-]."""
        return self.temperature / T0

    @property
    def delta(self):
        """Pressure ratio p / p0 [-]."""
        return self.pressure / P0

    @property
    def sigma(self):
        """Density ratio rho / rho0 [-]."""
        return self.density / RHO0

    @property
    def kinematic_viscosity(self):
        """nu = mu / rho [m^2/s]."""
        return self.dynamic_viscosity / self.density


def _sutherland(T):
    """Dynamic viscosity of air from Sutherland's law [Pa s]."""
    beta_s = 1.458e-6   # [kg/(m s K^0.5)]
    S = 110.4           # Sutherland temperature [K]
    return beta_s * T ** 1.5 / (T + S)


def isa(altitude, delta_T: float = 0.0) -> AtmosphereState:
    """Return the atmospheric state at the given altitude(s).

    Parameters
    ----------
    altitude : float or array_like
        Geopotential (pressure) altitude [m], between ``H_MIN`` and ``H_MAX``.
    delta_T : float, optional
        Temperature offset from the standard day [K] (e.g. ``+15`` for ISA+15).

    Raises
    ------
    InputError
        If the altitude is outside the model range or the offset produces a
        non-physical (negative) temperature.
    """
    ensure_in_range("altitude", altitude, H_MIN, H_MAX)
    delta_T = ensure_finite("delta_T", delta_T)
    h = np.atleast_1d(np.asarray(altitude, dtype=float))
    T_std, p = _standard_T_p(h)
    T = T_std + delta_T
    if np.any(T <= 0.0):
        raise InputError(f"delta_T = {delta_T} K gives a non-physical temperature.")
    rho = p / (R_AIR * T)
    a = np.sqrt(GAMMA_AIR * R_AIR * T)
    mu = _sutherland(T)

    def _out(x):
        return float(x[0]) if np.ndim(altitude) == 0 else x

    return AtmosphereState(_out(h), _out(T), _out(p), _out(rho), _out(a), _out(mu))


def density(altitude, delta_T: float = 0.0):
    """Air density [kg/m^3] at the given altitude(s) (shortcut for ``isa(h).density``)."""
    return isa(altitude, delta_T).density


def density_ratio(altitude, delta_T: float = 0.0):
    """Density ratio sigma = rho / rho0 [-] at the given altitude(s)."""
    return isa(altitude, delta_T).sigma


def altitude_from_density(rho: float, delta_T: float = 0.0) -> float:
    """Find the (pressure) altitude [m] at which the density equals ``rho``.

    Density decreases monotonically with altitude, so the inverse problem has
    a unique solution, found here with Brent's method. Useful e.g. to compute
    the final altitude of a cruise-climb (constant CL and TAS).

    Raises
    ------
    InputError
        If ``rho`` is outside the range covered by the model.
    """
    rho = ensure_positive("rho", rho)
    rho_max = float(density(H_MIN, delta_T))
    rho_min = float(density(H_MAX, delta_T))
    if not rho_min <= rho <= rho_max:
        raise InputError(
            f"Density {rho:.5f} kg/m^3 is outside the model range "
            f"[{rho_min:.5f}, {rho_max:.5f}] kg/m^3."
        )
    return float(brentq(lambda h: float(density(h, delta_T)) - rho, H_MIN, H_MAX, xtol=1e-6))


# ---------------------------------------------------------------------------
# Geometric vs geopotential altitude
# ---------------------------------------------------------------------------
EARTH_RADIUS = 6356766.0   #: effective Earth radius used by the ISA [m]


def geopotential_altitude(geometric_altitude):
    """Convert geometric altitude z [m] into geopotential altitude H = r z / (r + z) [m].

    The ``ambiance`` package used in ``Capitolo1`` takes *geometric* altitude as
    input, while this module (like most performance texts) uses geopotential
    altitude: the two differ by 0.3 % at 20 km, which changes the density by
    about 0.25 % at 11 km and about 2 % at 30 km.
    """
    z = ensure_finite("geometric_altitude", geometric_altitude)
    return EARTH_RADIUS * np.asarray(z) / (EARTH_RADIUS + np.asarray(z))


def geometric_altitude(geopotential_altitude_m):
    """Convert geopotential altitude H [m] into geometric altitude z = r H / (r - H) [m]."""
    H = ensure_finite("geopotential_altitude_m", geopotential_altitude_m)
    return EARTH_RADIUS * np.asarray(H) / (EARTH_RADIUS - np.asarray(H))


# ---------------------------------------------------------------------------
# Airspeed conversions
# ---------------------------------------------------------------------------
def true_to_equivalent_airspeed(V_true, altitude, delta_T: float = 0.0):
    """EAS = TAS * sqrt(sigma): the airspeed that gives the same dynamic pressure at sea level."""
    V = ensure_finite("V_true", V_true)
    return V * np.sqrt(density_ratio(altitude, delta_T))


def equivalent_to_true_airspeed(V_eq, altitude, delta_T: float = 0.0):
    """TAS = EAS / sqrt(sigma)."""
    V = ensure_finite("V_eq", V_eq)
    return V / np.sqrt(density_ratio(altitude, delta_T))


def mach_number(V_true, altitude, delta_T: float = 0.0):
    """Flight Mach number M = V / a."""
    V = ensure_finite("V_true", V_true)
    return V / isa(altitude, delta_T).speed_of_sound
