"""
The :class:`Aircraft` data container.

An ``Aircraft`` gathers the minimum data needed by every performance module:

* geometry: wing area S and span b (-> aspect ratio);
* weights (masses): reference (maximum take-off) mass and, optionally,
  operating empty mass, maximum payload and maximum fuel for payload-range;
* aerodynamics: one :class:`Configuration` (drag polar + CL_max) per flight
  phase, e.g. ``"clean"``, ``"takeoff"``, ``"landing"``;
* propulsion: a :class:`~aircraft_performance.propulsion.PropellerPropulsion`
  or :class:`~aircraft_performance.propulsion.JetPropulsion` object.

The container is *immutable* (frozen dataclass): to study a variant (heavier
aircraft, different engine...) create a modified copy with
:meth:`Aircraft.replace` instead of changing an object shared by other cells
of a notebook. This avoids a very common source of confusing results in
interactive work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Union

from .aerodynamics import ParabolicDragPolar
from .errors import InputError
from .propulsion import JetPropulsion, PropellerPropulsion
from .units import G0
from .validation import ensure_in_range, ensure_non_negative, ensure_positive

Propulsion = Union[PropellerPropulsion, JetPropulsion]


@dataclass(frozen=True)
class Configuration:
    """Aerodynamic configuration: drag polar and maximum lift coefficient."""

    polar: ParabolicDragPolar
    CL_max: float

    def __post_init__(self):
        if not isinstance(self.polar, ParabolicDragPolar):
            raise InputError("Configuration.polar must be a ParabolicDragPolar.")
        ensure_in_range("CL_max", self.CL_max, 0.3, 6.0)


@dataclass(frozen=True)
class Aircraft:
    """Minimum aircraft description for performance analysis (SI units).

    Parameters
    ----------
    name : str
        Descriptive name.
    mass : float
        Reference mass [kg], normally the maximum take-off mass.
    wing_area : float
        Reference wing area S [m^2].
    wing_span : float
        Wing span b [m].
    configurations : dict[str, Configuration]
        Must contain at least ``"clean"``. ``"takeoff"`` and ``"landing"`` are
        used by the field-performance module.
    propulsion : PropellerPropulsion or JetPropulsion
        Powerplant model.
    operating_empty_mass, max_payload_mass, max_fuel_mass : float, optional
        Masses [kg] used for payload-range and mission calculations.
    wing_height_above_ground : float, optional
        Height of the wing (mean aerodynamic chord) above the runway [m], used
        for ground-effect corrections during take-off and landing.
    """

    name: str
    mass: float
    wing_area: float
    wing_span: float
    configurations: Dict[str, Configuration]
    propulsion: Propulsion
    operating_empty_mass: Optional[float] = None
    max_payload_mass: Optional[float] = None
    max_fuel_mass: Optional[float] = None
    wing_height_above_ground: Optional[float] = None
    notes: str = field(default="", compare=False)

    def __post_init__(self):
        ensure_positive("mass", self.mass)
        ensure_positive("wing_area", self.wing_area)
        ensure_positive("wing_span", self.wing_span)
        if "clean" not in self.configurations:
            raise InputError("Aircraft.configurations must contain a 'clean' configuration.")
        for key, conf in self.configurations.items():
            if not isinstance(conf, Configuration):
                raise InputError(f"Configuration '{key}' is not a Configuration instance.")
        if not isinstance(self.propulsion, (PropellerPropulsion, JetPropulsion)):
            raise InputError("Aircraft.propulsion must be a PropellerPropulsion or JetPropulsion.")
        for name in ("operating_empty_mass", "max_payload_mass", "max_fuel_mass",
                     "wing_height_above_ground"):
            value = getattr(self, name)
            if value is not None:
                ensure_non_negative(name, value)
        if self.operating_empty_mass is not None and self.operating_empty_mass >= self.mass:
            raise InputError("operating_empty_mass must be smaller than the reference (take-off) mass.")
        AR = self.aspect_ratio
        if not 2.0 <= AR <= 40.0:
            raise InputError(f"Aspect ratio b^2/S = {AR:.2f} is unrealistic; check wing_span and wing_area.")

    # --- derived quantities ----------------------------------------------------
    @property
    def weight(self) -> float:
        """Reference weight W = m g0 [N]."""
        return self.mass * G0

    @property
    def aspect_ratio(self) -> float:
        """Wing aspect ratio AR = b^2 / S."""
        return self.wing_span ** 2 / self.wing_area

    @property
    def wing_loading(self) -> float:
        """Reference wing loading W / S [N/m^2]."""
        return self.weight / self.wing_area

    @property
    def is_jet(self) -> bool:
        """True for jet propulsion, False for propeller propulsion."""
        return isinstance(self.propulsion, JetPropulsion)

    def config(self, name: str = "clean") -> Configuration:
        """Return the named configuration, with a clear error if it is missing."""
        try:
            return self.configurations[name]
        except KeyError:
            raise InputError(
                f"Aircraft '{self.name}' has no configuration '{name}'. "
                f"Available: {sorted(self.configurations)}.") from None

    def polar(self, name: str = "clean") -> ParabolicDragPolar:
        """Drag polar of the named configuration."""
        return self.config(name).polar

    def weight_or_default(self, weight: Optional[float]) -> float:
        """Return ``weight`` if given (validated), otherwise the reference weight."""
        return self.weight if weight is None else ensure_positive("weight", weight)

    def replace(self, **changes) -> "Aircraft":
        """Return a modified copy (the original object is left untouched)."""
        return replace(self, **changes)

    def summary(self) -> str:
        """Human-readable multi-line summary of the main data."""
        p = self.propulsion
        lines = [
            f"Aircraft: {self.name}",
            f"  mass            = {self.mass:10.1f} kg   (W = {self.weight / 1000:.2f} kN)",
            f"  wing area S     = {self.wing_area:10.2f} m^2",
            f"  span b          = {self.wing_span:10.2f} m    (AR = {self.aspect_ratio:.2f})",
            f"  wing loading    = {self.wing_loading:10.1f} N/m^2",
        ]
        for key, conf in self.configurations.items():
            pol = conf.polar
            lines.append(f"  [{key:8s}] CD = {pol.CD0:.4f} + {pol.K:.4f} CL^2,  CL_max = {conf.CL_max:.2f},"
                         f"  E_max = {pol.max_lift_to_drag:.2f}")
        if isinstance(p, JetPropulsion):
            lines.append(f"  jet: T_SL = {p.max_thrust_sl / 1000:.2f} kN ({p.n_engines} engines), "
                         f"TSFC = {p.tsfc * 3600:.3f} 1/h, lapse sigma^{p.lapse_exponent}")
        else:
            lines.append(f"  propeller: P_SL = {p.max_shaft_power_sl / 1000:.1f} kW ({p.n_engines} engines), "
                         f"eta_p = {p.propeller_efficiency:.2f}, BSFC = {p.bsfc:.3e} 1/m, lapse '{p.power_lapse}'")
        return "\n".join(lines)


def check_weight_not_above_reference(aircraft: Aircraft, weight: float) -> None:
    """Raise if a weight exceeds the reference (maximum take-off) weight by more than 0.1 %."""
    if weight > aircraft.weight * 1.001:
        raise InputError(
            f"Weight {weight:.0f} N exceeds the reference weight {aircraft.weight:.0f} N "
            f"of '{aircraft.name}'.")


__all__ = ["Aircraft", "Configuration", "Propulsion", "check_weight_not_above_reference"]
