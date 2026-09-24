"""
Library of example aircraft used in the notebooks and in the tests.

The data are *representative*, not certified manufacturer data: they are taken
from (or patterned after) the worked examples of the references so that
results can be compared with the textbooks.

* :func:`light_single_piston` -- the light single-engine piston aircraft used in
  the existing ``Capitolo9`` notebook (M = 1100 kg, S = 16 m^2).
* :func:`roskam_light_twin` -- the light twin of Roskam & Lan Examples 9.1,
  10.2 and 10.3 (Cessna 310-like, W = 4,600 lb).
* :func:`regional_turboprop` -- an ATR 72-like regional turboprop, patterned after
  the ``Capitolo9`` ATR exercise.
* :func:`business_jet_giv` -- the Gulfstream IV-like business jet used throughout
  Anderson, *Aircraft Performance and Design*, Chapters 5-6.

Each function returns a new :class:`~aircraft_performance.aircraft.Aircraft`.
"""

from __future__ import annotations

from . import units as u
from .aerodynamics import ParabolicDragPolar
from .aircraft import Aircraft, Configuration
from .propulsion import JetPropulsion, PropellerPropulsion


def light_single_piston() -> Aircraft:
    """Light single-engine piston aircraft (Cessna 172-like), 160 hp."""
    AR = 11.0 ** 2 / 16.0
    clean = ParabolicDragPolar.from_aspect_ratio(0.030, AR, 0.80, name="clean")
    takeoff = clean.with_increments(delta_CD0=0.010, name="takeoff (flaps 10)")
    landing = clean.with_increments(delta_CD0=0.040, name="landing (flaps 30)")
    return Aircraft(
        name="Light single-engine piston (C172-like)",
        mass=1100.0, wing_area=16.0, wing_span=11.0,
        configurations={
            "clean": Configuration(clean, CL_max=1.6),
            "takeoff": Configuration(takeoff, CL_max=1.9),
            "landing": Configuration(landing, CL_max=2.1),
        },
        propulsion=PropellerPropulsion(
            max_shaft_power_sl=160.0 * u.HP, propeller_efficiency=0.75,
            bsfc=u.bsfc_to_si(0.45), n_engines=1, power_lapse="gagg_ferrar",
            static_thrust=2600.0, propeller_diameter=1.90),
        operating_empty_mass=700.0, max_payload_mass=300.0, max_fuel_mass=150.0,
        wing_height_above_ground=2.2,
        notes="Data consistent with Capitolo9/AutonomieMotoelica (CD0 = 0.030, e = 0.8).",
    )


def roskam_light_twin() -> Aircraft:
    """Light piston twin of Roskam & Lan Examples 9.1, 10.2 and 10.3 (W = 4,600 lb)."""
    S = 175.0 * u.FT2
    b = 35.0 * u.FT
    clean = ParabolicDragPolar(CD0=0.0293, K=0.0557, name="clean")
    takeoff = ParabolicDragPolar.from_aspect_ratio(0.0620, 7.0, 0.80, name="takeoff (flaps 15)")
    landing = ParabolicDragPolar.from_aspect_ratio(0.1000, 7.0, 0.80, name="landing (flaps 45, gear down)")
    return Aircraft(
        name="Roskam light twin (C310-like)",
        mass=4600.0 * u.LB, wing_area=S, wing_span=b,
        configurations={
            "clean": Configuration(clean, CL_max=1.31),
            "takeoff": Configuration(takeoff, CL_max=1.69),
            "landing": Configuration(landing, CL_max=2.12),
        },
        propulsion=PropellerPropulsion(
            max_shaft_power_sl=2 * 260.0 * u.HP, propeller_efficiency=0.80,
            bsfc=u.bsfc_to_si(0.45), n_engines=2, power_lapse="gagg_ferrar",
            static_thrust=2000.0 * u.LBF, propeller_diameter=1.93),
        operating_empty_mass=3100.0 * u.LB, max_payload_mass=900.0 * u.LB,
        max_fuel_mass=800.0 * u.LB, wing_height_above_ground=3.6 * u.FT,
        notes="Roskam & Lan, Examples 9.1 (clean polar), 10.2 (take-off), 10.3 (landing).",
    )


def regional_turboprop() -> Aircraft:
    """ATR 72-like regional twin turboprop."""
    AR = 26.833 ** 2 / 60.0
    clean = ParabolicDragPolar.from_aspect_ratio(0.028, AR, 0.80, name="clean",
                                                 mach_drag_divergence=0.60)
    takeoff = clean.with_increments(delta_CD0=0.015, name="takeoff (flaps 15)")
    landing = clean.with_increments(delta_CD0=0.050, name="landing (flaps 30, gear down)")
    return Aircraft(
        name="Regional turboprop (ATR 72-like)",
        mass=22800.0, wing_area=60.0, wing_span=26.833,
        configurations={
            "clean": Configuration(clean, CL_max=1.6),
            "takeoff": Configuration(takeoff, CL_max=2.1),
            "landing": Configuration(landing, CL_max=2.6),
        },
        propulsion=PropellerPropulsion(
            max_shaft_power_sl=2 * 1846.0 * u.KW, propeller_efficiency=0.80,
            bsfc=u.bsfc_to_si(0.50), n_engines=2, power_lapse="density_power",
            lapse_exponent=0.8, static_thrust=2 * 30000.0, propeller_diameter=3.93),
        operating_empty_mass=13000.0, max_payload_mass=7500.0, max_fuel_mass=5000.0,
        wing_height_above_ground=4.0,
        notes="Polar consistent with Capitolo9 ATR exercise (CD0 = 0.028, e = 0.8).",
    )


def business_jet_giv() -> Aircraft:
    """Gulfstream IV-like business jet of Anderson, Chapters 5 and 6."""
    clean = ParabolicDragPolar(CD0=0.015, K=0.08, name="clean", mach_drag_divergence=0.85)
    takeoff = clean.with_increments(delta_CD0=0.0177, name="takeoff (flaps TO, gear down)")
    landing = clean.with_increments(delta_CD0=0.0300, name="landing (flaps full, gear down)")
    return Aircraft(
        name="Business jet (Gulfstream IV-like)",
        mass=73000.0 * u.LB, wing_area=950.0 * u.FT2, wing_span=75.0 * u.FT,
        configurations={
            "clean": Configuration(clean, CL_max=1.20),
            "takeoff": Configuration(takeoff, CL_max=1.86),
            "landing": Configuration(landing, CL_max=2.39),
        },
        propulsion=JetPropulsion(
            max_thrust_sl=2 * 13850.0 * u.LBF, tsfc=u.tsfc_to_si(0.69), n_engines=2,
            lapse_exponent=0.6, bypass_ratio=3.04),
        operating_empty_mass=16100.0, max_payload_mass=2950.0, max_fuel_mass=29500.0 * u.LB,
        wing_height_above_ground=5.6 * u.FT,
        notes="Anderson Example 5.1: CD = 0.015 + 0.08 CL^2, T = 2 x 13,850 lb, m = 0.6, c_t = 0.69 1/h.",
    )


def all_examples() -> dict:
    """Dictionary with all example aircraft, keyed by a short name."""
    return {
        "light_single_piston": light_single_piston(),
        "roskam_light_twin": roskam_light_twin(),
        "regional_turboprop": regional_turboprop(),
        "business_jet_giv": business_jet_giv(),
    }
