"""The error-handling contract: bad inputs raise InputError, doubtful results warn."""

import warnings

import pytest

from aircraft_performance import climb, examples, level_flight
from aircraft_performance.aerodynamics import ParabolicDragPolar
from aircraft_performance.aircraft import Aircraft, Configuration
from aircraft_performance.errors import InputError, PerformanceError, PerformanceWarning
from aircraft_performance.propulsion import JetPropulsion
from aircraft_performance import units as u


def _jet(**overrides):
    data = dict(
        name="test", mass=10000.0, wing_area=30.0, wing_span=15.0,
        configurations={"clean": Configuration(ParabolicDragPolar(0.02, 0.05), 1.4)},
        propulsion=JetPropulsion(30000.0, u.tsfc_to_si(0.7)),
    )
    data.update(overrides)
    return Aircraft(**data)


def test_valid_aircraft_builds():
    ac = _jet()
    assert ac.aspect_ratio == pytest.approx(7.5)
    assert "E_max" in ac.summary()


@pytest.mark.parametrize("overrides", [
    dict(mass=-1.0),
    dict(wing_span=100.0),                                   # AR = 333
    dict(configurations={"takeoff": Configuration(ParabolicDragPolar(0.03, 0.05), 1.8)}),  # no clean
    dict(operating_empty_mass=20000.0),                      # OEM > MTOM
    dict(propulsion="a jet engine"),
])
def test_invalid_aircraft_rejected(overrides):
    with pytest.raises(InputError):
        _jet(**overrides)


def test_missing_configuration_message():
    with pytest.raises(InputError, match="Available"):
        _jet().config("landing")


def test_error_hierarchy():
    assert issubclass(InputError, PerformanceError)
    assert issubclass(InputError, ValueError)


def test_aircraft_is_immutable():
    ac = examples.light_single_piston()
    with pytest.raises(Exception):
        ac.mass = 2000.0
    heavier = ac.replace(mass=1150.0)
    assert heavier.mass == 1150.0 and ac.mass == 1100.0


def test_compressibility_warning_is_emitted():
    giv = examples.business_jet_giv()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        with pytest.raises(PerformanceWarning, match="drag-divergence"):
            level_flight.level_flight_speeds(giv, 0.0)


def test_steep_climb_warning():
    pol = ParabolicDragPolar(0.02, 0.05)
    ac = _jet(configurations={"clean": Configuration(pol, 1.4)},
              propulsion=JetPropulsion(60000.0, u.tsfc_to_si(0.7)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        with pytest.raises(PerformanceWarning, match="small-angle"):
            climb.max_rate_of_climb(ac, 0.0)
