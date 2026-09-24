"""Tests for units, input validation, atmosphere, drag polar and propulsion models."""

import numpy as np
import pytest

from aircraft_performance import aerodynamics as aero
from aircraft_performance import atmosphere as atm
from aircraft_performance import units as u
from aircraft_performance.aerodynamics import ParabolicDragPolar
from aircraft_performance.errors import InputError, PerformanceWarning
from aircraft_performance.propulsion import JetPropulsion, PropellerPropulsion, static_thrust_momentum_theory
from aircraft_performance.validation import ensure_positive, ensure_result_finite


# --------------------------------------------------------------------------- units
def test_unit_constants():
    assert u.HP == pytest.approx(745.69987, rel=1e-6)
    assert u.LBF == pytest.approx(4.4482216, rel=1e-7)
    assert u.KT == pytest.approx(0.514444, rel=1e-5)
    assert u.SLUG_FT3 * 0.002377 == pytest.approx(1.2250, rel=1e-3)


def test_sfc_conversions_round_trip():
    # 1/c = 603,500 m for c = 1 lb/(hp h): the factor used in Capitolo9
    assert 1.0 / u.bsfc_to_si(1.0) == pytest.approx(603_500, rel=1e-3)
    assert u.bsfc_from_si(u.bsfc_to_si(0.45)) == pytest.approx(0.45)
    assert u.tsfc_from_si(u.tsfc_to_si(0.69)) == pytest.approx(0.69)
    assert u.tsfc_to_si(0.69) == pytest.approx(1.917e-4, rel=1e-3)   # Anderson Example 5.20


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, "abc", [1.0, -2.0]])
def test_ensure_positive_rejects(bad):
    with pytest.raises(InputError):
        ensure_positive("x", bad)


def test_ensure_result_finite():
    with pytest.raises(Exception):
        ensure_result_finite("y", np.array([1.0, np.nan]))


# --------------------------------------------------------------------------- atmosphere
def test_isa_sea_level():
    s = atm.isa(0.0)
    assert s.temperature == pytest.approx(288.15)
    assert s.pressure == pytest.approx(101325.0)
    assert s.density == pytest.approx(1.2250, rel=1e-4)
    assert s.speed_of_sound == pytest.approx(340.29, rel=1e-4)
    assert s.sigma == pytest.approx(1.0)


def test_isa_tropopause_and_stratosphere():
    s = atm.isa(11000.0)
    assert s.temperature == pytest.approx(216.65)
    assert s.pressure == pytest.approx(22632.0, rel=1e-4)
    assert atm.isa(20000.0).pressure == pytest.approx(5474.9, rel=1e-4)


def test_isa_vectorised_and_monotonic():
    h = np.linspace(0, 30000, 31)
    rho = atm.density(h)
    assert rho.shape == h.shape
    assert np.all(np.diff(rho) < 0)


def test_isa_matches_ambiance():
    ambiance = pytest.importorskip("ambiance")
    z = np.linspace(0, 30000, 40)
    ref = ambiance.Atmosphere(z)
    ours = atm.isa(atm.geopotential_altitude(z))
    np.testing.assert_allclose(ours.density, ref.density, rtol=1e-5)
    np.testing.assert_allclose(ours.pressure, ref.pressure, rtol=1e-5)
    np.testing.assert_allclose(ours.temperature, ref.temperature, rtol=1e-6)


def test_altitude_from_density_inverse():
    for h in (0.0, 4000.0, 11000.0, 17000.0):
        assert atm.altitude_from_density(atm.density(h)) == pytest.approx(h, abs=1e-3)


def test_hot_day_lowers_density():
    assert atm.density(1000.0, delta_T=20.0) < atm.density(1000.0)


def test_isa_out_of_range():
    with pytest.raises(InputError):
        atm.isa(40000.0)
    with pytest.raises(InputError):
        atm.altitude_from_density(5.0)


def test_airspeed_conversions():
    V = 100.0
    h = 5000.0
    assert atm.equivalent_to_true_airspeed(atm.true_to_equivalent_airspeed(V, h), h) == pytest.approx(V)
    assert atm.mach_number(atm.isa(0).speed_of_sound, 0.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- drag polar
def test_characteristic_points_are_true_maxima():
    pol = ParabolicDragPolar(CD0=0.025, K=0.045)
    CL = np.linspace(0.05, 2.0, 200001)
    CD = pol.drag_coefficient(CL)
    assert CL[np.argmax(CL / CD)] == pytest.approx(pol.point_E.CL, abs=1e-4)
    assert CL[np.argmax(CL ** 1.5 / CD)] == pytest.approx(pol.point_P.CL, abs=1e-4)
    assert CL[np.argmax(CL ** 0.5 / CD)] == pytest.approx(pol.point_A.CL, abs=1e-4)
    assert pol.point_E.CD == pytest.approx(2 * pol.CD0)
    assert pol.point_P.CD == pytest.approx(4 * pol.CD0)
    assert pol.point_A.CD == pytest.approx(4 / 3 * pol.CD0)


def test_anderson_aerodynamic_ratios():
    """Anderson Example 5.4: (L/D)max = 14.43, (CL^1.5/CD)max ~ 10.8, (CL^0.5/CD)max = 25.0."""
    pol = ParabolicDragPolar(CD0=0.015, K=0.08)
    assert pol.max_lift_to_drag == pytest.approx(14.43, rel=1e-3)
    assert pol.point_P.endurance_parameter == pytest.approx(10.83, rel=1e-2)
    assert pol.point_A.range_parameter_jet == pytest.approx(0.75 * (1 / (3 * 0.08 * 0.015 ** 3)) ** 0.25)


def test_speed_ratios_between_points():
    """V_P : V_E : V_A = 0.76 : 1 : 1.32 (Roskam Eqn 8.75, Anderson Example 5.4)."""
    pol = ParabolicDragPolar(CD0=0.02, K=0.05)
    vE = aero.speed_at_polar_point(1e5, 50.0, 1.0, pol.point_E)
    assert aero.speed_at_polar_point(1e5, 50.0, 1.0, pol.point_P) / vE == pytest.approx(3 ** -0.25)
    assert aero.speed_at_polar_point(1e5, 50.0, 1.0, pol.point_A) / vE == pytest.approx(3 ** 0.25)


def test_polar_from_aspect_ratio_and_increments():
    pol = ParabolicDragPolar.from_aspect_ratio(0.03, 7.5, 0.8)
    assert pol.K == pytest.approx(1 / (np.pi * 7.5 * 0.8))
    flaps = pol.with_increments(delta_CD0=0.02, name="landing")
    assert flaps.CD0 == pytest.approx(0.05) and flaps.K == pol.K and pol.CD0 == 0.03


@pytest.mark.parametrize("kwargs", [dict(CD0=-0.01, K=0.05), dict(CD0=0.02, K=0.0), dict(CD0=0.9, K=0.05)])
def test_polar_rejects_bad_values(kwargs):
    with pytest.raises(InputError):
        ParabolicDragPolar(**kwargs)


def test_drag_breakdown_equal_at_min_drag():
    pol = ParabolicDragPolar(CD0=0.02, K=0.05)
    V = aero.speed_at_polar_point(50000.0, 30.0, 1.1, pol.point_E)
    D0, Di = aero.drag_breakdown(50000.0, 30.0, 1.1, V, pol)
    assert D0 == pytest.approx(Di)


def test_parasite_area_method():
    f = aero.equivalent_parasite_area(100.0, 0.0045)
    assert aero.cd0_from_parasite_area(f, 20.0) == pytest.approx(0.0225)


def test_ground_effect_factor_roskam_example_10_1():
    """Roskam Example 10.1: 2h/b = 0.36 -> sigma' = 0.32; Example 10.2: h/b = 0.103 -> 0.48."""
    assert aero.ground_effect_factor(0.18) == pytest.approx(0.32, abs=0.01)
    assert aero.ground_effect_factor(0.103) == pytest.approx(0.48, abs=0.01)
    with pytest.raises(InputError):
        aero.ground_effect_factor(0.5)


def test_mach_warning():
    pol = ParabolicDragPolar(CD0=0.02, K=0.05, mach_drag_divergence=0.8)
    with pytest.warns(PerformanceWarning):
        pol.check_mach(0.9)


# --------------------------------------------------------------------------- propulsion
def test_propeller_lapse_models():
    p = PropellerPropulsion(100e3, 0.8, u.bsfc_to_si(0.45))
    assert p.power_lapse_ratio(0.0) == pytest.approx(1.0)
    sigma = atm.density_ratio(3000.0)
    assert p.power_lapse_ratio(3000.0) == pytest.approx(1.132 * sigma - 0.132)
    turbo = PropellerPropulsion(100e3, 0.8, u.bsfc_to_si(0.45), power_lapse="turbocharged",
                                critical_altitude=5000.0)
    assert turbo.power_lapse_ratio(4000.0) == pytest.approx(1.0)
    assert turbo.power_lapse_ratio(7000.0) < 1.0


def test_propeller_thrust_and_oei():
    p = PropellerPropulsion(200e3, 0.8, u.bsfc_to_si(0.45), n_engines=2, static_thrust=6000.0)
    assert p.thrust_available(50.0, 0.0) == pytest.approx(0.8 * 200e3 / 50.0)
    assert p.thrust_available(0.0, 0.0) == pytest.approx(6000.0)
    assert p.thrust_available(50.0, 0.0, n_operative=1) == pytest.approx(0.8 * 100e3 / 50.0)
    with pytest.raises(InputError):
        p.thrust_available(50.0, 0.0, n_operative=3)
    no_static = PropellerPropulsion(200e3, 0.8, u.bsfc_to_si(0.45))
    with pytest.raises(InputError):
        no_static.thrust_available(0.0, 0.0)


def test_unconverted_sfc_is_rejected():
    with pytest.raises(InputError):
        PropellerPropulsion(100e3, 0.8, 0.45)        # lb/(hp h) passed as SI
    with pytest.raises(InputError):
        JetPropulsion(50e3, 0.69)                     # 1/h passed as 1/s


def test_jet_lapse_anderson_example_5_6():
    """Anderson Example 5.6: T(30,000 ft) = 27,700 (rho/rho0)^0.6 = 15,371 lb."""
    j = JetPropulsion(27700 * u.LBF, u.tsfc_to_si(0.69), lapse_exponent=0.6)
    assert j.thrust_available(200.0, 30000 * u.FT) / u.LBF == pytest.approx(15371, rel=3e-3)
    assert j.fuel_weight_flow_for_thrust(1000.0) == pytest.approx(1000.0 * u.tsfc_to_si(0.69))


def test_static_thrust_estimate_reasonable():
    T0 = static_thrust_momentum_theory(119e3, 1.9, figure_of_merit=0.6)
    assert 2000.0 < T0 < 5000.0
