"""Level flight, climb and descent: textbook examples and internal consistency."""

import numpy as np
import pytest

from aircraft_performance import climb, descent, level_flight
from aircraft_performance import units as u
from aircraft_performance.aerodynamics import ParabolicDragPolar
from aircraft_performance.atmosphere import isa
from aircraft_performance.errors import InfeasibleFlightConditionError, InputError

from conftest import RHO0_ROSKAM

H30K = 30000 * u.FT


# --------------------------------------------------------------------------- level flight
def test_anderson_min_thrust_and_speeds(giv):
    """Anderson Examples 5.2, 5.4, 5.8 (Gulfstream IV at 30,000 ft)."""
    W, S, pol = giv.weight, giv.wing_area, giv.polar()
    rho = isa(H30K).density
    assert level_flight.minimum_drag(W, pol) / u.LBF == pytest.approx(5058, rel=1e-3)
    assert level_flight.speed_minimum_drag(W, S, rho, pol) / u.FT == pytest.approx(631.2, rel=3e-3)
    assert level_flight.speed_minimum_power(W, S, rho, pol) / u.FT == pytest.approx(479.6, rel=3e-3)
    P_min = level_flight.minimum_power_required(W, S, rho, pol)
    assert P_min / (u.FT * u.LBF) == pytest.approx(2.8e6, rel=5e-3)


def test_anderson_max_speed_sea_level(giv):
    """Anderson Example 5.6(a): Eqn (5.50) with T/W = 0.3795, W/S = 76.84 psf."""
    V = level_flight.max_level_speed_jet_parabolic(giv.weight, giv.wing_area, isa(0).density,
                                                   giv.polar(), giv.propulsion.max_thrust_sl)
    TW, WS, CD0, K, rho = 0.3795, 76.84, 0.015, 0.08, 0.002377
    V_ref = np.sqrt((TW * WS + WS * np.sqrt(TW ** 2 - 4 * CD0 * K)) / (rho * CD0))
    assert V / u.FT == pytest.approx(V_ref, rel=2e-3)


@pytest.mark.parametrize("h", [0.0, 5000.0, 9000.0])
def test_numerical_max_speed_matches_closed_form_for_jet(giv, h):
    speeds = level_flight.level_flight_speeds(giv, h)
    T = giv.propulsion.thrust_available(0.0, h)
    V_cf = level_flight.max_level_speed_jet_parabolic(giv.weight, giv.wing_area, isa(h).density, giv.polar(), T)
    assert speeds.V_max == pytest.approx(V_cf, rel=1e-6)
    assert speeds.V_min >= speeds.V_stall


def test_propeller_max_speed_satisfies_power_balance(single):
    s = level_flight.level_flight_speeds(single, 2000.0)
    P_av = single.propulsion.power_available(s.V_max, 2000.0)
    P_req = level_flight.power_required(single.weight, single.wing_area, isa(2000.0).density,
                                        s.V_max, single.polar())
    assert P_av == pytest.approx(P_req, rel=1e-6)
    assert 50.0 < s.V_max < 80.0   # ~ 110-150 kt for a 160 hp light aircraft


def test_level_flight_impossible_above_ceiling(single):
    with pytest.raises(InfeasibleFlightConditionError):
        level_flight.level_flight_speeds(single, 9000.0)


def test_flight_envelope_stops_at_ceiling(single):
    env = level_flight.flight_envelope(single, np.arange(0, 10001, 1000))
    assert env["first_infeasible_altitude"] is not None
    assert np.all(env["V_max"] > env["V_min"])


def test_jet_level_flight_infeasible_thrust():
    pol = ParabolicDragPolar(0.02, 0.05)
    with pytest.raises(InfeasibleFlightConditionError):
        level_flight.max_level_speed_jet_parabolic(1e5, 50.0, 1.0, pol, 1e5 / 20.0)


# --------------------------------------------------------------------------- climb
@pytest.mark.parametrize("h_ft, rc_ref", [(0, 179.9), (10000, 156.6), (20000, 133.8), (30000, 111.0),
                                          (40000, 85.9), (50000, 58.2), (60000, 30.1)])
def test_anderson_max_rate_of_climb_table(giv, h_ft, rc_ref):
    """Anderson Example 5.16 table of (R/C)max versus altitude (ft/s)."""
    pt = climb.max_rate_of_climb(giv, h_ft * u.FT)
    assert pt.rate / u.FT == pytest.approx(rc_ref, abs=0.6)
    cf = climb.jet_max_rate_of_climb_parabolic(giv.weight, giv.wing_area, isa(h_ft * u.FT).density,
                                              giv.polar(), giv.propulsion.thrust_available(0, h_ft * u.FT))
    assert pt.rate == pytest.approx(cf.rate, rel=1e-5)
    assert pt.V == pytest.approx(cf.V, rel=1e-3)


def test_anderson_max_climb_angle(giv):
    """Anderson Example 5.13: theta_max = 18.07 deg at 376.8 ft/s (sea level)."""
    pt = climb.jet_max_climb_angle_parabolic(giv.weight, giv.wing_area, isa(0).density, giv.polar(),
                                             giv.propulsion.max_thrust_sl)
    assert pt.gamma_deg == pytest.approx(18.07, abs=0.02)
    assert pt.V / u.FT == pytest.approx(376.8, rel=2e-3)


def test_roskam_example_9_2():
    """Roskam Example 9.2: turboprop, RC_max = 5,116 ft/min at 147 kt, gamma = 0.34 rad (small angles)."""
    W, S = 36000 * u.LBF, 450 * u.FT2
    pol = ParabolicDragPolar(0.02, 0.05)
    V400 = 400 * u.KT
    CL = W / (0.5 * RHO0_ROSKAM * V400 ** 2 * S)
    THP = 0.5 * RHO0_ROSKAM * V400 ** 3 * S * (0.02 + 0.05 * CL ** 2 + 0.0015)
    assert THP / u.HP == pytest.approx(6768, rel=2e-3)
    pt = climb.propeller_max_rate_of_climb_parabolic(W, S, RHO0_ROSKAM, pol, THP)
    assert pt.CL == pytest.approx(1.095, rel=1e-3)
    assert pt.rate_fpm == pytest.approx(5116, rel=2e-3)
    assert pt.V / u.KT == pytest.approx(147, rel=5e-3)
    assert pt.rate / pt.V == pytest.approx(0.34, abs=0.005)
    sin_g = climb.propeller_steep_climb_sin_gamma(THP, W, pt.V, pt.CL, pol.drag_coefficient(pt.CL))
    assert np.degrees(np.arcsin(sin_g)) == pytest.approx(20.4, abs=0.1)


def test_steep_climb_consistent_with_small_angle_for_shallow_climbs(single):
    V = 40.0
    exact = climb.steep_climb(single, V, 0.0)
    approx = climb.rate_of_climb(single, V, 0.0)
    assert exact.rate == pytest.approx(approx, rel=0.02)


def test_numerical_best_rc_propeller_vs_closed_form(single):
    """With constant power (no static-thrust cap) the optimum is at point P."""
    from dataclasses import replace
    single = single.replace(propulsion=replace(single.propulsion, static_thrust=None))
    h = 1000.0
    pt = climb.max_rate_of_climb(single, h)
    cf = climb.propeller_max_rate_of_climb_parabolic(single.weight, single.wing_area, isa(h).density,
                                                     single.polar(), single.propulsion.power_available(1.0, h))
    assert pt.rate == pytest.approx(cf.rate, rel=1e-4)


def test_ceilings(single):
    c = climb.ceilings(single, include_cruise=True)
    # propeller aircraft: service ceiling at 100 ft/min, "cruise" ceiling at 300 ft/min (lower)
    assert c.cruise < c.service < c.absolute
    assert climb.max_rate_of_climb(single, c.absolute).rate == pytest.approx(0.0, abs=2e-3)
    assert climb.max_rate_of_climb(single, c.service).rate == pytest.approx(100 * u.FPM, abs=2e-3)


def test_time_to_climb_numerical_vs_linear(single):
    """For a nearly linear RC_max(h), the numerical integration approaches Roskam Eqn 9.74."""
    c = climb.ceilings(single)
    rc0 = climb.max_rate_of_climb(single, 0.0).rate
    h = 0.7 * c.absolute
    prof = climb.time_to_climb(single, 0.0, h, n_steps=40)
    t_lin = climb.time_to_climb_linear(rc0, c.absolute, h)
    assert prof.total_time == pytest.approx(t_lin, rel=0.05)
    assert prof.total_fuel_weight > 0 and np.all(np.diff(prof.weight) < 0)


def test_time_to_climb_above_ceiling_raises(single):
    with pytest.raises(InfeasibleFlightConditionError):
        climb.time_to_climb(single, 0.0, 8000.0)


def test_roskam_linear_time_to_climb():
    """Roskam p. 410: RC0 = 1,850 ft/min, h_abs = 32,000 ft -> 48 min to 30,000 ft."""
    t = climb.time_to_climb_linear(1850 * u.FPM, 32000 * u.FT, 30000 * u.FT)
    assert t / 60 == pytest.approx(48.0, abs=0.2)


@pytest.mark.parametrize("mach", [0.3, 0.5])
def test_acceleration_factor_vs_roskam(mach):
    """Roskam Eqns 9.84 / 9.87 (troposphere): 0.567 M^2 at constant EAS, -0.133 M^2 at constant M."""
    h = 5000.0
    V = mach * isa(h).speed_of_sound
    f_eas = climb.acceleration_factor_constant_eas(V * np.sqrt(isa(h).sigma), h)
    f_m = climb.acceleration_factor_constant_mach(mach, h)
    assert f_eas == pytest.approx(0.567 * mach ** 2, rel=0.03)
    assert f_m == pytest.approx(-0.133 * mach ** 2, rel=0.03)
    assert climb.acceleration_factor_constant_mach(mach, 15000.0) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- descent
def test_anderson_glide(giv):
    """Anderson Examples 5.14-5.15: gamma_min = 3.964 deg, R = 432,900 ft from 30,000 ft, RD = 43.6 ft/s."""
    rho = isa(H30K).density
    g = descent.best_glide(giv.weight, giv.wing_area, rho, giv.polar())
    assert g.gamma_deg == pytest.approx(3.964, abs=0.002)
    assert descent.max_glide_distance(H30K, giv.polar()) / u.FT == pytest.approx(432900, rel=1e-3)
    assert g.sink_rate / u.FT == pytest.approx(43.6, rel=3e-3)
    assert descent.minimum_sink(giv.weight, giv.wing_area, rho, giv.polar()).sink_rate / u.FT \
        == pytest.approx(38.6, rel=0.015)


def test_roskam_glider_best_glide_speed():
    """Roskam p. 341: A = 15.6, e = 0.82, CD0 = 0.015, W = 300 kg, S = 14.1 m^2 -> 20.9 m/s at sea level."""
    pol = ParabolicDragPolar.from_aspect_ratio(0.015, 15.6, 0.82)
    g = descent.best_glide(300 * 9.80665, 14.1, 1.225, pol)
    assert g.V == pytest.approx(20.9, abs=0.1)


def test_glide_time_integrated_vs_constant_density(single):
    """The sink rate decreases in denser air, so the integrated time exceeds Roskam Eqn (8.31)
    evaluated with the density at the start altitude."""
    t_int = descent.glide_time(single.weight, single.wing_area, single.polar(), 1500.0)
    t_cd = descent.glide_time_constant_density(single.weight, single.wing_area, isa(1500.0).density,
                                               single.polar(), 1500.0)
    assert t_cd < t_int < 1.1 * t_cd


def test_glide_with_wind(single):
    rho = isa(1000.0).density
    args = (single.weight, single.wing_area, rho, single.polar(), 1.6)
    calm = descent.best_glide_with_wind(*args)
    head = descent.best_glide_with_wind(*args, headwind=8.0)
    tail = descent.best_glide_with_wind(*args, headwind=-8.0)
    assert calm["ground_glide_ratio"] == pytest.approx(single.polar().max_lift_to_drag, rel=2e-3)
    assert head["ground_glide_ratio"] < calm["ground_glide_ratio"] < tail["ground_glide_ratio"]
    assert head["state"].V > calm["state"].V > tail["state"].V   # speed-to-fly theory
    with pytest.raises(InfeasibleFlightConditionError):
        descent.best_glide_with_wind(*args, headwind=200.0)


def test_drift_down(turboprop):
    prof = descent.drift_down(turboprop, 7500.0, extra_CD0=0.004)
    assert prof.final_altitude < 7500.0
    assert np.all(np.diff(prof.altitude) < 0)
    assert prof.total_time > 0 and prof.total_distance > 0
    assert np.all(prof.rate_of_descent >= 100 * u.FPM - 1e-9)


def test_drift_down_single_engine_rejected(single):
    with pytest.raises(InputError):
        descent.drift_down(single, 3000.0)


def test_idle_descent(giv):
    prof = descent.descent_at_constant_eas(giv, 10000.0, 1000.0, 130.0)
    assert prof.total_time > 0
    assert np.all(prof.rate_of_descent > 0)
