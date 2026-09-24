"""Range/endurance, take-off/landing and manoeuvring: textbook examples and consistency checks."""

import numpy as np
import pytest

from aircraft_performance import maneuvering as mnv
from aircraft_performance import range_endurance as rng
from aircraft_performance import takeoff_landing as tl
from aircraft_performance import units as u
from aircraft_performance.aerodynamics import ParabolicDragPolar
from aircraft_performance.atmosphere import isa
from aircraft_performance.errors import InfeasibleFlightConditionError, InputError, PerformanceWarning

from conftest import RHO0_ROSKAM


# --------------------------------------------------------------------------- Breguet / Roskam Ch. 11
def test_roskam_example_11_1_propeller_range():
    """Roskam Example 11.1: 1,500 nm at (L/D)max = 15.8, eta = 0.87, c = 0.45 -> W_end = 25,805 lb."""
    pol = ParabolicDragPolar(0.02, 0.05)
    assert pol.max_lift_to_drag == pytest.approx(15.8, abs=0.02)
    c = u.bsfc_to_si(0.45)
    W0 = 30000 * u.LBF
    W1 = 25805 * u.LBF
    R = rng.breguet_range_propeller(0.87, c, pol.max_lift_to_drag, W0, W1)
    assert R / u.NM == pytest.approx(1500, rel=2e-3)


def test_roskam_example_11_2_jet_constant_mach():
    """Roskam Example 11.2: M = 0.75 at 35,000 ft, L/D = 14.8, c_j = 0.65 -> 2,247 nm and 5.2 h."""
    ct = u.tsfc_to_si(0.65)
    W0, W1 = 49000 * u.LBF, 39000 * u.LBF
    V = 0.75 * isa(35000 * u.FT).speed_of_sound
    assert rng.breguet_range_jet_constant_speed(V, ct, 14.8, W0, W1) / u.NM == pytest.approx(2247, rel=3e-3)
    assert rng.breguet_endurance_jet(ct, 14.8, W0, W1) / 3600 == pytest.approx(5.2, abs=0.02)


def test_roskam_example_11_3_jet_constant_altitude():
    """Roskam Example 11.3: max range at constant altitude, point A (CL = 0.34) -> 2,331 nm."""
    pol = ParabolicDragPolar(0.019, 0.055)
    ct = u.tsfc_to_si(0.65)
    rho = isa(35000 * u.FT).density
    A = pol.point_A
    assert A.CL == pytest.approx(0.34, abs=0.005)
    assert A.range_parameter_jet == pytest.approx(23.0, abs=0.05)
    R = rng.breguet_range_jet_constant_altitude(ct, rho, 500 * u.FT2, A.CL, A.CD, 49000 * u.LBF, 39000 * u.LBF)
    assert R / u.NM == pytest.approx(2331, rel=3e-3)


def test_anderson_example_5_20_endurance():
    """Anderson Example 5.20: E = (1/c_t)(L/D)max ln(73,000/43,500) = 38,969 s."""
    E = rng.breguet_endurance_jet(u.tsfc_to_si(0.69), ParabolicDragPolar(0.015, 0.08).max_lift_to_drag,
                                  73000 * u.LBF, 43500 * u.LBF)
    assert E == pytest.approx(38969, rel=2e-3)


def test_cruise_climb_altitude_capitolo9():
    """Capitolo9 exercise: from 4,000 m with W1/W0 = 1000/1100, the final altitude is ~4,894 m."""
    h1 = rng.cruise_climb_final_altitude(4000.0, 1100.0, 1000.0)
    assert h1 == pytest.approx(4894, abs=15)   # ambiance (geometric altitude) gives 4894.5 m


# --------------------------------------------------------------------------- numerical cruise vs closed forms
def _fuel_weights(ac, fraction=0.9):
    W0 = ac.weight
    return W0, W0 - fraction * ac.max_fuel_mass * 9.80665


def test_numerical_cruise_matches_breguet_propeller(single):
    W0, W1 = _fuel_weights(single)
    pol = single.polar()
    h = 2000.0
    res = rng.cruise(single, W0, W1, "constant_altitude_constant_CL", h, CL=pol.point_E.CL)
    R = rng.breguet_range_propeller(single.propulsion.propeller_efficiency, single.propulsion.bsfc,
                                    pol.max_lift_to_drag, W0, W1)
    assert res.range == pytest.approx(R, rel=1e-5)
    E = rng.breguet_endurance_propeller(single.propulsion.propeller_efficiency, single.propulsion.bsfc,
                                        pol.point_E.CL, pol.point_E.CD, isa(h).density,
                                        single.wing_area, W0, W1)
    assert res.endurance == pytest.approx(E, rel=1e-5)


def test_numerical_cruise_matches_breguet_jet(giv):
    W0, W1 = _fuel_weights(giv, 0.7)
    pol = giv.polar()
    h = 12000.0
    rho = isa(h).density
    CL = 0.5
    r1 = rng.cruise(giv, W0, W1, "constant_altitude_constant_CL", h, CL=CL)
    assert r1.range == pytest.approx(rng.breguet_range_jet_constant_altitude(
        giv.propulsion.tsfc, rho, giv.wing_area, CL, pol.drag_coefficient(CL), W0, W1), rel=1e-5)
    r3 = rng.cruise(giv, W0, W1, "constant_speed_constant_CL", h, CL=CL)
    assert r3.range == pytest.approx(rng.breguet_range_jet_constant_speed(
        r3.speed[0], giv.propulsion.tsfc, CL / pol.drag_coefficient(CL), W0, W1), rel=1e-5)
    assert r3.altitude[-1] > h
    V = r1.speed[0]
    r2 = rng.cruise(giv, W0, W1, "constant_altitude_constant_speed", h, speed=V)
    assert r2.range == pytest.approx(rng.range_constant_altitude_speed_jet(
        V, rho, giv.wing_area, pol, giv.propulsion.tsfc, W0, W1), rel=1e-5)
    # the cruise-climb is the most efficient of the three at the same initial condition
    assert r3.range > r1.range


def test_cruise_with_headwind_reduces_range(single):
    W0, W1 = _fuel_weights(single)
    calm = rng.cruise(single, W0, W1, "constant_altitude_constant_speed", 2000.0, speed=55.0)
    head = rng.cruise(single, W0, W1, "constant_altitude_constant_speed", 2000.0, speed=55.0, headwind=10.0)
    assert head.range == pytest.approx(calm.range * 45.0 / 55.0, rel=1e-9)
    assert head.endurance == pytest.approx(calm.endurance)


def test_cruise_input_errors(single):
    W0, W1 = _fuel_weights(single)
    with pytest.raises(InputError):
        rng.cruise(single, W0, W1, "constant_altitude_constant_CL", 2000.0)              # no speed nor CL
    with pytest.raises(InputError):
        rng.cruise(single, W1, W0, "constant_altitude_constant_CL", 2000.0, CL=0.5)      # weights swapped
    with pytest.raises(InputError):
        rng.cruise(single, W0, W1, "warp_speed", 2000.0, CL=0.5)
    with pytest.raises(InfeasibleFlightConditionError):
        rng.cruise(single, W0, W1, "constant_altitude_constant_speed", 2000.0, speed=90.0)  # too fast
    with pytest.raises(InfeasibleFlightConditionError):
        rng.cruise(single, W0, W1, "constant_altitude_constant_CL", 2000.0, CL=1.9)        # stall


def test_best_speeds_parabolic(giv):
    """Without wind and inside the envelope, the optima are at the polar points."""
    h = 11000.0
    W = 0.9 * giv.weight
    rho = isa(h).density
    V_E = float(np.sqrt(2 * W / (rho * giv.wing_area * giv.polar().point_E.CL)))
    assert rng.best_endurance_speed(giv, h, W)["V"] == pytest.approx(V_E, rel=1e-3)
    V_A = float(np.sqrt(2 * W / (rho * giv.wing_area * giv.polar().point_A.CL)))
    best = rng.best_range_speed(giv, h, W)
    assert best["V"] == pytest.approx(V_A, rel=1e-3)
    # a headwind increases the best-range speed (Roskam Fig. 11.7)
    assert rng.best_range_speed(giv, h, W, headwind=30.0)["V"] > best["V"]


def test_loiter_optimum_lift_coefficient(giv, turboprop):
    """Jets loiter at point E; propeller aircraft at point P, limited to V >= 1.2 V_S."""
    W0, W1 = _fuel_weights(giv, 0.3)
    assert rng.loiter(giv, W0, W1, 9000.0).CL[0] == pytest.approx(giv.polar().point_E.CL)
    W0, W1 = _fuel_weights(turboprop, 0.3)
    CL_P = turboprop.polar().point_P.CL
    CL_limit = turboprop.config().CL_max / 1.2 ** 2
    assert CL_P > CL_limit          # for this aircraft point P is too close to the stall
    with pytest.warns(PerformanceWarning):
        res = rng.loiter(turboprop, W0, W1, 3000.0)
    assert res.CL[0] == pytest.approx(CL_limit)


def test_payload_range(giv):
    pr = rng.payload_range(giv, 12000.0, CL=0.45)
    assert pr["A"]["range"] == 0.0
    assert 0.0 < pr["B"]["range"] <= pr["C"]["range"] < pr["D"]["range"]
    for key in "ABCD":
        assert pr[key]["takeoff_mass"] <= giv.mass + 1e-6
        assert pr[key]["fuel"] <= giv.max_fuel_mass + 1e-6


# --------------------------------------------------------------------------- take-off
@pytest.fixture
def roskam_ex_10_2():
    """Data of Roskam Example 10.2 (propeller twin, sea level)."""
    W, S = 4600 * u.LBF, 175 * u.FT2
    polar = ParabolicDragPolar.from_aspect_ratio(0.062, 7.0, 0.80)
    V_S = np.sqrt(2 * W / (RHO0_ROSKAM * S * 1.69))
    thrust = tl.linear_thrust_model(2000 * u.LBF, 1200 * u.LBF, 1.10 * V_S)   # Figure 10.9
    return dict(weight=W, wing_area=S, rho=RHO0_ROSKAM, CL_max_TO=1.69, polar_TO=polar,
                CL_ground=0.89, CD_ground=0.0862, mu=0.03, thrust=thrust, t_rotate=1.0)


def test_roskam_example_10_2(roskam_ex_10_2):
    r = tl.takeoff_roskam_approximate(**roskam_ex_10_2)
    ft = u.FT
    assert r.V_S / ft == pytest.approx(114.4, abs=0.2)
    assert r.S_NGR / ft == pytest.approx(870, rel=0.01)
    assert r.S_R / ft == pytest.approx(129, rel=0.01)
    assert r.delta_CL == pytest.approx(0.12, abs=0.005)
    assert r.theta_climb == pytest.approx(0.128, abs=0.006)   # Roskam reads T(V_LOF) = 1,150 lb from a graph
    assert r.S_TR / ft == pytest.approx(732, rel=0.02)
    assert r.total_distance / ft == pytest.approx(1754, rel=0.01)
    assert r.total_time == pytest.approx(22.2, rel=0.03)


def test_roskam_example_10_2_statistical():
    """FAR 23 statistical method: TOP23 = 153 -> S_TO = 1,593 ft (Roskam p. 468)."""
    W, S = 4600 * u.LBF, 175 * u.FT2
    res = tl.takeoff_distance_far23_statistical(W / S, W / (0.9 * 2 * 260 * u.HP), 1.0, 1.69)
    assert res["TOP23"] == pytest.approx(153, abs=0.5)
    assert res["takeoff_distance"] / u.FT == pytest.approx(1593, rel=3e-3)


def test_ground_roll_numerical_constant_acceleration():
    """With constant thrust and CD_g = mu CL_g the acceleration is constant: S = V^2 / (2a)."""
    W, S, rho, mu = 50000.0, 20.0, 1.225, 0.03
    T = 12000.0
    res = tl.takeoff_ground_roll_numerical(W, S, rho, 0.5, 0.5 * mu, mu, lambda V: T, 40.0)
    a = 9.80665 * (T / W - mu)
    assert res["distance"] == pytest.approx(40.0 ** 2 / (2 * a), rel=1e-5)
    assert res["time"] == pytest.approx(40.0 / a, rel=1e-5)


def test_ground_roll_numerical_vs_roskam_quadratic_thrust(roskam_ex_10_2):
    """If a(V) is exactly linear in V^2, Roskam's k-method (Eqn 10.29) is exact."""
    d = roskam_ex_10_2
    V_R = 1.10 * np.sqrt(2 * d["weight"] / (d["rho"] * d["wing_area"] * 1.69))
    T0, TR = 2000 * u.LBF, 1200 * u.LBF

    def thrust(V):
        return T0 + (TR - T0) * (V / V_R) ** 2
    d = dict(d, thrust=thrust)
    approx = tl.takeoff_roskam_approximate(**d)
    num = tl.takeoff_ground_roll_numerical(d["weight"], d["wing_area"], d["rho"], d["CL_ground"],
                                           d["CD_ground"], d["mu"], thrust, V_R)
    assert num["distance"] == pytest.approx(approx.S_NGR, rel=1e-4)


def test_takeoff_wind_and_slope(twin):
    base = tl.takeoff_distance(twin)
    assert tl.takeoff_distance(twin, headwind=5.0).total_distance < base.total_distance
    assert tl.takeoff_distance(twin, headwind=-5.0).total_distance > base.total_distance
    assert tl.takeoff_distance(twin, runway_slope=0.02).total_distance > base.total_distance
    assert tl.takeoff_distance(twin, altitude=1500.0).total_distance > base.total_distance


def test_takeoff_infeasible(twin):
    with pytest.raises(InfeasibleFlightConditionError):
        tl.takeoff_distance(twin, mu=0.45)


def test_far25_statistical_and_bfl(giv):
    TW = giv.propulsion.max_thrust_sl / giv.weight
    st = tl.takeoff_field_length_far25_statistical(giv.wing_loading, TW, 1.0, 1.86)
    assert st["field_length"] == pytest.approx(37.5 * st["TOP25"] * u.FT)
    T_mean = tl.mean_takeoff_thrust_jet(giv.propulsion.max_thrust_sl, 3.04)
    bfl = tl.balanced_field_length_torenbeek(giv.weight, giv.wing_area, isa(0).density, 1.86, T_mean, 2, 0.05)
    # Business jets of this class have BFL of the order of 1,500-2,000 m
    assert 1200.0 < bfl["BFL"] < 2500.0
    with pytest.warns(PerformanceWarning):
        tl.balanced_field_length_torenbeek(giv.weight, giv.wing_area, isa(0).density, 1.86, T_mean, 2, 0.01)


def test_transition_delta_cl_roskam():
    """Roskam p. 465: CL_max_TO = 2.0 and V_LOF = 1.2 V_S -> delta CL = 0.15."""
    assert tl.transition_delta_CL(2.0, 1.2) == pytest.approx(0.15, abs=0.01)


# --------------------------------------------------------------------------- landing
def test_roskam_example_10_3():
    W, S = 4600 * u.LBF, 175 * u.FT2
    polar = ParabolicDragPolar.from_aspect_ratio(0.100, 7.0, 0.80)
    r = tl.landing_roskam_approximate(W, S, RHO0_ROSKAM, 2.12, polar, mu_brake=0.40, CL_ground=0.40,
                                      CD_ground=0.30, approach_thrust=260 * u.LBF,
                                      ground_thrust=260 * u.LBF, n_flare=1.08, t_free_roll=1.0)
    ft = u.FT
    assert r.V_S / ft == pytest.approx(102.1, abs=0.2)
    assert np.degrees(r.gamma_A) == pytest.approx(5.4, abs=0.05)
    assert r.R_flare / ft == pytest.approx(6173, rel=3e-3)
    assert r.S_LA / ft == pytest.approx(821, rel=5e-3)
    assert r.S_LR / ft == pytest.approx(117, rel=5e-3)
    assert r.S_LNGR / ft == pytest.approx(601, rel=5e-3)
    assert r.total_distance / ft == pytest.approx(1539, rel=3e-3)
    assert r.far25_field_length == pytest.approx(r.total_distance / 0.6)


def test_landing_statistical_far23():
    """Roskam p. 502: V_S = 102.1 ft/s -> S_LG = 968 ft, S_L = 1,876 ft."""
    res = tl.landing_distance_far23_statistical(102.1 * u.FT)
    assert res["ground_roll"] / u.FT == pytest.approx(968, rel=3e-3)
    assert res["landing_distance"] / u.FT == pytest.approx(1876, rel=3e-3)


def test_landing_ground_roll_numerical_matches_closed_form():
    """No free roll, no nose load: the numerical roll equals Roskam Eqn (10.112)."""
    W, S, rho = 20000.0, 16.0, 1.225
    r = tl.landing_roskam_approximate(W, S, rho, 2.1, ParabolicDragPolar(0.08, 0.05), 0.4, 0.4, 0.25,
                                      approach_angle_deg=3.0, nose_load_ratio=0.0, t_free_roll=0.0)
    num = tl.landing_ground_roll_numerical(W, S, rho, r.V_TD, 0.4, 0.4, 0.25, t_free_roll=0.0,
                                           nose_load_ratio=0.0)
    assert num["distance"] == pytest.approx(r.S_LNGR, rel=1e-5)
    assert num["time"] == pytest.approx(r.t_LNGR, rel=1e-4)


def test_landing_aircraft_level(twin):
    base = tl.landing_distance(twin)
    assert tl.landing_distance(twin, headwind=5.0).total_distance < base.total_distance
    assert tl.landing_distance(twin, mu_brake=0.2).total_distance > base.total_distance


def test_landing_approach_too_much_thrust():
    with pytest.raises(InfeasibleFlightConditionError):
        tl.approach_angle(1.2, 0.12, 0.2)


# --------------------------------------------------------------------------- manoeuvring
def test_turn_relations():
    n = mnv.load_factor_from_bank_angle(60.0)
    assert n == pytest.approx(2.0)
    assert mnv.bank_angle_from_load_factor(2.0) == pytest.approx(60.0)
    V = 100.0
    assert mnv.turn_radius(V, n) == pytest.approx(V ** 2 / (9.80665 * np.sqrt(3.0)))
    assert mnv.turn_rate(V, n) * mnv.turn_radius(V, n) == pytest.approx(V)
    with pytest.raises(InputError):
        mnv.turn_radius(V, 1.0)


def test_sustained_turn_unity_at_max_speed(giv):
    from aircraft_performance.level_flight import level_flight_speeds
    s = level_flight_speeds(giv, 9000.0)
    assert mnv.sustained_load_factor(giv, s.V_max, 9000.0) == pytest.approx(1.0, rel=1e-5)


def test_anderson_max_turn_rate_load_factor(giv):
    """Anderson Example 6.2: the sustained load factor for max turn rate is n = 3.16 at sea level.

    For constant thrust the maximum of sqrt(n^2-1)/V occurs at V = (2 W/S / rho)^0.5 (K/CD0)^0.25.
    """
    W, S, rho = giv.weight, giv.wing_area, isa(0).density
    V = np.sqrt(2 * W / S / rho) * (0.08 / 0.015) ** 0.25
    n = mnv.sustained_load_factor(giv, V, 0.0)
    assert n == pytest.approx(3.16, abs=0.02)


def test_turn_performance_limits(giv):
    V = np.linspace(80.0, 250.0, 30)
    tp = mnv.turn_performance(giv, 0.0, V, n_limit=2.5)
    assert np.all(tp.n_instantaneous <= 2.5 + 1e-12)
    assert np.all(tp.n_sustained_limited <= tp.n_instantaneous + 1e-12)
    assert np.all(tp.radius("instantaneous") <= tp.radius("sustained") + 1e-9)


def test_far23_load_factors_and_vn():
    n_pos, n_neg = mnv.far23_limit_load_factors(1100 * 9.80665)
    assert n_pos == pytest.approx(3.8)        # light aircraft: capped at 3.8
    assert n_neg == pytest.approx(-1.52)
    vn = mnv.vn_maneuver_diagram(1100 * 9.80665, 16.0, 1.6, 1.0, n_pos, n_neg, 60.0, 75.0)
    assert vn["V_A"] == pytest.approx(vn["V_S"] * np.sqrt(n_pos))
    assert np.all(vn["n_upper"] >= vn["n_lower"])
