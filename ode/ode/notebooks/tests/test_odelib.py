"""Tests of odelib against values printed in the reference textbooks.

Lebl = J. Lebl, Notes on Diffy Qs (ode/FullText_Diff_Eq_For_Engineers.pdf)
Adkins-Davidson = W.A. Adkins, M.G. Davidson, Ordinary Differential Equations, Springer 2012
Run from the ode/notebooks folder:  python -m pytest
"""

import numpy as np
import pytest

import odelib as ol


# --- exponential growth and decay --------------------------------------------------------
def test_lebl_example_0_2_1_bacteria():
    # 100 bacteria at t = 0, 200 at t = 10 s  ->  k = ln 2 / 10, P(60) = 6400, P(61) ~ 6859.35
    k = ol.growth_rate_from_two_values(0, 100, 10, 200)
    assert k == pytest.approx(np.log(2) / 10)
    assert ol.exponential_solution(60, 100, k) == pytest.approx(6400)
    assert ol.exponential_solution(61, 100, k) == pytest.approx(6859.35, abs=0.01)


def test_half_life_and_decay_constant_are_inverse():
    assert ol.half_life(ol.decay_constant(5730.0)) == pytest.approx(5730.0)
    # the amount halves after each half-life, whatever N0
    lam = ol.decay_constant(5730.0)
    for N0 in (1.0, 2.0, 37.5):
        assert ol.exponential_solution(5730.0, N0, -lam) == pytest.approx(N0 / 2)


def test_adkins_davidson_example_8_radiocarbon():
    # 75% of carbon-14 left, half-life 5730 years  ->  about 2378 years
    assert ol.age_from_fraction(0.75, 5730) == pytest.approx(2378, abs=0.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, 1.5])
def test_age_rejects_impossible_fractions(bad):
    with pytest.raises(ValueError):
        ol.age_from_fraction(bad, 5730)


# --- logistic equation x' = k x (M - x) -----------------------------------------------------
@pytest.mark.parametrize("x0", [0.5, 2.5, 5.0, 7.0])
def test_logistic_solution_satisfies_the_equation(x0):
    k, M = 0.1, 5.0                                   # Lebl Figure 1.6.2
    t = np.linspace(0, 20, 2001)
    x = ol.logistic_solution(t, x0, k, M)
    dxdt = np.gradient(x, t)
    residual = dxdt - k * x * (M - x)
    assert np.max(np.abs(residual[2:-2])) < 1e-4
    assert x[0] == pytest.approx(x0)
    assert ol.logistic_solution(1e3, x0, k, M) == pytest.approx(M)   # stable equilibrium M


def test_logistic_negative_initial_value_blows_up_in_finite_time():
    k, M, x0 = 0.1, 5.0, -0.5
    t_star = ol.logistic_blowup_time(x0, k, M)
    assert t_star == pytest.approx(np.log((M - x0) / -x0) / (k * M))
    assert ol.logistic_solution(0.999 * t_star, x0, k, M) < -100
    assert np.isnan(ol.logistic_solution(1.01 * t_star, x0, k, M))
    assert ol.logistic_blowup_time(1.0, k, M) == np.inf


def test_lebl_harvesting_equilibria():
    # Lebl Section 1.6: M = 8, k = 0.1;  h = 1 -> B ~ 1.55, A ~ 6.45;  h = 1.6 -> A = B = 4;  h = 2 -> none
    B, A = ol.harvesting_equilibria(0.1, 8, 1.0)
    assert (B, A) == (pytest.approx(1.55, abs=0.01), pytest.approx(6.45, abs=0.01))
    assert ol.harvesting_equilibria(0.1, 8, 1.6) == (pytest.approx(4.0),)
    assert ol.harvesting_equilibria(0.1, 8, 2.0) == ()


# --- numerical methods ------------------------------------------------------------------------
LEBL_TABLE_1_7_1 = {  # Euler for y' = y^2/3, y(0) = 1, approximations of y(2) (exact value 3)
    1: 1.92593, 0.5: 2.20861, 0.25: 2.47250, 0.125: 2.68034,
    0.0625: 2.82040, 0.03125: 2.90412, 0.015625: 2.95035, 0.0078125: 2.97472,
}


@pytest.mark.parametrize(("h", "value"), LEBL_TABLE_1_7_1.items())
def test_euler_reproduces_lebl_table_1_7_1(h, value):
    # Tolerance: one unit in the 5th decimal. For h = 0.25 Euler gives 2.472494, which rounds
    # to 2.47249; the book prints 2.47250 (its own error column, 0.52751, agrees with 2.472494).
    _, y = ol.euler(lambda x, y: y**2 / 3, (0.0, 2.0), 1.0, h)
    assert y[-1] == pytest.approx(value, abs=1e-5)


def test_lebl_exercise_1_7_9_improved_euler():
    # y' = x + y, y(0) = 1, exact y = 2e^x - x - 1, y(1) = 2e - 2 ~ 3.4366
    f = lambda x, y: x + y                                            # noqa: E731
    exact = 2 * np.e - 2
    assert ol.heun(f, (0, 1), 1.0, 0.25)[1][-1] == pytest.approx(3.3897, abs=1e-4)
    assert ol.heun(f, (0, 1), 1.0, 0.125)[1][-1] == pytest.approx(3.4237, abs=1e-4)
    assert ol.euler(f, (0, 1), 1.0, 0.25)[1][-1] == pytest.approx(2.8828, abs=1e-4)
    assert ol.euler(f, (0, 1), 1.0, 0.125)[1][-1] == pytest.approx(3.1316, abs=1e-4)
    assert exact - ol.heun(f, (0, 1), 1.0, 0.25)[1][-1] == pytest.approx(0.046852, abs=1e-6)


@pytest.mark.parametrize(("method", "order"), [(ol.euler, 1), (ol.heun, 2), (ol.rk4, 4)])
def test_observed_orders_of_accuracy(method, order):
    steps = [0.1 / 2**i for i in range(5)]
    errors = [abs(method(lambda t, y: y, (0, 1), 1.0, h)[1][-1] - np.e) for h in steps]
    assert ol.observed_order(errors)[-1] == pytest.approx(order, abs=0.1)


def test_euler_detects_blow_up_instead_of_returning_garbage():
    # Lebl Table 1.7.2: y' = y^2/3 has no solution at x = 3 (it blows up there)
    with pytest.raises(FloatingPointError):
        ol.euler(lambda x, y: y**2 / 3, (0, 30), 1.0, 0.5)


def test_grid_must_contain_whole_number_of_steps():
    with pytest.raises(ValueError):
        ol.euler(lambda t, y: y, (0, 1), 1.0, 0.3)


def test_systems_are_supported():
    # x'' = -x as a system: (x, v)' = (v, -x); after one period the state returns to the start
    t, y = ol.rk4(lambda t, u: np.array([u[1], -u[0]]), (0, 2 * np.pi), [1.0, 0.0], 2 * np.pi / 1000)
    assert y[-1] == pytest.approx([1.0, 0.0], abs=1e-9)


def test_classify_equilibrium():
    assert ol.classify_equilibrium(-0.5) == "stable"
    assert ol.classify_equilibrium(0.5) == "unstable"
    assert ol.classify_equilibrium(0.0) == "inconclusive"
