"""
odelib - small, tested helpers for the MATH 331 notebooks on ordinary differential equations.

The notebooks keep the mathematics visible: every formula implemented here is derived
in the notebooks, with references to the course textbook,

    J. Lebl, *Notes on Diffy Qs: Differential Equations for Engineers* (CC BY-SA),
    file ``ode/FullText_Diff_Eq_For_Engineers.pdf`` (section numbers as in that edition).

Contents
--------
Analytical solutions
    exponential_solution          P(t) = P0 e^{k (t - t0)}                   (Lebl 0.2, 1.1)
    decay_constant, half_life     λ = ln 2 / τ                                (decay law)
    age_from_fraction             t = -ln(fraction) / λ                       (radiocarbon dating)
    logistic_solution             x' = k x (M - x)                            (Lebl 1.6)
    logistic_blowup_time          finite-time blow-up when x0 < 0
    harvesting_equilibria         x' = k x (M - x) - h                        (Lebl 1.6)
Numerical methods (fixed step)
    euler, heun, rk4              one-step methods for y' = f(t, y)           (Lebl 1.7)
    observed_order                order of convergence from successive errors
Plotting
    slope_field                   short segments of slope f(t, y)             (Lebl 1.2)
    phase_line                    critical points and arrows of x' = f(x)     (Lebl 1.6)

Error handling: invalid arguments raise ``ValueError`` (never a silent NaN), with one
documented exception: ``logistic_solution`` returns NaN for times at or beyond the
blow-up time, where the solution does not exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

ArrayLike = float | Sequence[float] | np.ndarray
RHS = Callable[[float, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Exponential growth and decay
# ---------------------------------------------------------------------------
def exponential_solution(t: ArrayLike, P0: float, k: float, t0: float = 0.0) -> np.ndarray:
    """Unique solution of P' = k P, P(t0) = P0:  P(t) = P0 exp(k (t - t0)).

    k > 0 gives growth (Malthus), k < 0 decay (k = -λ). Defined for every real t.
    """
    P0, k, t0 = _require_finite("P0", P0), _require_finite("k", k), _require_finite("t0", t0)
    return P0 * np.exp(k * (np.asarray(t, dtype=float) - t0))


def growth_rate_from_two_values(t1: float, P1: float, t2: float, P2: float) -> float:
    """k such that P(t) = C e^{kt} passes through (t1, P1) and (t2, P2): k = ln(P2/P1)/(t2 - t1).

    Requires P1, P2 of the same (nonzero) sign and t1 != t2.
    """
    if t1 == t2:
        raise ValueError("the two times must be different")
    if P1 == 0 or P2 == 0 or np.sign(P1) != np.sign(P2):
        raise ValueError("P1 and P2 must be nonzero and of the same sign (P never changes sign)")
    return float(np.log(P2 / P1) / (t2 - t1))


def decay_constant(half_life: float) -> float:
    """Decay constant λ = ln 2 / τ from the half-life τ > 0."""
    return float(np.log(2.0) / _require_positive("half_life", half_life))


def half_life(decay_const: float) -> float:
    """Half-life τ = ln 2 / λ from the decay constant λ > 0 (it does not depend on N0)."""
    return float(np.log(2.0) / _require_positive("decay_const", decay_const))


def age_from_fraction(fraction: float, half_life_value: float) -> float:
    """Elapsed time after which a fraction 0 < f <= 1 of the isotope remains: t = -ln(f) / λ."""
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError(f"the remaining fraction must be in (0, 1], got {fraction}")
    return float(-np.log(fraction) / decay_constant(half_life_value))


# ---------------------------------------------------------------------------
# Logistic equation  x' = k x (M - x)   (notation of Lebl, Section 1.6)
# ---------------------------------------------------------------------------
def logistic_blowup_time(x0: float, k: float, M: float, t0: float = 0.0) -> float:
    """Time t* at which the solution with x(t0) = x0 < 0 tends to -infinity.

    From the closed form (see ``logistic_solution``) the denominator
    x0 + (M - x0) e^{-kM(t - t0)} vanishes at  t* = t0 + ln((M - x0)/(-x0)) / (kM).
    Returns +inf when x0 >= 0 (the solution then exists for all t >= t0).
    """
    k, M = _require_positive("k", k), _require_positive("M", M)
    x0 = _require_finite("x0", x0)
    if x0 >= 0:
        return float("inf")
    return float(t0 + np.log((M - x0) / (-x0)) / (k * M))


def logistic_solution(t: ArrayLike, x0: float, k: float, M: float, t0: float = 0.0) -> np.ndarray:
    """Solution of the logistic IVP  x' = k x (M - x),  x(t0) = x0  (k, M > 0).

        x(t) = M x0 / (x0 + (M - x0) e^{-k M (t - t0)}).

    Valid for t >= t0 in all cases. For x0 < 0 the solution exists only up to the blow-up
    time t* (``logistic_blowup_time``); for t >= t* NaN is returned on purpose, since no
    solution exists there. Times t < t0 (backward in time) are also allowed, but then a
    solution with x0 > M blows up to +infinity in the past; such points are returned as NaN.
    """
    k, M = _require_positive("k", k), _require_positive("M", M)
    x0, t0 = _require_finite("x0", x0), _require_finite("t0", t0)
    t = np.asarray(t, dtype=float)
    denominator = x0 + (M - x0) * np.exp(-k * M * (t - t0))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = M * x0 / denominator
    # The denominator equals M > 0 at t = t0 and is monotonic in t, so the formula holds
    # exactly on the interval where it stays positive; elsewhere the solution does not exist.
    return np.where(denominator > 0, x, np.nan)


def harvesting_equilibria(k: float, M: float, h: float) -> tuple[float, ...]:
    """Critical points of x' = k x (M - x) - h  (h >= 0 constant harvesting rate).

    Roots of k x^2 - k M x + h = 0:  x = (kM ± sqrt((kM)^2 - 4hk)) / (2k)  (Lebl, Section 1.6).
    Returns (B, A) with B <= A, a single value when h = kM^2/4 (saddle-node), or () when
    h > kM^2/4 (no equilibria: the population always collapses).
    """
    k, M = _require_positive("k", k), _require_positive("M", M)
    h = float(h)
    if h < 0:
        raise ValueError("the harvesting rate h must be >= 0")
    discriminant = (k * M) ** 2 - 4 * h * k
    if np.isclose(discriminant, 0.0, atol=1e-12 * (k * M) ** 2):
        return (M / 2,)
    if discriminant < 0:
        return ()
    root = np.sqrt(discriminant)
    return ((k * M - root) / (2 * k), (k * M + root) / (2 * k))


# ---------------------------------------------------------------------------
# Fixed-step one-step methods for y' = f(t, y),  y(t0) = y0
# ---------------------------------------------------------------------------
def _grid(t_span: tuple[float, float], h: float) -> np.ndarray:
    """Uniform grid t0, t0 + h, ..., t1. The interval must contain a whole number of steps."""
    t0, t1 = (_require_finite("t0", t_span[0]), _require_finite("t1", t_span[1]))
    h = _require_positive("h", h)
    if t1 <= t0:
        raise ValueError("t_span must satisfy t0 < t1")
    n_steps = (t1 - t0) / h
    n = int(round(n_steps))
    if n < 1 or not np.isclose(n_steps, n, rtol=1e-9, atol=1e-12):
        raise ValueError(f"(t1 - t0)/h = {n_steps} must be a positive integer")
    return t0 + h * np.arange(n + 1)


def _integrate(step, f: RHS, t_span, y0, h):
    t = _grid(t_span, h)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    y = np.empty((t.size, y0.size))
    y[0] = y0
    for i in range(t.size - 1):
        with np.errstate(over="ignore", invalid="ignore"):   # overflow is detected just below
            y[i + 1] = step(f, t[i], y[i], h)
        if not np.all(np.isfinite(y[i + 1])):
            raise FloatingPointError(
                f"the numerical solution overflowed at t = {t[i + 1]:.6g} "
                "(blow-up of the exact solution, or an unstable step size)")
    return t, (y[:, 0] if y.shape[1] == 1 else y)


def _euler_step(f, t, y, h):
    return y + h * np.asarray(f(t, y), dtype=float)


def _heun_step(f, t, y, h):
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h, y + h * k1), dtype=float)
    return y + h * (k1 + k2) / 2


def _rk4_step(f, t, y, h):
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h / 2, y + h / 2 * k1), dtype=float)
    k3 = np.asarray(f(t + h / 2, y + h / 2 * k2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def euler(f: RHS, t_span: tuple[float, float], y0: ArrayLike, h: float):
    """Euler's method  y_{i+1} = y_i + h f(t_i, y_i)  (Lebl 1.7). Order 1. Returns (t, y)."""
    return _integrate(_euler_step, f, t_span, y0, h)


def heun(f: RHS, t_span: tuple[float, float], y0: ArrayLike, h: float):
    """Improved Euler (Heun): average of the slopes at both ends of the step. Order 2."""
    return _integrate(_heun_step, f, t_span, y0, h)


def rk4(f: RHS, t_span: tuple[float, float], y0: ArrayLike, h: float):
    """Classical fourth-order Runge-Kutta method. Order 4."""
    return _integrate(_rk4_step, f, t_span, y0, h)


def observed_order(errors: Sequence[float], ratio: float = 2.0) -> np.ndarray:
    """Observed orders log(e_i / e_{i+1}) / log(ratio) for step sizes divided by ``ratio``."""
    e = np.asarray(errors, dtype=float)
    if np.any(e <= 0):
        raise ValueError("errors must be positive")
    return np.log(e[:-1] / e[1:]) / np.log(ratio)


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib is imported lazily: odelib works without it)
# ---------------------------------------------------------------------------
def slope_field(ax, f: Callable[[np.ndarray, np.ndarray], np.ndarray], t_range: tuple[float, float],
                y_range: tuple[float, float], n: int = 21, color: str = "0.6", length: float = 0.7):
    """Draw the slope field of y' = f(t, y): at each grid point a short segment of slope f.

    All segments have the same length measured in grid cells (``length``), whatever the
    slope, so steep regions stay readable (Lebl, Section 1.2). ``f`` must accept NumPy
    arrays (it is evaluated on the whole grid at once).
    """
    from matplotlib.collections import LineCollection

    t = np.linspace(*t_range, n)
    y = np.linspace(*y_range, n)
    T, Y = np.meshgrid(t, y)
    S = np.asarray(f(T, Y), dtype=float) * np.ones_like(T)     # slopes (broadcast if constant)
    dt = (t_range[1] - t_range[0]) / (n - 1)                   # grid spacing in t
    dy = (y_range[1] - y_range[0]) / (n - 1)                   # grid spacing in y
    # Direction (1, S) measured in grid cells: (1/dt, S/dy). Normalise it to half the
    # requested length, then convert back to data units.
    u, v = 1.0 / dt, S / dy
    scale = (length / 2) / np.hypot(u, v)
    half_t, half_y = u * scale * dt, v * scale * dy
    segments = np.stack([np.stack([T - half_t, Y - half_y], axis=-1),
                         np.stack([T + half_t, Y + half_y], axis=-1)], axis=-2).reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=1.0))
    ax.set_xlim(*t_range)
    ax.set_ylim(*y_range)
    return ax


def phase_line(ax, f: Callable[[float], float], critical_points: Sequence[float],
               x_range: tuple[float, float], x_position: float = 0.0):
    """Vertical phase line of the autonomous equation x' = f(x) (Lebl, Section 1.6).

    Critical points are marked with a dot (filled = stable, open = unstable, half = semi-stable,
    decided from the sign of f just below and just above each point); arrows show the
    direction of motion between them.
    """
    lo, hi = x_range
    points = sorted(c for c in critical_points if lo < c < hi)
    ax.axvline(x_position, color="0.3", lw=1.5)
    edges = [lo, *points, hi]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        mid = (a + b) / 2
        sign = np.sign(f(mid))
        if sign != 0:
            ax.annotate("", xy=(x_position, mid + sign * (b - a) * 0.15),
                        xytext=(x_position, mid - sign * (b - a) * 0.15),
                        arrowprops={"arrowstyle": "-|>", "color": "C0", "lw": 2})
    eps = 1e-6 * max(1.0, hi - lo)
    for c in points:
        below, above = np.sign(f(c - eps)), np.sign(f(c + eps))
        if below > 0 and above < 0:
            style = {"markerfacecolor": "k"}                     # stable: arrows point in
        elif below < 0 and above > 0:
            style = {"markerfacecolor": "white"}                 # unstable: arrows point out
        else:
            style = {"markerfacecolor": "0.6"}                   # semi-stable
        ax.plot([x_position], [c], "o", color="k", markersize=9, **style)
    ax.set_ylim(lo, hi)
    ax.set_xticks([])
    return ax


def classify_equilibrium(fprime_at_point: float) -> str:
    """Linear stability of an equilibrium x* of x' = f(x) from f'(x*):
    'stable' if f'(x*) < 0, 'unstable' if f'(x*) > 0, 'inconclusive' if f'(x*) = 0."""
    if fprime_at_point < 0:
        return "stable"
    if fprime_at_point > 0:
        return "unstable"
    return "inconclusive"
