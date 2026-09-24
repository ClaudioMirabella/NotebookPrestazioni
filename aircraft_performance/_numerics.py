"""
Robust numerical building blocks shared by the performance modules.

Performance curves (excess power versus speed, maximum climb rate versus
altitude...) are smooth but not always unimodal over the whole search
interval, and a naive optimiser started in the wrong place may silently return
a local optimum or the interval end. The helpers below therefore always

1. sample the function on a grid (global, robust, easy to understand), then
2. refine the best grid point with a bracketed method (Brent),

and they raise :class:`~aircraft_performance.errors.ConvergenceError` instead
of returning a doubtful number.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .errors import ConvergenceError, InputError


def maximize_on_interval(func: Callable[[float], float], lower: float, upper: float,
                         n_grid: int = 300, log_spacing: bool = False,
                         xtol: float = 1e-8) -> Tuple[float, float]:
    """Global-then-local maximisation of a scalar function on [lower, upper].

    Returns
    -------
    (x_opt, f_opt) : tuple of float
    """
    if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
        raise InputError(f"Invalid search interval [{lower}, {upper}].")
    grid = (np.geomspace(lower, upper, n_grid) if log_spacing and lower > 0
            else np.linspace(lower, upper, n_grid))
    values = np.array([func(x) for x in grid], dtype=float)
    if not np.any(np.isfinite(values)):
        raise ConvergenceError("The function is not finite anywhere on the search interval.")
    values = np.where(np.isfinite(values), values, -np.inf)
    i = int(np.argmax(values))
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, n_grid - 1)]
    if b <= a:
        return float(grid[i]), float(values[i])
    res = minimize_scalar(lambda x: -func(x), bounds=(a, b), method="bounded",
                          options={"xatol": xtol * max(1.0, abs(b))})
    if not res.success:
        raise ConvergenceError(f"Bounded maximisation failed: {res.message}")
    x_opt, f_opt = float(res.x), float(-res.fun)
    # keep the grid value if the refinement was not better (flat maxima)
    if f_opt < values[i]:
        return float(grid[i]), float(values[i])
    return x_opt, f_opt


def root_in_bracket(func: Callable[[float], float], a: float, b: float,
                    what: str = "root", xtol: float = 1e-9) -> float:
    """Find a root of ``func`` in [a, b] with Brent's method.

    Raises
    ------
    ConvergenceError
        If ``func(a)`` and ``func(b)`` do not have opposite signs (the root is
        not bracketed) or Brent's method fails.
    """
    fa, fb = func(a), func(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ConvergenceError(f"Cannot bracket the {what}: function not finite at the interval ends.")
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if np.sign(fa) == np.sign(fb):
        raise ConvergenceError(
            f"The {what} is not bracketed in [{a:.6g}, {b:.6g}] "
            f"(f(a) = {fa:.4g}, f(b) = {fb:.4g}).")
    try:
        return float(brentq(func, a, b, xtol=xtol, maxiter=200))
    except (RuntimeError, ValueError) as exc:
        raise ConvergenceError(f"Brent's method failed while computing the {what}: {exc}") from exc
