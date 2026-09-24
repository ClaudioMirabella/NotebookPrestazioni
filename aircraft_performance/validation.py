"""
Small helpers that check the inputs of every public function.

Each helper either returns the (possibly converted) value or raises an
:class:`~aircraft_performance.errors.InputError` with a message that names the
offending quantity and shows its value. They accept scalars *and* NumPy arrays:
for arrays, every element is checked.

The goal is to make failures happen **at the point where the wrong number
enters the calculation**, with a readable message, rather than several cells
later as a mysterious ``nan`` or a ``RuntimeWarning: invalid value``.
"""

from __future__ import annotations

import numpy as np

from .errors import InputError


def _as_array(name: str, value) -> np.ndarray:
    """Convert ``value`` to a float array, raising InputError if impossible."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"'{name}' must be a number (or an array of numbers), got {value!r}.") from exc
    if arr.size == 0:
        raise InputError(f"'{name}' is empty.")
    return arr


def _return_like_input(arr: np.ndarray):
    """Return a Python float for 0-d arrays, the array otherwise."""
    return float(arr) if arr.ndim == 0 else arr


def ensure_finite(name: str, value):
    """Check that ``value`` contains no NaN or infinity."""
    arr = _as_array(name, value)
    if not np.all(np.isfinite(arr)):
        raise InputError(f"'{name}' must be finite, got {value!r}.")
    return _return_like_input(arr)


def ensure_positive(name: str, value):
    """Check that ``value`` is finite and strictly greater than zero."""
    arr = _as_array(name, ensure_finite(name, value))
    if not np.all(arr > 0.0):
        raise InputError(f"'{name}' must be strictly positive, got {value!r}.")
    return _return_like_input(arr)


def ensure_non_negative(name: str, value):
    """Check that ``value`` is finite and greater than or equal to zero."""
    arr = _as_array(name, ensure_finite(name, value))
    if not np.all(arr >= 0.0):
        raise InputError(f"'{name}' must be non-negative, got {value!r}.")
    return _return_like_input(arr)


def ensure_in_range(name: str, value, lower: float, upper: float, *,
                    include_lower: bool = True, include_upper: bool = True):
    """Check that ``lower <= value <= upper`` (bounds inclusive by default)."""
    arr = _as_array(name, ensure_finite(name, value))
    ok_low = arr >= lower if include_lower else arr > lower
    ok_up = arr <= upper if include_upper else arr < upper
    if not np.all(ok_low & ok_up):
        lb = "[" if include_lower else "("
        ub = "]" if include_upper else ")"
        raise InputError(f"'{name}' must lie in {lb}{lower}, {upper}{ub}, got {value!r}.")
    return _return_like_input(arr)


def ensure_fraction(name: str, value, *, allow_zero: bool = False):
    """Check that ``value`` is an efficiency-like number in (0, 1] (or [0, 1])."""
    return ensure_in_range(name, value, 0.0, 1.0, include_lower=allow_zero)


def ensure_greater(name_a: str, a: float, name_b: str, b: float) -> None:
    """Check that ``a > b`` (e.g. initial weight larger than final weight)."""
    if not a > b:
        raise InputError(f"'{name_a}' ({a!r}) must be greater than '{name_b}' ({b!r}).")


def ensure_result_finite(name: str, value):
    """Guard for *outputs*: raise if a computed result is not finite.

    This is the last line of defence against silent failures: if, despite the
    input checks, a formula produced NaN or infinity, we stop here instead of
    handing a meaningless number to the user.
    """
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        from .errors import PerformanceError
        raise PerformanceError(
            f"The computed quantity '{name}' is not finite ({value!r}). "
            "This usually means the inputs describe a physically impossible condition."
        )
    return _return_like_input(arr)
