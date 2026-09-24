"""
Exceptions and warnings used throughout the ``aircraft_performance`` package.

Why a dedicated module?
-----------------------
Performance calculations fail in two very different ways:

1. The *inputs* are wrong (a negative wing area, an efficiency larger than one,
   an altitude outside the atmosphere model...). These are programming or data
   entry mistakes and must be reported immediately and loudly.

2. The inputs are fine, but the *physics* says "no": the aircraft cannot fly
   level at that altitude, the climb rate is negative, the take-off run never
   reaches lift-off speed... These are not bugs, they are legitimate
   engineering answers, and the caller may want to catch them (for instance to
   find a ceiling by bisection).

Using distinct exception classes lets a notebook (or a student) tell the two
situations apart, instead of receiving a silent ``nan`` that propagates
through every subsequent plot and table.
"""


class PerformanceError(Exception):
    """Base class for every error raised by this package."""


class InputError(PerformanceError, ValueError):
    """An input value is not acceptable (wrong sign, out of range, not finite...).

    It also derives from :class:`ValueError`, so generic code that already
    catches ``ValueError`` keeps working.
    """


class InfeasibleFlightConditionError(PerformanceError):
    """The requested flight condition cannot be achieved by the aircraft.

    Typical examples: level flight above the absolute ceiling, a steady climb
    with more drag than thrust, a take-off run in which the aircraft never
    accelerates up to the lift-off speed.
    """


class ConvergenceError(PerformanceError):
    """A numerical procedure (root finding, integration...) did not converge."""


class PerformanceWarning(UserWarning):
    """Warning for results that are computed but should be used with care.

    Examples: a Mach number beyond the drag-divergence Mach number (the
    parabolic polar is no longer representative), a lift coefficient beyond
    CL_max, a steep climb angle computed with small-angle formulas.
    """
