"""
Plain-text tables for the notebooks (no external dependency).

Example::

    print_table(["h [m]", "V_max [m/s]"], [[0, 62.1], [1000, 61.4]], formats=["{:.0f}", "{:.1f}"])
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


def format_table(headers: Sequence[str], rows: Iterable[Sequence], formats: Optional[Sequence[str]] = None) -> str:
    """Return a right-aligned text table. ``formats`` holds one format string per column."""
    rows = [list(r) for r in rows]
    ncol = len(headers)
    if formats is None:
        formats = ["{:.4g}"] * ncol
    if len(formats) != ncol or any(len(r) != ncol for r in rows):
        raise ValueError("headers, formats and every row must have the same number of columns.")

    def fmt(value, f):
        if isinstance(value, str):
            return value
        if value is None:
            return "-"
        return f.format(value)

    cells = [[fmt(v, f) for v, f in zip(r, formats)] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.rjust(w) for h, w in zip(headers, widths)), line]
    out += ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in cells]
    return "\n".join(out)


def print_table(headers: Sequence[str], rows: Iterable[Sequence], formats: Optional[Sequence[str]] = None) -> None:
    """Print :func:`format_table`."""
    print(format_table(headers, rows, formats))
