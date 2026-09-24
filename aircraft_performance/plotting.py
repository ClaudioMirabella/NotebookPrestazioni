"""
Consistent, readable plot style for the notebooks (matplotlib).

Call :func:`apply_style` once at the top of a notebook. The categorical
colours are assigned in a fixed order and have been checked for colour-vision
deficiency separation; every multi-series plot should also carry a legend (and
where useful a different line style), so that no information is carried by
colour alone.
"""

from __future__ import annotations

#: Categorical colours, always used in this order (blue, orange, aqua, yellow, magenta).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
#: Line styles used as secondary encoding together with COLORS.
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))
#: Neutral colour for reference lines (limits, stall, ceilings...).
REFERENCE_COLOR = "#6b6b66"


def apply_style() -> None:
    """Apply the notebook plot style (thin lines, recessive grid, fixed colour order)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.figsize": (7.5, 4.5),
        "figure.dpi": 100,
        "axes.prop_cycle": cycler(color=COLORS) + cycler(linestyle=LINESTYLES),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8a85",
        "axes.labelcolor": "#2b2b28",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#e2e1dc",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "legend.frameon": False,
        "xtick.color": "#5b5b57",
        "ytick.color": "#5b5b57",
        "font.size": 10,
    })


def reference_line(ax, value, orientation: str = "v", label: str = None,
                   label_position: float = 1.0) -> None:
    """Draw a thin neutral reference line (e.g. stall speed, ceiling) with an optional label.

    ``label_position`` is the position of the label along the line, as a fraction
    of the axis (use different values to avoid overlapping labels).
    """
    kw = dict(color=REFERENCE_COLOR, linewidth=1.0, linestyle=(0, (3, 3)))
    if orientation == "v":
        ax.axvline(value, **kw)
        if label:
            ax.annotate(label, (value, label_position), xycoords=("data", "axes fraction"),
                        xytext=(3, -12), textcoords="offset points", color=REFERENCE_COLOR, fontsize=9)
    else:
        ax.axhline(value, **kw)
        if label:
            ax.annotate(label, (label_position, value), xycoords=("axes fraction", "data"),
                        xytext=(-3, 3), textcoords="offset points", ha="right",
                        color=REFERENCE_COLOR, fontsize=9)
