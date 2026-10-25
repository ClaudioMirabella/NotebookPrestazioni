"""
odekit - the shared toolkit of the MATH 331 video series on differential equations.

Every episode imports this module, so all videos share the same look (colours, fonts,
title cards), the same way of pacing the commentary, and the same drawing helpers.
It follows the design of ``waves/wavekit.py`` (see that folder's README for the ideas).

Contents
--------
* COLOUR PALETTE: one colour per mathematical role, the same in every episode;
* ``OdeScene``: a ``manim.Scene`` subclass with helper methods
    - ``narrate``        one line of commentary: written to the subtitle file (.srt) and
                         timed at ``WORDS_PER_SECOND``, optionally while animations play;
    - ``title_card``     opening card of an episode;
    - ``section_title``  heading in the top-left corner;
    - ``key_idea``       framed formula or sentence to remember;
    - ``clear_screen``   freezes every updater, then fades everything out;
* drawing helpers for ``Axes``:
    - ``slope_field``    short segments of slope f(t, y) at grid points (Lebl, Section 1.2);
    - ``solution_curve`` graph of an exact solution, clipped to the visible window.

Why the commentary lives in the code: ``Scene.add_subcaption`` writes a subtitle entry at the
current time of the animation, so the .srt file always matches the video. It can be published
as captions or read aloud to record a voice-over.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from manim import (DOWN, UL, UP, Animation, Axes, FadeIn, FadeOut, Line, ManimColor, Mobject,
                   Scene, SurroundingRectangle, Text, VGroup, VMobject, Write)

# ---------------------------------------------------------------------------
# Colour palette (hexadecimal RGB strings) - each colour has ONE meaning
# ---------------------------------------------------------------------------
BACKGROUND = ManimColor("#14181f")       # dark blue-grey background
TEXT_COLOR = ManimColor("#e8e6e3")       # main text
MUTED = ManimColor("#8a93a3")            # axes, secondary text, guide lines
SLOPE = ManimColor("#5d6675")            # slope-field segments (discreet)
SOLUTION = ManimColor("#4fc3f7")         # exact solutions: light blue
NUMERIC = ManimColor("#ffd54f")          # numerical approximations (Euler...): amber
STABLE = ManimColor("#81c784")           # stable equilibria: green
UNSTABLE = ManimColor("#f06292")         # unstable equilibria: pink
PARAMETER = ManimColor("#ba68c8")        # parameters being varied (k, M, h): violet
HIGHLIGHT = ManimColor("#ff8a65")        # things to notice: coral

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
TITLE_SIZE = 56
MAX_TEXT_WIDTH = 12.5    # widest text block on screen [units] (the frame is 14.2 wide)
HEADING_SIZE = 34
BODY_SIZE = 30
MATH_SIZE = 44

# ---------------------------------------------------------------------------
# Pacing of the commentary
# ---------------------------------------------------------------------------
WORDS_PER_SECOND = 2.5   # relaxed conversational rate (150 words per minute)
MIN_LINE_DURATION = 1.5  # shortest time a line stays [s]
PAUSE_AFTER_LINE = 0.3   # breathing space after each line [s]


def speaking_time(text: str) -> float:
    """Seconds needed to say ``text`` aloud at ``WORDS_PER_SECOND``."""
    return max(len(text.split()) / WORDS_PER_SECOND, MIN_LINE_DURATION)


class OdeScene(Scene):
    """Base class of every scene of the series."""

    def setup(self) -> None:
        """Called by Manim before ``construct``: apply the common background."""
        self.camera.background_color = BACKGROUND

    # --- commentary ----------------------------------------------------------------
    def narrate(self, text: str, *animations: Animation, run_time: float | None = None) -> None:
        """Say one line of the commentary, optionally while playing ``animations``.

        The line goes to the subtitle file; the scene lasts at least the speaking time of
        ``text``. Animations last ``run_time`` (default: the speaking time, at most 3 s).
        """
        duration = speaking_time(text)
        self.add_subcaption(text, duration=duration)
        elapsed = 0.0
        if animations:
            elapsed = run_time if run_time is not None else min(duration, 3.0)
            self.play(*animations, run_time=elapsed)
        if duration - elapsed > 0:
            self.wait(duration - elapsed)
        self.wait(PAUSE_AFTER_LINE)

    # --- recurring visual elements ----------------------------------------------------
    def title_card(self, episode: str, title: str, subtitle: str) -> None:
        number = Text(episode, font_size=HEADING_SIZE, color=MUTED)
        main = Text(title, font_size=TITLE_SIZE, color=TEXT_COLOR)
        if main.width > MAX_TEXT_WIDTH:                    # long titles: shrink to fit
            main.scale_to_fit_width(MAX_TEXT_WIDTH)
        sub = Text(subtitle, font_size=BODY_SIZE, color=SOLUTION)
        card = VGroup(number, main, sub).arrange(DOWN, buff=0.4)
        self.play(FadeIn(number, shift=UP * 0.3), Write(main), run_time=2)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(card))

    def section_title(self, text: str) -> Text:
        heading = Text(text, font_size=HEADING_SIZE, color=MUTED).to_corner(UL, buff=0.4)
        self.play(Write(heading), run_time=1)
        return heading

    def key_idea(self, content: Mobject, color: ManimColor = HIGHLIGHT) -> VGroup:
        box = SurroundingRectangle(content, color=color, buff=0.25, corner_radius=0.15)
        self.play(FadeIn(box, scale=1.1))
        return VGroup(content, box)

    def clear_screen(self, run_time: float = 1.0) -> None:
        """Freeze all updaters, then fade out everything.

        Objects built with ``always_redraw`` are rebuilt at every frame; if one changed shape
        during the fade-out, Manim would fail (two shapes with different numbers of points).
        """
        for mobject in self.mobjects:
            mobject.clear_updaters()
        if self.mobjects:
            self.play(*(FadeOut(m) for m in self.mobjects), run_time=run_time)
        self.clear()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def slope_field(axes: Axes, f: Callable[[float, float], float], t_step: float, y_step: float,
                length: float = 0.32, color: ManimColor = SLOPE) -> VGroup:
    """Short segments of slope f(t, y) at the grid points of ``axes`` (Lebl, Section 1.2).

    Every segment has the same length ``length`` ON SCREEN, whatever the slope; the slope
    is converted from math units to screen units with the axes' scales.
    """
    t_min, t_max = axes.x_range[:2]
    y_min, y_max = axes.y_range[:2]
    # screen units per math unit along each axis
    sx = axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]
    sy = axes.c2p(0, 1)[1] - axes.c2p(0, 0)[1]
    segments = VGroup()
    for t in np.arange(t_min + t_step / 2, t_max, t_step):
        for y in np.arange(y_min + y_step / 2, y_max, y_step):
            direction = np.array([sx, sy * f(t, y), 0.0])      # (dt, dy) on screen for dt = 1
            direction *= (length / 2) / np.linalg.norm(direction)
            center = axes.c2p(t, y)
            segments.add(Line(center - direction, center + direction, color=color, stroke_width=2))
    return segments


def solution_curve(axes: Axes, solution: Callable[[float], float], t_start: float, t_end: float,
                   color: ManimColor = SOLUTION, stroke_width: float = 4) -> VMobject:
    """Graph of ``solution`` on [t_start, t_end], cut where it leaves the y-range of ``axes``."""
    y_min, y_max = axes.y_range[:2]
    ts = np.linspace(t_start, t_end, 300)
    points = [axes.c2p(t, solution(t)) for t in ts if y_min <= solution(t) <= y_max]
    curve = VMobject(color=color, stroke_width=stroke_width)
    if len(points) >= 2:
        curve.set_points_smoothly(points)
    return curve
