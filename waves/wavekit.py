"""
wavekit - the shared toolkit of the "Mathematics of Waves" video series.

Every episode imports this module, so all videos share the same look (colours,
fonts, title cards) and the same way of pacing the commentary.

Contents
--------
* the COLOUR PALETTE: one colour per physical/mathematical role, used in every
  episode, so that viewers learn to read the colours (e.g. "yellow = real part");
* ``WaveScene``: a ``manim.Scene`` subclass with a few helper methods:
    - ``narrate``        shows nothing on screen, but writes one line of the
                         commentary to the subtitle file (.srt) and keeps the
                         animation on screen long enough to read it aloud;
    - ``title_card``     the opening card of an episode;
    - ``section_title``  a small heading in the top-left corner;
    - ``key_idea``       a framed formula or sentence to remember;
    - ``clear_screen``   fades out everything currently on screen.

Why a base class (and not a ``manim.cfg`` file)?
    Manim reads ``manim.cfg`` only from the folder in which the ``manim`` command
    is launched. Putting the style in Python code makes it work wherever the
    scenes are rendered from, and keeps the style visible and documented.

How the commentary works
    Manim can write a subtitle file next to the video: ``Scene.add_subcaption``
    adds one entry at the current time of the animation. ``narrate`` uses it to
    store the spoken text, and then waits for the time needed to say it
    (at ``WORDS_PER_SECOND``). The result is a video whose pacing matches the
    script, plus an .srt file that can be shown as subtitles or read aloud to
    record a voice-over.
"""

from __future__ import annotations

from manim import (DOWN, UL, UP, Animation, FadeIn, FadeOut, ManimColor,
                   Mobject, Scene, SurroundingRectangle, Text, VGroup, Write)

# ---------------------------------------------------------------------------
# Colour palette (hexadecimal RGB strings, like in HTML/CSS)
# ---------------------------------------------------------------------------
# A dark blue-grey background is easier on the eyes than pure black and makes
# the coloured elements stand out.
BACKGROUND = ManimColor("#14181f")
TEXT_COLOR = ManimColor("#e8e6e3")       # main text: warm off-white
MUTED = ManimColor("#8a93a3")            # secondary text, axes, guide lines
WAVE = ManimColor("#4fc3f7")             # waves and wave functions: light blue
REAL = ManimColor("#ffd54f")             # real parts / x components: amber
IMAG = ManimColor("#f06292")             # imaginary parts / y components: pink
VECTOR_A = ManimColor("#81c784")         # first vector: green
VECTOR_B = ManimColor("#ba68c8")         # second vector: violet
HIGHLIGHT = ManimColor("#ff8a65")        # things to notice: coral

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
TITLE_SIZE = 56          # font size of episode titles
MAX_TEXT_WIDTH = 12.5    # widest text block allowed on screen [units] (frame: 14.2)
HEADING_SIZE = 34        # section headings
BODY_SIZE = 30           # sentences on screen
MATH_SIZE = 44           # formulas

# ---------------------------------------------------------------------------
# Pacing of the commentary
# ---------------------------------------------------------------------------
# A relaxed, conversational speaking rate is about 150 words per minute,
# i.e. 2.5 words per second. All waits are derived from this single number,
# so the whole video can be made faster or slower by changing it.
WORDS_PER_SECOND = 2.5
MIN_LINE_DURATION = 1.5  # even a very short line stays at least this long [s]
PAUSE_AFTER_LINE = 0.3   # breathing space after each line [s]


def speaking_time(text: str) -> float:
    """Seconds needed to say ``text`` aloud at ``WORDS_PER_SECOND``."""
    words = len(text.split())
    return max(words / WORDS_PER_SECOND, MIN_LINE_DURATION)


class WaveScene(Scene):
    """Base class of every scene of the series.

    Subclasses override ``construct`` exactly as with a plain ``Scene``; the
    helper methods below are simply available as ``self.narrate(...)`` etc.
    """

    def setup(self) -> None:
        """Called by Manim once, before ``construct`` (a standard Scene hook)."""
        self.camera.background_color = BACKGROUND

    # --- commentary ------------------------------------------------------------
    def narrate(self, text: str, *animations: Animation, run_time: float | None = None) -> None:
        """Say one line of the commentary, optionally while playing animations.

        Parameters
        ----------
        text
            The spoken sentence. It goes into the subtitle file (.srt).
        animations
            Zero or more animations to play while the sentence is spoken.
        run_time
            Duration of the animations. By default they last as long as the
            sentence (but at most 3 s, so that they do not feel sluggish).

        The total time spent here is at least the speaking time of ``text``:
        if the animations are shorter, the scene waits for the remainder.
        """
        duration = speaking_time(text)
        # One subtitle entry starting NOW (at self.time) and lasting `duration`.
        self.add_subcaption(text, duration=duration)
        elapsed = 0.0
        if animations:
            elapsed = run_time if run_time is not None else min(duration, 3.0)
            self.play(*animations, run_time=elapsed)
        remaining = duration - elapsed
        if remaining > 0:
            self.wait(remaining)
        self.wait(PAUSE_AFTER_LINE)

    # --- recurring visual elements -------------------------------------------------
    def title_card(self, episode: str, title: str, subtitle: str) -> None:
        """Opening card: episode number, title and subtitle, then fade out."""
        number = Text(episode, font_size=HEADING_SIZE, color=MUTED)
        main = Text(title, font_size=TITLE_SIZE, color=TEXT_COLOR)
        # Long titles would run off the screen: shrink them to the maximum width.
        if main.width > MAX_TEXT_WIDTH:
            main.scale_to_fit_width(MAX_TEXT_WIDTH)
        sub = Text(subtitle, font_size=BODY_SIZE, color=WAVE)
        # VGroup groups mobjects so they can be positioned/animated together;
        # .arrange stacks them vertically (DOWN) with a gap of `buff` units.
        card = VGroup(number, main, sub).arrange(DOWN, buff=0.4)
        self.play(FadeIn(number, shift=UP * 0.3), Write(main), run_time=2)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(card))

    def section_title(self, text: str) -> Text:
        """Write a heading in the top-left corner and return it (to remove it later)."""
        heading = Text(text, font_size=HEADING_SIZE, color=MUTED)
        # to_corner(UL) = place in the Upper-Left corner, `buff` units from the edges
        heading.to_corner(UL, buff=0.4)
        self.play(Write(heading), run_time=1)
        return heading

    def key_idea(self, content: Mobject, color: ManimColor = HIGHLIGHT) -> VGroup:
        """Frame ``content`` with a rounded box: "remember this!"."""
        box = SurroundingRectangle(content, color=color, buff=0.25, corner_radius=0.15)
        self.play(FadeIn(box, scale=1.1))
        return VGroup(content, box)

    def clear_screen(self, run_time: float = 1.0) -> None:
        """Fade out every mobject currently on screen.

        Updaters are removed FIRST. A mobject made with ``always_redraw`` is rebuilt
        at every frame: if it changes while it fades out (e.g. a curve gaining
        points), the fade compares two shapes with different numbers of points and
        Manim fails. Freezing everything first makes the fade-out safe.
        """
        for mobject in self.mobjects:
            mobject.clear_updaters()           # also clears the updaters of sub-mobjects
        if self.mobjects:
            self.play(*(FadeOut(m) for m in self.mobjects), run_time=run_time)
        self.clear()                           # remove anything that is left


__all__ = [
    "BACKGROUND", "TEXT_COLOR", "MUTED", "WAVE", "REAL", "IMAG", "VECTOR_A", "VECTOR_B",
    "HIGHLIGHT", "TITLE_SIZE", "MAX_TEXT_WIDTH", "HEADING_SIZE", "BODY_SIZE", "MATH_SIZE",
    "WORDS_PER_SECOND", "speaking_time", "WaveScene",
]
