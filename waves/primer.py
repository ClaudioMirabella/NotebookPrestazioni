"""
A Manim primer: five short scenes, each introducing ONE core idea of Manim.

Read them in order; every line is commented. Render one with, for example,

    manim -pql primer.py P1_MobjectsAndAnimations

    -p   = play the video when it is ready
    -ql  = quality low (480p, 15 frames/s): fast, ideal while developing
    -qm / -qh = medium (720p30) / high (1080p60) quality, for the final video

The five ideas
--------------
P1  Mobjects and animations: what you draw (Mobject) vs how it changes (Animation),
    the ``self.play`` / ``self.wait`` timeline, positioning.
P2  Mathematics on screen: ``MathTex`` (LaTeX), colouring parts of a formula,
    morphing one formula into another.
P3  Axes and graphs: coordinate systems, plotting functions, converting between
    "math coordinates" and "screen coordinates".
P4  Things that change continuously: ``ValueTracker``, updaters and ``always_redraw``.
    This is THE key technique for animating waves.
P5  Choreography: the ``.animate`` syntax, rate functions (easing), and playing
    several animations together or one after the other.

Two facts that help everything else make sense
----------------------------------------------
* The screen is a coordinate system: the frame is 14.2 units wide and 8 units
  tall, with (0, 0) at the centre. Points are NumPy arrays [x, y, z]; Manim
  provides the unit vectors UP, DOWN, LEFT, RIGHT (and UL, UR, DL, DR...).
* A Scene is a timeline: ``construct`` runs from top to bottom, and only
  ``self.play(...)`` and ``self.wait(...)`` make time pass in the video.
  Everything else (creating, positioning objects) takes zero video time.
"""

from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, PI, RIGHT, UP, AnimationGroup, Axes, Circle,
                   Create, DecimalNumber, Dot, FadeIn, FadeOut, GrowArrow, Arrow,
                   LaggedStart, MathTex, Square, Text, Transform, TransformMatchingTex,
                   ValueTracker, VGroup, Write, always_redraw, linear, smooth,
                   there_and_back)

# The shared style of the series (background colour, palette, helpers).
from wavekit import HIGHLIGHT, IMAG, MUTED, REAL, WAVE, WaveScene


class P1_MobjectsAndAnimations(WaveScene):
    """Mobjects (things on screen) and Animations (how they appear or change)."""

    def construct(self):
        # 1. Create mobjects. Creating them does NOT show them yet.
        circle = Circle(radius=1.2, color=WAVE)            # a circle of radius 1.2 units
        square = Square(side_length=2.0, color=REAL)       # a square, 2 units per side
        label = Text("a Mobject", font_size=30)            # plain text (no LaTeX needed)

        # 2. Position them. Methods return the mobject itself, so they can be chained.
        circle.shift(LEFT * 3)                             # move 3 units to the left
        square.shift(RIGHT * 3)                            # move 3 units to the right
        label.next_to(circle, DOWN, buff=0.3)              # place just below the circle

        # 3. Animate them onto the screen. self.play makes video time pass.
        self.play(Create(circle))                          # draw the outline progressively
        self.play(Write(label))                            # handwriting effect for text
        self.play(FadeIn(square, shift=UP))                # fade in while sliding upwards

        # 4. Transform one mobject into the shape of another.
        #    After a Transform, the object on screen is still `circle`
        #    (it now just looks like a square).
        self.play(Transform(circle, square.copy().shift(LEFT * 6)), run_time=2)

        self.wait(1)                                       # hold the frame for 1 second
        self.play(FadeOut(circle), FadeOut(square), FadeOut(label))


class P2_MathOnScreen(WaveScene):
    """Formulas with MathTex: LaTeX, coloured parts, morphing formulas."""

    def construct(self):
        # MathTex compiles LaTeX (math mode) into a vector drawing.
        # Splitting the formula into several strings creates separate PARTS
        # (sub-mobjects) that can be coloured or animated individually:
        #   part 0: "y(x,t)"   part 1: "="   part 2: "A"   part 3: "\sin(kx - \omega t)"
        wave = MathTex(r"y(x,t)", "=", "A", r"\sin(kx - \omega t)", font_size=60)
        wave[2].set_color(REAL)                             # colour the amplitude A
        wave[3].set_color(WAVE)                             # colour the sine
        self.play(Write(wave))
        self.wait(1)

        # TransformMatchingTex morphs a formula into another one, moving the
        # parts that are identical (same LaTeX string) and fading the others.
        moved = MathTex(r"y(x,t)", "=", "A", r"\sin(kx + \omega t)", font_size=60)
        moved[2].set_color(REAL)
        moved[3].set_color(HIGHLIGHT)
        self.play(TransformMatchingTex(wave, moved), run_time=2)
        caption = Text("+ ωt: the wave now travels to the left", font_size=28, color=MUTED)
        caption.next_to(moved, DOWN, buff=0.6)
        self.play(FadeIn(caption))
        self.wait(1.5)
        self.play(FadeOut(moved), FadeOut(caption))


class P3_AxesAndGraphs(WaveScene):
    """Coordinate systems, plotted functions, and the coordinate conversion c2p."""

    def construct(self):
        # Axes(x_range=[min, max, tick step], y_range=[...], size on screen)
        axes = Axes(
            x_range=[0, 2 * PI, PI / 2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10, y_length=4,                        # size on screen, in units
            axis_config={"color": MUTED, "include_tip": False},
        )
        labels = axes.get_axis_labels(MathTex("x"), MathTex("y"))
        self.play(Create(axes), FadeIn(labels))

        # plot takes an ordinary Python function of ONE number (math coordinates)
        sine = axes.plot(lambda x: np.sin(x), color=WAVE)
        self.play(Create(sine), run_time=2)

        # c2p = "coordinates to point": converts math coordinates (x, y) into a
        # screen position. It is how we put things exactly ON the graph.
        peak = Dot(axes.c2p(PI / 2, 1.0), color=HIGHLIGHT)
        peak_label = MathTex(r"(\tfrac{\pi}{2},\, 1)", font_size=36).next_to(peak, UP)
        self.play(FadeIn(peak, scale=2), Write(peak_label))
        self.wait(1.5)
        self.play(*(FadeOut(m) for m in (axes, labels, sine, peak, peak_label)))


class P4_ValueTrackerAndUpdaters(WaveScene):
    """Continuous change: ValueTracker + always_redraw (the heart of wave animations)."""

    def construct(self):
        axes = Axes(x_range=[0, 4 * PI, PI], y_range=[-1.5, 1.5, 0.5],
                    x_length=11, y_length=4,
                    axis_config={"color": MUTED, "include_tip": False})

        # A ValueTracker is an invisible number that CAN BE ANIMATED.
        # Here it stores the time t of a travelling wave.
        t = ValueTracker(0.0)

        # always_redraw(f) calls the function f at EVERY frame and shows the
        # mobject it returns. Because f reads t.get_value(), the drawing follows t.
        wave = always_redraw(
            lambda: axes.plot(lambda x: np.sin(x - t.get_value()), color=WAVE)
        )

        # A dot that rides on the wave at x = PI: an updater is a function
        # called at every frame with the mobject as argument.
        rider = Dot(color=HIGHLIGHT)
        rider.add_updater(lambda dot: dot.move_to(axes.c2p(PI, np.sin(PI - t.get_value()))))

        # A live read-out of t: DecimalNumber displays a number; its updater
        # copies the current value of the tracker.
        readout = DecimalNumber(0.0, num_decimal_places=2, font_size=36)
        readout.add_updater(lambda number: number.set_value(t.get_value()))
        t_label = VGroup(MathTex("t =", font_size=36), readout).arrange(RIGHT)
        t_label.to_edge(UP)

        self.add(axes, wave, rider, t_label)               # add = show instantly (no animation)
        # Animate the TRACKER: the wave, the dot and the read-out follow it.
        # rate_func=linear keeps the speed constant (no easing in/out).
        self.play(t.animate.set_value(2 * PI), run_time=5, rate_func=linear)
        self.wait(0.5)
        # Updaters keep running until removed; clear them before fading out.
        rider.clear_updaters()
        readout.clear_updaters()
        self.play(*(FadeOut(m) for m in (axes, wave, rider, t_label)))


class P5_Choreography(WaveScene):
    """The .animate syntax, rate functions, and grouping animations."""

    def construct(self):
        dots = VGroup(*(Dot(radius=0.15, color=WAVE) for _ in range(7)))
        dots.arrange(RIGHT, buff=0.6)                      # a row of 7 dots

        # LaggedStart: the same kind of animation on several objects, each one
        # starting a little after the previous (lag_ratio = fraction of overlap).
        self.play(LaggedStart(*(FadeIn(d, scale=3) for d in dots), lag_ratio=0.15))

        # .animate turns any method call into an animation: here, each dot moves
        # up and back down (rate_func=there_and_back), with increasing delay.
        # Together they look like a pulse travelling along the row: a wave!
        self.play(LaggedStart(*(d.animate(rate_func=there_and_back).shift(UP)
                                for d in dots), lag_ratio=0.2), run_time=3)

        # AnimationGroup plays animations SIMULTANEOUSLY; each may have its own
        # rate function. `smooth` (default) eases in and out, `linear` does not.
        arrow = Arrow(LEFT * 3, RIGHT * 3, color=IMAG, buff=0).next_to(dots, DOWN, buff=0.5)
        self.play(AnimationGroup(GrowArrow(arrow, rate_func=linear),
                                 dots.animate(rate_func=smooth).set_color(REAL)), run_time=2)
        self.wait(1)
        self.play(FadeOut(dots), FadeOut(arrow))
