"""
Episode 1 - Growth and decay: your first differential equation.

MATH 331 - Differential Equations (units U1, U2, U4). Course text: J. Lebl, *Notes on Diffy Qs*
(Sections 0.2, 1.2, 1.3); radioactive decay and dating as in Adkins and Davidson, *Ordinary
Differential Equations* (Springer, 2012), Section 1.3. Companion notebooks: 01 and 02.

Sections (methods of ``Episode01``; each can be rendered alone, e.g. ``Ep01Decay``):
    intro            what is a differential equation?
    the_model        bacteria: "rate proportional to population" becomes P' = kP
    slope_field      the slope field, solutions following it, solutions never cross
    solving          separation of variables, Lebl's Example 0.2.1 (100 -> 6400), doubling
    decay            N' = -lambda N, half-life, radiocarbon dating (75 % -> 2378 years)
    one_equation     one equation, two behaviours: the sign of k, stable vs unstable 0
    recap

Render (from this folder):
    manim -pql ep01_growth_decay.py Episode01      # preview of the whole episode
    python render.py ep01                          # final video + subtitles in output/
"""

from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, UR, Axes, Create, DashedLine, Dot, FadeIn,
                   LaggedStart, MathTex, ReplacementTransform, Text,
                   TransformMatchingTex, ValueTracker, VGroup, Write, always_redraw, linear)

from odekit import (BODY_SIZE, HIGHLIGHT, MATH_SIZE, MUTED, PARAMETER, SOLUTION, STABLE,
                    UNSTABLE, OdeScene, slope_field, solution_curve)

# Lebl, Example 0.2.1: 100 bacteria at t = 0 and 200 at t = 10 s
P0_BACTERIA = 100.0
K_BACTERIA = np.log(2) / 10          # growth rate [1/s]: doubling time 10 s
C14_HALF_LIFE = 5730.0               # carbon-14 half-life [years]


class Episode01(OdeScene):
    def construct(self):
        self.intro()
        self.the_model()
        self.slope_field_section()
        self.solving()
        self.decay()
        self.one_equation()
        self.recap()

    # =====================================================================================
    def intro(self):
        self.title_card("Episode 1", "Growth and Decay", "Differential Equations for Engineers")
        self.narrate("Welcome to our series on differential equations! Today we meet the most "
                     "important differential equation of all, and we will see it in two "
                     "disguises: growth and decay.")
        question = MathTex(r"\frac{dy}{dt} = f(t, y)", font_size=MATH_SIZE + 10)
        meaning = Text("an equation for an unknown FUNCTION, involving its derivative",
                       font_size=BODY_SIZE, color=MUTED).next_to(question, DOWN, buff=0.6)
        self.narrate("A differential equation is an equation for an unknown function, involving "
                     "its derivatives. Its solution is not a number: it is a whole function.",
                     Write(question), FadeIn(meaning))
        self.narrate("Let's see where such an equation comes from.")
        self.clear_screen()

    # =====================================================================================
    def the_model(self):
        self.section_title("1. A model for growth")
        # bacteria doubling: 1, 2, 4, ... 64 dots
        counter = MathTex("P = 1", font_size=MATH_SIZE).to_edge(RIGHT, buff=1.2).shift(UP * 2)
        colony = VGroup(Dot(radius=0.12, color=SOLUTION))
        colony.move_to(LEFT * 2.5 + UP * 0.6)
        self.narrate("Picture bacteria on a Petri dish, with plenty of food and space. Every "
                     "bacterium divides, again and again.", FadeIn(colony), Write(counter))
        for generation in range(1, 7):
            size = 2 ** generation
            columns = int(np.ceil(np.sqrt(size)))
            new_colony = VGroup(*(Dot(radius=0.12, color=SOLUTION) for _ in range(size)))
            new_colony.arrange_in_grid(cols=columns, buff=0.18).move_to(LEFT * 2.5 + UP * 0.6)
            new_counter = MathTex(f"P = {size}", font_size=MATH_SIZE).move_to(counter)
            # ReplacementTransform splits each dot into the new, larger group
            self.play(ReplacementTransform(colony, new_colony),
                      ReplacementTransform(counter, new_counter), run_time=0.9)
            colony, counter = new_colony, new_counter
        self.narrate("The more bacteria there are, the more new bacteria appear per second. "
                     "Twice the population, twice the growth.")

        law = MathTex(r"\text{rate of change}", "=", "k", r"\times", r"\text{population}",
                      font_size=MATH_SIZE).to_edge(DOWN, buff=0.9)
        law[2].set_color(PARAMETER)
        equation = MathTex(r"\frac{dP}{dt}", "=", "k", "P", font_size=MATH_SIZE + 12).move_to(law)
        equation[2].set_color(PARAMETER)
        self.narrate("In words: the rate of change of the population is proportional to the "
                     "population itself.", Write(law))
        self.narrate("In symbols: d P by d t equals k times P. The constant k is the growth rate. "
                     "This is our differential equation.", TransformMatchingTex(law, equation))
        self.key_idea(equation)
        self.narrate("It's first order, linear, and autonomous: the right-hand side does not "
                     "depend on time explicitly. Lebl's notes start exactly here, in section 0.2.")
        self.clear_screen()

    # =====================================================================================
    def slope_field_section(self):
        self.section_title("2. The slope field")
        axes = Axes(x_range=[0, 60, 10], y_range=[0, 7000, 1000], x_length=10, y_length=5.2,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": range(0, 61, 10)},
                    y_axis_config={"numbers_to_include": range(0, 7001, 2000)}).shift(DOWN * 0.4)
        labels = axes.get_axis_labels(MathTex("t"), MathTex("P"))
        field = slope_field(axes, lambda t, p: K_BACTERIA * p, t_step=5, y_step=583.33)
        self.narrate("Even before solving it, the equation tells us a lot. At every point, t and "
                     "P, it gives the slope of the solution passing through that point.",
                     Create(axes), FadeIn(labels))
        self.narrate("Draw a short segment with that slope at many points, and you get the slope "
                     "field.", LaggedStart(*(Create(s) for s in field), lag_ratio=0.01), run_time=3)
        start = Dot(axes.c2p(0, P0_BACTERIA), color=HIGHLIGHT)
        curve = solution_curve(axes, lambda t: P0_BACTERIA * np.exp(K_BACTERIA * t), 0, 60)
        self.narrate("A solution is a curve that follows the slopes everywhere. Start from 100 "
                     "bacteria at time zero, and just follow the arrows.",
                     FadeIn(start, scale=2), Create(curve), run_time=4)

        # a family of solutions, P0 animated by a ValueTracker
        p0 = ValueTracker(100.0)
        family = always_redraw(lambda: solution_curve(
            axes, lambda t: p0.get_value() * np.exp(K_BACTERIA * t), 0, 60, color=PARAMETER))
        self.add(family)
        self.narrate("Change the starting value and you get another solution. They never cross: "
                     "the equation has exactly one solution through each point.",
                     p0.animate.set_value(900), run_time=5)
        self.play(p0.animate.set_value(30), run_time=3, rate_func=linear)
        picard = Text("Picard's theorem: one and only one solution through each point",
                      font_size=24, color=MUTED).to_corner(UR, buff=0.45).shift(DOWN * 0.6)
        self.narrate("That's Picard's existence and uniqueness theorem at work, from section 1.2 "
                     "of the notes.", FadeIn(picard))
        self.clear_screen()

    # =====================================================================================
    def solving(self):
        self.section_title("3. Solving it")
        steps = [
            MathTex(r"\frac{dP}{dt} = kP"),
            MathTex(r"\frac{1}{P}\,\frac{dP}{dt} = k"),
            MathTex(r"\int_{0}^{t} \frac{P'(s)}{P(s)}\,ds = \int_{0}^{t} k\,ds"),
            MathTex(r"\ln\frac{P(t)}{P_0} = k\,t"),
            MathTex(r"P(t) = P_0\,e^{kt}"),
        ]
        notes = ["the equation", "divide by P (it is never zero if P0 is not zero)",
                 "integrate from 0 to t", "the left side is a logarithm", "exponentiate"]
        current = steps[0].scale(1.3).shift(UP * 1.2)
        note = Text(notes[0], font_size=26, color=MUTED).next_to(current, DOWN, buff=0.5)
        self.narrate("Now let's solve it, by separation of variables, section 1.3 of the notes.",
                     Write(current), FadeIn(note))
        lines = [
            "Divide both sides by P. That's allowed: by uniqueness, a solution that starts away "
            "from zero never touches zero.",
            "Integrate both sides from time zero to time t.",
            "The left-hand side is the derivative of the logarithm of P, so we get log of P over "
            "P zero equals k t.",
            "Exponentiate, and there it is: P equals P zero, e to the k t.",
        ]
        for step, text, line in zip(steps[1:], notes[1:], lines, strict=True):
            step.scale(1.3).move_to(current)
            new_note = Text(text, font_size=26, color=MUTED).next_to(step, DOWN, buff=0.5)
            self.narrate(line, ReplacementTransform(current, step), ReplacementTransform(note, new_note))
            current, note = step, new_note
        self.key_idea(current, color=SOLUTION)
        check = MathTex(r"P'(t) = k P_0 e^{kt} = k\,P(t)\ \checkmark", font_size=38)
        check.next_to(note, DOWN, buff=0.5)
        self.narrate("Always check: the derivative is k times P, and at time zero we get P zero. "
                     "It works.", Write(check))
        self.clear_screen()

        # Lebl, Example 0.2.1
        self.section_title("3. Solving it: an example")
        data = MathTex(r"P(0) = 100, \quad P(10) = 200 \quad\Rightarrow\quad k = \frac{\ln 2}{10}",
                       font_size=40).to_edge(UP, buff=1.1)
        axes = Axes(x_range=[0, 60, 10], y_range=[0, 7000, 1000], x_length=10, y_length=4.4,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": range(0, 61, 10)},
                    y_axis_config={"numbers_to_include": range(0, 7001, 2000)}).shift(DOWN * 0.9)
        curve = solution_curve(axes, lambda t: P0_BACTERIA * np.exp(K_BACTERIA * t), 0, 60)
        self.narrate("Lebl's first example: 100 bacteria at time zero, 200 after 10 seconds. "
                     "Plugging in gives k equal to log 2 over 10.", Write(data), Create(axes))
        self.narrate("So the population doubles every 10 seconds, whatever its size.", Create(curve))
        marks = VGroup()
        for i in range(1, 7):
            t, p = 10 * i, P0_BACTERIA * 2 ** i
            marks.add(Dot(axes.c2p(t, p), color=HIGHLIGHT, radius=0.06),
                      DashedLine(axes.c2p(t, 0), axes.c2p(t, p), color=MUTED, stroke_width=1.5))
        answer = MathTex(r"P(60) = 100 \cdot 2^6 = 6400", color=HIGHLIGHT, font_size=40)
        answer.next_to(axes.c2p(30, 5200), UP)
        self.narrate("Ten seconds, 200. Twenty, 400. Then 800, 1600, 3200... and after one minute, "
                     "6400 bacteria.", LaggedStart(*(FadeIn(m) for m in marks), lag_ratio=0.15),
                     Write(answer), run_time=4)
        self.narrate("Of course it's a model: real bacteria come in whole numbers, and the food "
                     "eventually runs out. But it's an excellent first approximation.")
        self.clear_screen()

    # =====================================================================================
    def decay(self):
        self.section_title("4. Radioactive decay")
        law = MathTex(r"\frac{dN}{dt}", "=", r"-\lambda", "N", font_size=MATH_SIZE + 8).to_edge(UP, buff=1.1)
        law[2].set_color(PARAMETER)
        self.narrate("Now flip the sign. Radioactive atoms decay at random, and the number of decays "
                     "per second is proportional to the number of atoms left.", Write(law))
        self.narrate("Same equation, with k equal to minus lambda. The solution is N zero, e to the "
                     "minus lambda t: exponential decay.")
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 1.1, 0.25], x_length=10, y_length=4.4,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": [1, 2, 3, 4]},
                    y_axis_config={"numbers_to_include": [0.25, 0.5, 0.75, 1]}).shift(DOWN * 0.8)
        t_label = Text("time, in half-lives", font_size=24, color=MUTED).next_to(axes, DOWN, buff=0.2)
        curve = solution_curve(axes, lambda s: 2.0**-s, 0, 4)
        self.narrate("Let's measure time in half-lives.", Create(axes), FadeIn(t_label), Create(curve))
        staircase = VGroup()
        for n in range(1, 4):
            staircase.add(DashedLine(axes.c2p(n, 0), axes.c2p(n, 2.0**-n), color=MUTED),
                          DashedLine(axes.c2p(0, 2.0**-n), axes.c2p(n, 2.0**-n), color=MUTED),
                          Dot(axes.c2p(n, 2.0**-n), color=HIGHLIGHT))
        self.narrate("After one half-life, half the atoms are left. After two, a quarter. After "
                     "three, an eighth.", LaggedStart(*(Create(m) for m in staircase), lag_ratio=0.2),
                     run_time=4)
        half = MathTex(r"\tau_{1/2} = \frac{\ln 2}{\lambda}", font_size=MATH_SIZE).to_edge(RIGHT, buff=1)
        half.shift(UP * 1.4)
        self.narrate("Setting N equal to half of N zero gives the half-life: log 2 over lambda. "
                     "It doesn't depend on how much you start with.", Write(half))
        self.key_idea(half)
        self.clear_screen()

        # radiocarbon dating (Adkins and Davidson, Example 8)
        self.section_title("4. Radiocarbon dating")
        axes = Axes(x_range=[0, 12000, 2000], y_range=[0, 1.05, 0.25], x_length=10, y_length=4.4,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": range(0, 12001, 4000)},
                    y_axis_config={"numbers_to_include": [0.25, 0.5, 0.75, 1]}).shift(DOWN * 0.6)
        lam = np.log(2) / C14_HALF_LIFE
        curve = solution_curve(axes, lambda t: np.exp(-lam * t), 0, 12000)
        caption = Text("fraction of carbon-14 left, versus years", font_size=24, color=MUTED)
        caption.next_to(axes, DOWN, buff=0.2)
        self.narrate("A famous application: carbon-14, with a half-life of 5730 years. Living things "
                     "keep a constant share of it; after death, it decays.", Create(axes), Create(curve),
                     FadeIn(caption))
        age = -np.log(0.75) / lam
        horizontal = DashedLine(axes.c2p(0, 0.75), axes.c2p(age, 0.75), color=HIGHLIGHT)
        vertical = DashedLine(axes.c2p(age, 0.75), axes.c2p(age, 0), color=HIGHLIGHT)
        result = MathTex(r"t = -\frac{\ln 0.75}{\lambda} \approx 2378\ \text{years}", color=HIGHLIGHT,
                         font_size=40).next_to(axes.c2p(6500, 0.8), UP)
        self.narrate("A piece of wood has 75 percent of the original carbon-14. Read the curve "
                     "backwards: it's about 2378 years old.", Create(horizontal), Create(vertical),
                     Write(result), run_time=3)
        caveat = Text("(real dating also corrects for changes of atmospheric carbon-14)",
                      font_size=22, color=MUTED).to_corner(UR, buff=0.45).shift(DOWN * 0.6)
        self.narrate("That's the example in Adkins and Davidson. Real laboratories also calibrate "
                     "for changes in the atmosphere, but the core is this one equation.", FadeIn(caveat))
        self.clear_screen()

    # =====================================================================================
    def one_equation(self):
        self.section_title("5. One equation, two behaviours")
        left = Axes(x_range=[0, 4, 1], y_range=[-2, 2, 1], x_length=5.5, y_length=4,
                    axis_config={"color": MUTED, "include_tip": False}).shift(LEFT * 3.4 + DOWN * 0.4)
        right = left.copy().shift(RIGHT * 6.8)
        growth = slope_field(left, lambda t, y: 0.8 * y, 0.4, 0.4)
        decay = slope_field(right, lambda t, y: -0.8 * y, 0.4, 0.4)
        title_l = MathTex(r"y' = k y,\ k > 0", font_size=38).next_to(left, UP)
        title_r = MathTex(r"y' = k y,\ k < 0", font_size=38).next_to(right, UP)
        curves_l = VGroup(*(solution_curve(left, lambda t, c=c: c * np.exp(0.8 * t), 0, 4, color=UNSTABLE,
                                           stroke_width=3) for c in (-0.3, -0.1, 0.1, 0.3)))
        curves_r = VGroup(*(solution_curve(right, lambda t, c=c: c * np.exp(-0.8 * t), 0, 4, color=STABLE,
                                           stroke_width=3) for c in (-1.8, -0.9, 0.9, 1.8)))
        self.narrate("Same equation, two behaviours, decided by the sign of k.",
                     Create(left), Create(right), Write(title_l), Write(title_r))
        self.narrate("Look at the solution y equal to zero. With k positive, nearby solutions run "
                     "away from it: zero is an unstable equilibrium.", FadeIn(growth), Create(curves_l),
                     run_time=3)
        self.narrate("With k negative, every solution is pulled back to zero: a stable equilibrium.",
                     FadeIn(decay), Create(curves_r), run_time=3)
        others = Text("also: compound interest, capacitor discharge (k = -1/RC), T - A in Newton cooling",
                      font_size=24, color=MUTED).to_edge(DOWN, buff=0.25)
        self.narrate("And the same equation describes compound interest, a capacitor discharging "
                     "through a resistor, and much more.", FadeIn(others))
        self.clear_screen()

    # =====================================================================================
    def recap(self):
        self.section_title("Recap")
        points = VGroup(
            MathTex(r"\text{model: rate} \propto \text{amount} \;\Rightarrow\; y' = k y"),
            MathTex(r"\text{solution: } y(t) = y_0\, e^{kt} \text{ (unique, by Picard)}"),
            MathTex(r"k > 0:\ \text{doubling time } \ln 2 / k"),
            MathTex(r"k = -\lambda < 0:\ \text{half-life } \ln 2 / \lambda"),
            MathTex(r"\text{but: real growth cannot go on forever}", color=HIGHLIGHT),
        )
        points.arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.9)
        lines = ["Let's recap. Rate proportional to amount gives y prime equals k y.",
                 "Its unique solution is y zero, e to the k t.",
                 "Growth doubles every log 2 over k.",
                 "Decay halves every log 2 over lambda.",
                 "But real populations cannot grow forever. Next time: the logistic equation. "
                 "See you there!"]
        for point, line in zip(points, lines, strict=True):
            self.narrate(line, FadeIn(point, shift=RIGHT * 0.3))
        self.clear_screen()


# Sections as standalone scenes (faster to render while editing)
class Ep01Intro(Episode01):
    def construct(self):
        self.intro()


class Ep01Model(Episode01):
    def construct(self):
        self.the_model()


class Ep01SlopeField(Episode01):
    def construct(self):
        self.slope_field_section()


class Ep01Solving(Episode01):
    def construct(self):
        self.solving()


class Ep01Decay(Episode01):
    def construct(self):
        self.decay()


class Ep01OneEquation(Episode01):
    def construct(self):
        self.one_equation()


class Ep01Recap(Episode01):
    def construct(self):
        self.recap()
