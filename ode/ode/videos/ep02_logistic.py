"""
Episode 2 - The logistic equation: limits to growth.

MATH 331 - Differential Equations (unit U4: logistic equation, autonomous equations, phase line).
Course text: J. Lebl, *Notes on Diffy Qs*, Section 1.6 (notation x' = kx(M - x), Figures 1.6.2 and
1.6.5-1.6.7). Companion notebook: 03.

Sections (methods of ``Episode02``):
    intro          exponential growth cannot last
    the_model      per-capita growth rate decreasing linearly -> x' = k x (M - x)
    phase_line     f(x) on a graph, arrows on the axis, the phase line, stability by perturbation
    solutions      slope field (k = 0.1, M = 5), family of solutions, inflection at M/2
    exact          the closed form, and the finite-time blow-up for x0 < 0
    harvesting     x' = k x (M - x) - h: equilibria merge and vanish (saddle-node at h = kM^2/4)
    recap

Render (from this folder):
    manim -pql ep02_logistic.py Episode02
    python render.py ep02
"""

from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, UR, Arrow, Axes, Create, DashedLine, Dot, FadeIn, Line,
                   MathTex, Text, TransformMatchingTex, ValueTracker, VGroup, Write, always_redraw,
                   linear, there_and_back)

from odekit import (BODY_SIZE, HIGHLIGHT, MATH_SIZE, MUTED, PARAMETER, SOLUTION, STABLE, UNSTABLE,
                    OdeScene, slope_field, solution_curve)

K, M = 0.1, 5.0                      # Lebl, Figure 1.6.2
K_H, M_H = 0.1, 8.0                  # Lebl, harvesting example (Figures 1.6.5-1.6.7)


def logistic(t: float, x0: float, k: float = K, m: float = M) -> float:
    """Exact solution of x' = k x (m - x), x(0) = x0 (Lebl, Section 1.6); NaN after blow-up."""
    denominator = x0 + (m - x0) * np.exp(-k * m * t)
    return m * x0 / denominator if denominator > 0 else np.nan


class Episode02(OdeScene):
    def construct(self):
        self.intro()
        self.the_model()
        self.phase_line()
        self.solutions()
        self.exact()
        self.harvesting()
        self.recap()

    # =====================================================================================
    def intro(self):
        self.title_card("Episode 2", "The Logistic Equation", "Limits to growth")
        exponential = MathTex(r"P(t) = P_0\,e^{kt} \;\longrightarrow\; \infty", font_size=MATH_SIZE + 8)
        self.narrate("Last time, exponential growth: the population doubles again and again, "
                     "forever.", Write(exponential))
        self.narrate("But no Petri dish, no lake, no planet has unlimited food and space. Sooner or "
                     "later, growth must slow down. How do we put that into an equation?")
        self.clear_screen()

    # =====================================================================================
    def the_model(self):
        self.section_title("1. Slowing down")
        axes = Axes(x_range=[0, 6, 1], y_range=[-0.2, 0.6, 0.1], x_length=7.5, y_length=4.2,
                    axis_config={"color": MUTED, "include_tip": False}).to_edge(LEFT, buff=0.8)
        axes.shift(DOWN * 0.4)
        x_label = MathTex("x", font_size=36).next_to(axes.c2p(6, 0), DOWN)
        y_label = MathTex(r"\frac{1}{x}\frac{dx}{dt}", font_size=34).next_to(axes.c2p(0, 0.6), RIGHT)
        flat = Line(axes.c2p(0, 0.5), axes.c2p(6, 0.5), color=MUTED, stroke_width=4)
        flat_label = Text("exponential: constant", font_size=22, color=MUTED).next_to(flat, UP, buff=0.1)
        self.narrate("Look at the growth rate per individual: the derivative divided by x. For "
                     "exponential growth it's a constant, whatever the size of the population.",
                     Create(axes), FadeIn(x_label), FadeIn(y_label), Create(flat), FadeIn(flat_label))
        sloped = Line(axes.c2p(0, 0.5), axes.c2p(6, -0.1), color=SOLUTION, stroke_width=5)
        m_dot = Dot(axes.c2p(5, 0), color=STABLE)
        m_label = MathTex("M", color=STABLE, font_size=36).next_to(m_dot, DOWN)
        self.narrate("A more realistic choice: the per-capita rate decreases as the population "
                     "grows, and it reaches zero at a limiting population, M.",
                     Create(sloped), FadeIn(m_dot), Write(m_label))
        per_capita = MathTex(r"\frac{1}{x}\frac{dx}{dt}", "=", "k", "(M - x)", font_size=MATH_SIZE)
        per_capita.to_edge(RIGHT, buff=0.8).shift(UP * 1.2)
        logistic_eq = MathTex(r"\frac{dx}{dt}", "=", "k", "x", "(M - x)", font_size=MATH_SIZE + 6)
        logistic_eq.move_to(per_capita)
        self.narrate("The simplest such law is a straight line: one over x, d x d t, equals k times "
                     "M minus x.", Write(per_capita))
        self.narrate("Multiply by x and you get the logistic equation: x prime equals k x, times M "
                     "minus x. That's the notation of Lebl's notes, section 1.6.",
                     TransformMatchingTex(per_capita, logistic_eq))
        self.key_idea(logistic_eq)
        carrying = Text("M = carrying capacity, k > 0", font_size=BODY_SIZE, color=MUTED)
        carrying.next_to(logistic_eq, DOWN, buff=0.7).to_edge(RIGHT, buff=0.4)
        self.narrate("M is the carrying capacity, the largest population the environment can "
                     "sustain. This equation is nonlinear, because of the x squared term.",
                     FadeIn(carrying))
        self.clear_screen()

    # =====================================================================================
    def phase_line(self):
        self.section_title("2. Reading the equation without solving it")
        axes = Axes(x_range=[-1.5, 7, 1], y_range=[-1.2, 0.9, 0.5], x_length=9, y_length=4.2,
                    axis_config={"color": MUTED, "include_tip": False}).shift(DOWN * 0.3 + LEFT * 1.4)
        f = lambda x: K * x * (M - x)                        # noqa: E731
        graph = axes.plot(f, x_range=[-1.3, 6.6], color=SOLUTION)
        f_label = MathTex(r"f(x) = kx(M - x)", color=SOLUTION, font_size=36).next_to(axes.c2p(2.5, 0.75), UP)
        x_label = MathTex("x", font_size=36).next_to(axes.c2p(7, 0), DOWN)
        self.narrate("The equation is autonomous: x prime depends only on x. So let's simply plot "
                     "the right-hand side, f of x.", Create(axes), FadeIn(x_label), Create(graph),
                     Write(f_label))
        zero = Dot(axes.c2p(0, 0), color=UNSTABLE, radius=0.11)
        cap = Dot(axes.c2p(M, 0), color=STABLE, radius=0.11)
        self.narrate("Where f is zero, x prime is zero: nothing changes. Those are the critical "
                     "points, x equal zero and x equal M: two equilibrium solutions.",
                     FadeIn(zero, scale=2), FadeIn(cap, scale=2))
        arrows = VGroup(
            Arrow(axes.c2p(-1.3, 0), axes.c2p(-0.5, 0), buff=0, color=UNSTABLE).rotate(np.pi),
            Arrow(axes.c2p(1.2, 0), axes.c2p(3.8, 0), buff=0, color=STABLE),
            Arrow(axes.c2p(6.7, 0), axes.c2p(5.6, 0), buff=0, color=STABLE),
        )
        self.narrate("Between them, f is positive, so x increases: arrow to the right. Above M, "
                     "f is negative: x decreases. Below zero, also negative: x runs away downwards.",
                     *(Create(a) for a in arrows), run_time=3)
        notes = VGroup(MathTex(r"f'(0) = kM > 0", r"\;\Rightarrow\;\text{unstable}", font_size=34),
                       MathTex(r"f'(M) = -kM < 0", r"\;\Rightarrow\;\text{stable}", font_size=34))
        notes.arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_edge(RIGHT, buff=0.4).shift(UP * 1.5)
        notes[0][1].set_color(UNSTABLE)
        notes[1][1].set_color(STABLE)
        self.narrate("The arrows point towards M from both sides: M is stable. They point away from "
                     "zero: zero is unstable. The slope of f at each point tells the same story.",
                     Write(notes))

        # perturbation experiments
        probe = Dot(axes.c2p(M, 0), color=HIGHLIGHT, radius=0.09)
        self.add(probe)
        self.narrate("Nudge a population sitting at M, and it comes back.",
                     probe.animate(rate_func=there_and_back).move_to(axes.c2p(M + 0.9, 0)), run_time=2)
        probe.move_to(axes.c2p(0, 0))
        self.narrate("Nudge the empty population at zero with a few individuals, and it grows away "
                     "towards M.", probe.animate.move_to(axes.c2p(M - 0.05, 0)), run_time=3)
        self.clear_screen()

    # =====================================================================================
    def solutions(self):
        self.section_title("3. The solutions")
        axes = Axes(x_range=[0, 10, 2], y_range=[-1, 8, 1], x_length=10, y_length=5.4,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": range(0, 11, 2)},
                    y_axis_config={"numbers_to_include": [5]}).shift(DOWN * 0.4)
        field = slope_field(axes, lambda t, x: K * x * (M - x), t_step=0.625, y_step=0.5)
        equation = MathTex(r"x' = 0.1\,x\,(5 - x)", font_size=38).to_corner(UR, buff=0.5)
        self.narrate("Here's the slope field for k equal to 0.1 and M equal to 5, exactly Lebl's "
                     "figure 1.6.2.", Create(axes), FadeIn(field), Write(equation))
        equilibria = VGroup(DashedLine(axes.c2p(0, M), axes.c2p(10, M), color=STABLE),
                            DashedLine(axes.c2p(0, 0), axes.c2p(10, 0), color=UNSTABLE))
        self.play(Create(equilibria))
        x0 = ValueTracker(0.3)
        curve = always_redraw(lambda: solution_curve(axes, lambda t: logistic(t, x0.get_value()), 0, 10))
        start = always_redraw(lambda: Dot(axes.c2p(0, x0.get_value()), color=HIGHLIGHT))
        self.add(curve, start)
        self.narrate("Start with a small population: slow growth, then faster, then it levels off "
                     "at M. The famous S-shaped curve.")
        self.narrate("Move the starting value up, and every solution still approaches M; start "
                     "above M, and the population shrinks towards it.",
                     x0.animate.set_value(7.6), run_time=6, )
        self.play(x0.animate.set_value(0.6), run_time=3, rate_func=linear)
        inflection_t = np.log((M - 0.6) / 0.6) / (K * M)
        inflection = Dot(axes.c2p(inflection_t, M / 2), color=PARAMETER, radius=0.1)
        half = DashedLine(axes.c2p(0, M / 2), axes.c2p(10, M / 2), color=PARAMETER, stroke_width=2)
        note = MathTex(r"x = M/2:\ \text{fastest growth}", color=PARAMETER, font_size=34)
        note.next_to(axes.c2p(8.0, M / 2), DOWN, buff=0.45)       # below the M/2 line, right side
        self.narrate("Notice the inflection point: growth is fastest exactly when the population is "
                     "half the carrying capacity.", Create(half), FadeIn(inflection, scale=2), Write(note))
        self.clear_screen()

    # =====================================================================================
    def exact(self):
        self.section_title("4. The exact solution")
        formula = MathTex(r"x(t) = \frac{M\,x_0}{x_0 + (M - x_0)\,e^{-kMt}}", font_size=MATH_SIZE + 6)
        formula.shift(UP * 1.6)
        how = Text("separation of variables + partial fractions (notebook 03)", font_size=26, color=MUTED)
        how.next_to(formula, DOWN, buff=0.4)
        self.narrate("The equation is separable, and with partial fractions we get this formula. "
                     "Check: at time zero it gives x zero, and as t grows it tends to M.",
                     Write(formula), FadeIn(how))
        self.key_idea(formula, color=SOLUTION)
        axes = Axes(x_range=[0, 6, 1], y_range=[-6, 1, 1], x_length=7, y_length=3.2,
                    axis_config={"color": MUTED, "include_tip": False}).to_edge(DOWN, buff=0.4)
        x0 = -0.5
        t_star = np.log((M - x0) / -x0) / (K * M)
        curve = solution_curve(axes, lambda t: logistic(t, x0), 0, t_star * 0.999, color=UNSTABLE)
        asymptote = DashedLine(axes.c2p(t_star, 1), axes.c2p(t_star, -6), color=HIGHLIGHT)
        label = MathTex(r"t^* = \frac{1}{kM}\ln\frac{M - x_0}{-x_0}", color=HIGHLIGHT, font_size=34)
        label.next_to(asymptote, RIGHT, buff=0.3)
        self.narrate("A surprise hides in the formula. If x zero is negative, the denominator "
                     "reaches zero at a finite time, t star, and the solution dives to minus infinity.",
                     Create(axes), Create(curve), run_time=3)
        self.narrate("The equation is a perfectly smooth polynomial, yet the solution exists only "
                     "for a finite time. Picard's theorem only promises a solution near the start.",
                     Create(asymptote), Write(label))
        self.clear_screen()

    # =====================================================================================
    def harvesting(self):
        self.section_title("5. Harvesting: a bifurcation")
        equation = MathTex(r"x' = kx(M - x)", "-", "h", font_size=MATH_SIZE).to_corner(UR, buff=0.5)
        equation[2].set_color(PARAMETER)
        self.narrate("Now remove individuals at a constant rate h: fishing, hunting, or, in Lebl's "
                     "notes, aliens who harvest humans.", Write(equation))
        axes = Axes(x_range=[-1, 9, 1], y_range=[-3, 1.8, 0.5], x_length=9.5, y_length=4.8,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": range(0, 9, 2)}).shift(DOWN * 0.5 + LEFT * 0.8)
        h = ValueTracker(0.0)
        graph = always_redraw(lambda: axes.plot(
            lambda x: K_H * x * (M_H - x) - h.get_value(), x_range=[-0.8, 8.8], color=SOLUTION))

        def equilibria() -> VGroup:
            """Dots at the zeros of f(x) - h: B unstable (pink), A stable (green).

            At h = kM^2/4 the two zeros coincide: one semi-stable point (coral).
            """
            disc = (K_H * M_H) ** 2 - 4 * K_H * h.get_value()
            dots = VGroup()
            if abs(disc) < 1e-3:
                dots.add(Dot(axes.c2p(M_H / 2, 0), color=HIGHLIGHT, radius=0.13))
            elif disc > 0:
                root = np.sqrt(max(disc, 0.0))
                b, a = (K_H * M_H - root) / (2 * K_H), (K_H * M_H + root) / (2 * K_H)
                dots.add(Dot(axes.c2p(b, 0), color=UNSTABLE, radius=0.11),
                         Dot(axes.c2p(a, 0), color=STABLE, radius=0.11))
            return dots

        dots = always_redraw(equilibria)
        readout = always_redraw(lambda: MathTex(f"h = {h.get_value():.2f}", color=PARAMETER, font_size=40)
                                .next_to(equation, DOWN, buff=0.5))
        self.narrate("Harvesting simply shifts the graph of f down by h. Its zeros are the equilibria: "
                     "with no harvesting, zero and M equal 8.", Create(axes), Create(graph), FadeIn(dots),
                     FadeIn(readout))
        self.narrate("Harvest one million per year: two equilibria, B, about 1.55, unstable, and A, "
                     "about 6.45, stable. Start above B and the population survives.",
                     h.animate.set_value(1.0), run_time=4)
        self.narrate("Increase h to 1.6, which is k M squared over 4. The two equilibria collide "
                     "at x equal 4.", h.animate.set_value(1.6), run_time=4)
        critical = MathTex(r"h_c = \frac{kM^2}{4}", color=HIGHLIGHT, font_size=40)
        critical.next_to(readout, DOWN, buff=0.4)
        self.narrate("This is a saddle-node bifurcation. The merged point attracts from above but "
                     "repels from below: the smallest disturbance can wipe the population out.",
                     Write(critical))
        self.narrate("Harvest any more, and there are no equilibria at all. Whatever the starting "
                     "population, it collapses. A sobering lesson for fisheries.",
                     h.animate.set_value(2.0), run_time=3)
        self.clear_screen()

    # =====================================================================================
    def recap(self):
        self.section_title("Recap")
        points = VGroup(
            MathTex(r"x' = kx(M - x):\ \text{per-capita rate } k(M - x)"),
            MathTex(r"\text{equilibria } 0 \text{ (unstable)},\ M \text{ (stable)}"),
            MathTex(r"\text{phase line: long-term behaviour without solving}"),
            MathTex(r"x(t) = \frac{M x_0}{x_0 + (M - x_0)e^{-kMt}},\ \text{fastest at } M/2"),
            MathTex(r"\text{harvesting: saddle-node at } h = kM^2/4", color=HIGHLIGHT),
        )
        points.arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.85)
        lines = ["To recap: the logistic equation lets the per-capita growth rate decrease linearly.",
                 "It has two equilibria: zero, unstable, and the carrying capacity, stable.",
                 "The phase line gives the long-term behaviour without solving anything.",
                 "The exact solution is the S-curve, steepest at half the carrying capacity.",
                 "And harvesting can push the system over a tipping point. Next time: how to solve "
                 "equations numerically, with Euler's method. See you!"]
        for point, line in zip(points, lines, strict=True):
            self.narrate(line, FadeIn(point, shift=RIGHT * 0.3))
        self.clear_screen()


class Ep02Intro(Episode02):
    def construct(self):
        self.intro()


class Ep02Model(Episode02):
    def construct(self):
        self.the_model()


class Ep02PhaseLine(Episode02):
    def construct(self):
        self.phase_line()


class Ep02Solutions(Episode02):
    def construct(self):
        self.solutions()


class Ep02Exact(Episode02):
    def construct(self):
        self.exact()


class Ep02Harvesting(Episode02):
    def construct(self):
        self.harvesting()


class Ep02Recap(Episode02):
    def construct(self):
        self.recap()
