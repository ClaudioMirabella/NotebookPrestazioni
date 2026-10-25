"""
Episode 3 - Euler's method: solving differential equations with a computer.

MATH 331 - Differential Equations (Euler preview in Session 8, unit U14 in Session 29).
Course text: J. Lebl, *Notes on Diffy Qs*, Section 1.7 (Figures 1.7.1-1.7.2, Tables 1.7.1-1.7.2,
Exercise 1.7.4 on instability). Absolute stability as in Meseguer (ed.), *Fundamentals of
Numerical Mathematics for Physicists and Engineers*, Section 8.2. Companion notebook: 04.

Sections (methods of ``Episode03``):
    intro        most equations cannot be solved by hand
    the_idea     follow the tangent: Euler's steps on y' = y^2/3, y(0) = 1 (Lebl, Figure 1.7.1)
    refining     smaller steps, Lebl's Table 1.7.1: the error halves when h halves
    why_order    local error ~ h^2, number of steps ~ 1/h, global error ~ h
    heun         improved Euler: average two slopes (second order); RK4 (fourth order)
    instability  y' = -5y: the factor 1 - 5h, oscillations and blow-up of the numbers
    recap        including the blow-up warning of Table 1.7.2

Render (from this folder):
    manim -pql ep03_euler.py Episode03
    python render.py ep03
"""

from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, UR, Axes, Create, Dot, FadeIn, FadeOut, Line,
                   MathTex, Table, Text, VGroup, Write)

from odekit import (HIGHLIGHT, MATH_SIZE, MUTED, NUMERIC, SOLUTION, STABLE, UNSTABLE,
                    OdeScene, slope_field, solution_curve)


def f_lebl(x: float, y: float) -> float:
    """Right-hand side of Lebl's example y' = y^2/3."""
    return y**2 / 3


def exact_lebl(x: float) -> float:
    """Exact solution of y' = y^2/3, y(0) = 1: y = 3/(3 - x) (blows up at x = 3)."""
    return 3 / (3 - x)


def euler_points(f, x0: float, y0: float, h: float, n: int) -> list[tuple[float, float]]:
    """The n + 1 points (x_i, y_i) of Euler's method: y_{i+1} = y_i + h f(x_i, y_i)."""
    points = [(x0, y0)]
    for _ in range(n):
        x, y = points[-1]
        points.append((x + h, y + h * f(x, y)))
    return points


class Episode03(OdeScene):
    def construct(self):
        self.intro()
        self.the_idea()
        self.refining()
        self.why_order()
        self.heun()
        self.instability()
        self.recap()

    # =====================================================================================
    def intro(self):
        self.title_card("Episode 3", "Euler's Method", "Solving differential equations with a computer")
        hard = MathTex(r"y' = \sin(x\,y) + e^{-y^2}", font_size=MATH_SIZE + 8)
        self.narrate("Growth, decay, the logistic equation: so far we could always find a formula. "
                     "Most differential equations, like this one, have no formula at all.", Write(hard))
        self.narrate("But the equation still tells us the slope at every point. And that is enough "
                     "to approximate the solution. Let's see how.")
        self.clear_screen()

    # =====================================================================================
    def make_axes(self) -> Axes:
        return Axes(x_range=[0, 2.2, 0.5], y_range=[0, 3.4, 0.5], x_length=8.5, y_length=5.4,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": [0.5, 1, 1.5, 2]},
                    y_axis_config={"numbers_to_include": [1, 2, 3]}).shift(LEFT * 1.6 + DOWN * 0.4)

    def the_idea(self):
        self.section_title("1. Follow the tangent")
        axes = self.make_axes()
        field = slope_field(axes, f_lebl, t_step=0.1375, y_step=0.2125)
        equation = MathTex(r"y' = \frac{y^2}{3},\quad y(0) = 1", font_size=40).to_corner(UR, buff=0.5)
        self.narrate("Lebl's example: y prime equals y squared over 3, starting from y equal 1 at x "
                     "equal 0. Here's its slope field.", Create(axes), FadeIn(field), Write(equation))
        start = Dot(axes.c2p(0, 1), color=HIGHLIGHT)
        self.play(FadeIn(start, scale=2))
        rule = MathTex(r"y_{i+1} = y_i + h\,f(x_i, y_i)", font_size=40, color=NUMERIC)
        rule.next_to(equation, DOWN, buff=0.6)
        points = euler_points(f_lebl, 0, 1, 1.0, 2)
        lines = ["At the starting point, the equation gives the slope: one third. Follow the "
                 "tangent line for a step of length h, here h equal 1.",
                 "At the new point, compute the new slope, and follow it for another step. "
                 "Repeat. That's all: Euler's method."]
        for i, line in enumerate(lines):
            (x0, y0), (x1, y1) = points[i], points[i + 1]
            segment = Line(axes.c2p(x0, y0), axes.c2p(x1, y1), color=NUMERIC, stroke_width=5)
            dot = Dot(axes.c2p(x1, y1), color=NUMERIC)
            extra = [Write(rule)] if i == 1 else []
            self.narrate(line, Create(segment), FadeIn(dot), *extra, run_time=2.5)
        values = MathTex(r"y_1 = \tfrac{4}{3}\approx 1.333,\quad y_2 = \tfrac{52}{27}\approx 1.926",
                         font_size=30).next_to(rule, DOWN, buff=0.5).to_edge(RIGHT, buff=0.4)
        exact = solution_curve(axes, exact_lebl, 0, 2.15)
        exact_label = MathTex(r"y = \frac{3}{3 - x}", color=SOLUTION, font_size=36)
        exact_label.next_to(values, DOWN, buff=0.5)
        self.narrate("Two steps give y at 2 approximately 1.926. The exact solution, 3 over 3 minus x, "
                     "gives 3. We are quite far off.", Write(values), Create(exact), Write(exact_label))
        self.clear_screen()

    # =====================================================================================
    def refining(self):
        self.section_title("2. Smaller steps")
        axes = self.make_axes()
        exact = solution_curve(axes, exact_lebl, 0, 2.15)
        self.play(Create(axes), Create(exact))
        polygon = None
        for h in (1.0, 0.5, 0.25, 0.125):
            n = int(round(2 / h))
            pts = euler_points(f_lebl, 0, 1, h, n)
            new_polygon = VGroup(*(Line(axes.c2p(*a), axes.c2p(*b), color=NUMERIC, stroke_width=4)
                                   for a, b in zip(pts[:-1], pts[1:], strict=True)),
                                 *(Dot(axes.c2p(*p), color=NUMERIC, radius=0.05) for p in pts))
            caption = f"h = {h:g}: y(2) is about {pts[-1][1]:.3f}"
            if polygon is None:
                self.narrate("The cure is obvious: take smaller steps. " + caption + ".",
                             Create(new_polygon), run_time=2)
            else:
                self.narrate(caption + ".", FadeOut(polygon), Create(new_polygon), run_time=2)
            polygon = new_polygon
        self.clear_screen()

        self.section_title("2. Smaller steps: the error")
        rows = [["1", "1.92593", "1.07407", ""], ["0.5", "2.20861", "0.79139", "0.74"],
                ["0.25", "2.47249", "0.52751", "0.67"], ["0.125", "2.68034", "0.31966", "0.61"],
                ["0.0625", "2.82040", "0.17960", "0.56"], ["0.03125", "2.90412", "0.09588", "0.53"]]
        table = Table(rows, col_labels=[MathTex("h"), Text("y(2) approx."), Text("error"),
                                        Text("error / previous")],
                      include_outer_lines=False, line_config={"stroke_color": MUTED, "stroke_width": 1},
                      element_to_mobject_config={"font_size": 30}).scale(0.62)
        table.shift(DOWN * 0.3)
        self.narrate("Here's Lebl's table 1.7.1. Each time we halve h, the error is multiplied by a "
                     "number that approaches one half.", Create(table), run_time=4)
        verdict = MathTex(r"\text{error} \approx C\,h \quad\Rightarrow\quad \text{first-order method}",
                          color=NUMERIC, font_size=40).to_edge(DOWN, buff=0.4)
        self.narrate("The error is proportional to h: Euler's method is a first-order method. To "
                     "gain one digit of accuracy, you need ten times as many steps.", Write(verdict))
        self.clear_screen()

    # =====================================================================================
    def why_order(self):
        self.section_title("3. Why first order?")
        taylor = MathTex(r"y(x + h) = ", r"y(x) + h\,y'(x)", r"+ \tfrac{h^2}{2}\,y''(\xi)",
                         font_size=MATH_SIZE).shift(UP * 1.6)
        taylor[1].set_color(NUMERIC)
        taylor[2].set_color(HIGHLIGHT)
        self.narrate("Taylor's theorem explains it. The exact solution after one step is y plus h y "
                     "prime, which is Euler's step, plus a remainder of order h squared.", Write(taylor))
        local = MathTex(r"\text{one step: error} \sim h^2", color=HIGHLIGHT, font_size=40)
        steps = MathTex(r"\text{number of steps} = \frac{X - x_0}{h}", font_size=40)
        total = MathTex(r"\text{total: } h^2 \times \frac{1}{h} = h", color=NUMERIC, font_size=44)
        VGroup(local, steps, total).arrange(DOWN, buff=0.5).next_to(taylor, DOWN, buff=0.8)
        self.narrate("So each step makes an error of size h squared, the local error.", Write(local))
        self.narrate("To reach a fixed point we need about one over h steps...", Write(steps))
        self.narrate("...and the errors add up to something proportional to h. A more careful "
                     "argument, in notebook 04, turns this into a rigorous bound.", Write(total))
        self.key_idea(total, color=NUMERIC)
        self.clear_screen()

    # =====================================================================================
    def heun(self):
        self.section_title("4. Doing better: average the slopes")
        axes = Axes(x_range=[0, 1.2, 0.5], y_range=[0.8, 2.2, 0.5], x_length=7.5, y_length=5,
                    axis_config={"color": MUTED, "include_tip": False}).shift(LEFT * 2.4 + DOWN * 0.4)
        exact = solution_curve(axes, exact_lebl, 0, 1.15)
        h, y0 = 1.0, 1.0
        k1 = f_lebl(0, y0)
        guess = y0 + h * k1
        k2 = f_lebl(h, guess)
        heun_y = y0 + h * (k1 + k2) / 2
        p0, p_guess, p_heun = axes.c2p(0, y0), axes.c2p(h, guess), axes.c2p(h, heun_y)
        step1 = Line(p0, p_guess, color=NUMERIC, stroke_width=4)
        slope2 = Line(p0, axes.c2p(h, y0 + h * k2), color=UNSTABLE, stroke_width=4)
        average = Line(p0, p_heun, color=STABLE, stroke_width=5)
        self.narrate("Euler uses only the slope at the start of the step, so it always lags behind a "
                     "curving solution.", Create(axes), Create(exact), Create(step1),
                     FadeIn(Dot(p_guess, color=NUMERIC)))
        self.narrate("Improved Euler, or Heun's method, also computes the slope at the end of the "
                     "Euler step, and then uses the average of the two slopes.",
                     Create(slope2), Create(average), FadeIn(Dot(p_heun, color=STABLE)), run_time=3)
        formulas = VGroup(
            MathTex(r"k_1 = f(x_i, y_i)", font_size=36),
            MathTex(r"k_2 = f(x_i + h,\ y_i + h k_1)", font_size=36),
            MathTex(r"y_{i+1} = y_i + h\,\frac{k_1 + k_2}{2}", font_size=36, color=STABLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_corner(UR, buff=0.5).shift(DOWN * 0.6)
        self.narrate("In formulas: k one, k two, and the average.", Write(formulas))
        orders = VGroup(
            MathTex(r"\text{Euler: error} \propto h", color=NUMERIC, font_size=34),
            MathTex(r"\text{Heun: error} \propto h^2", color=STABLE, font_size=34),
            MathTex(r"\text{RK4: error} \propto h^4", color=SOLUTION, font_size=34),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        orders.next_to(formulas, DOWN, buff=0.6).align_to(formulas, LEFT)
        self.narrate("One extra evaluation of f, and the method becomes second order: halve h, and "
                     "the error drops by four. The classical Runge-Kutta method uses four slopes "
                     "and is fourth order: halve h, error divided by sixteen.", Write(orders), run_time=3)
        self.clear_screen()

    # =====================================================================================
    def instability(self):
        self.section_title("5. A warning: numerical instability")
        equation = MathTex(r"y' = -5y,\quad y(0) = 1", font_size=40).to_corner(UR, buff=0.5)
        factor = MathTex(r"y_{i+1} = (1 - 5h)\,y_i", font_size=40, color=NUMERIC)
        factor.next_to(equation, DOWN, buff=0.5)
        axes = Axes(x_range=[0, 4, 1], y_range=[-2, 2, 1], x_length=8.5, y_length=5,
                    axis_config={"color": MUTED, "include_tip": False},
                    x_axis_config={"numbers_to_include": [1, 2, 3, 4]},
                    y_axis_config={"numbers_to_include": [-1, 1]}).shift(LEFT * 1.8 + DOWN * 0.4)
        exact = solution_curve(axes, lambda x: np.exp(-5 * x), 0, 4)
        self.narrate("One more lesson, from Lebl's exercise 1.7.4. Take y prime equals minus 5 y: "
                     "the solution decays quietly to zero.", Write(equation), Create(axes), Create(exact))
        self.narrate("Euler's step multiplies y by one minus 5 h, at every step.", Write(factor))
        polygon, h_label = None, None
        cases = [(0.1, "h = 0.1: factor 0.5, a smooth decay. Fine.", STABLE),
                 (0.3, "h = 0.3: factor minus 0.5. The numbers oscillate around zero: an artefact "
                       "of the method.", NUMERIC),
                 (0.45, "h = 0.45: factor minus 1.25. The numbers grow and explode, while the true "
                        "solution decays!", UNSTABLE)]
        for h, line, color in cases:
            n = int(round(4 / h)) if h != 0.45 else 8
            pts = [(i * h, (1 - 5 * h) ** i) for i in range(n + 1)]
            pts = [(x, y) for x, y in pts if abs(y) <= 2 and x <= 4]
            new_polygon = VGroup(*(Line(axes.c2p(*a), axes.c2p(*b), color=color, stroke_width=4)
                                   for a, b in zip(pts[:-1], pts[1:], strict=True)),
                                 *(Dot(axes.c2p(*p), color=color, radius=0.06) for p in pts))
            new_label = MathTex(f"h = {h:g}", color=color, font_size=40).next_to(axes.c2p(3, 1.6), DOWN)
            anims = [Create(new_polygon), FadeIn(new_label)]
            if polygon is not None:
                anims += [FadeOut(polygon), FadeOut(h_label)]
            self.narrate(line, *anims, run_time=2.5)
            polygon, h_label = new_polygon, new_label
        rule = MathTex(r"|1 - 5h| < 1 \iff h < 0.4", font_size=40, color=HIGHLIGHT)
        rule.next_to(factor, DOWN, buff=0.6).to_edge(RIGHT, buff=0.4)
        self.narrate("The numbers decay only if one minus 5 h is smaller than one in absolute value, "
                     "that is, h smaller than 0.4. This is called absolute stability.", Write(rule))
        self.narrate("Equations that force tiny steps for stability, not for accuracy, are called "
                     "stiff. Professional solvers have special methods for them.")
        self.clear_screen()

    # =====================================================================================
    def recap(self):
        self.section_title("Recap")
        points = VGroup(
            MathTex(r"\text{Euler: } y_{i+1} = y_i + h\,f(x_i, y_i)"),
            MathTex(r"\text{global error} \propto h \ \text{(first order)}"),
            MathTex(r"\text{Heun} \propto h^2,\quad \text{RK4} \propto h^4"),
            MathTex(r"\text{stability: } y' = \lambda y \text{ needs } |1 + h\lambda| < 1"),
            MathTex(r"\text{if the exact solution blows up, the numbers never settle}", color=HIGHLIGHT),
        )
        points.arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.85)
        lines = ["Let's recap. Euler's method follows the tangent line, one step at a time.",
                 "Its error is proportional to the step size.",
                 "Averaging slopes, as Heun and Runge-Kutta do, gives much faster convergence.",
                 "Steps that are too large can make a decaying solution explode in the computer.",
                 "And one last warning, from Lebl's table 1.7.2: if the exact solution blows up, the "
                 "numbers keep changing as you refine h. Numbers that settle are evidence; numbers "
                 "that keep changing are a red flag. Thanks for watching!"]
        for point, line in zip(points, lines, strict=True):
            self.narrate(line, FadeIn(point, shift=RIGHT * 0.3))
        self.clear_screen()


class Ep03Intro(Episode03):
    def construct(self):
        self.intro()


class Ep03Idea(Episode03):
    def construct(self):
        self.the_idea()


class Ep03Refining(Episode03):
    def construct(self):
        self.refining()


class Ep03WhyOrder(Episode03):
    def construct(self):
        self.why_order()


class Ep03Heun(Episode03):
    def construct(self):
        self.heun()


class Ep03Instability(Episode03):
    def construct(self):
        self.instability()


class Ep03Recap(Episode03):
    def construct(self):
        self.recap()
