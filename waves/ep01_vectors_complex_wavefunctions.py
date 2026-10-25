"""
Episode 1 - Vectors, complex numbers and wave functions.

Following Chapter 1 ("Wave fundamentals") of D. Fleisch & L. Kinnaman,
*A Student's Guide to Waves* (Cambridge University Press, 2015).

Sections (each is a method of ``Episode01`` and can also be rendered alone):

    0. intro              title card and roadmap
    1. what_is_a_wave     disturbance vs medium, transverse/longitudinal,
                          wavelength, period, frequency, v = λf, k, ω
    2. vectors            components, unit vectors, magnitude, direction,
                          addition, dot product
    3. complex_numbers    complex plane, magnitude and phase, multiplying by i
                          rotates, products, conjugate
    4. euler_relation     e^{iθ} = cos θ + i sin θ on the unit circle,
                          cos θ from two counter-rotating arrows
    5. wave_functions     f(x - vt), f(x + vt), harmonic waves, phase,
                          complex form (teaser of episode 2: phasors)
    6. recap

Render (from inside this folder):

    manim -pql ep01_vectors_complex_wavefunctions.py Episode01   # whole episode, preview
    manim -qh  ep01_vectors_complex_wavefunctions.py Episode01   # 1080p60 final
    manim -pql ep01_vectors_complex_wavefunctions.py Ep01Euler   # one section only

Next to the video Manim writes ``Episode01.srt``: the commentary, with timings.

Reading guide for the code
--------------------------
* Positions are in screen units (frame: 14.2 x 8, origin at the centre).
* Every quantity that changes continuously (time, angles) is a ``ValueTracker``;
  the drawings depending on it are built by small functions passed to
  ``always_redraw`` (rebuilt at every frame). See ``primer.py``, scene P4.
* ``self.narrate(text, *animations)`` (from ``wavekit``) writes a line of the
  commentary to the subtitles and keeps the video going while it is "spoken".
"""

from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, PI, RIGHT, TAU, UP, UR, Arc, Arrow, Axes,
                   Brace, BraceBetweenPoints, Circle, ComplexPlane, Create, DashedLine, Dot,
                   DoubleArrow, FadeIn, FadeOut, GrowArrow, Line, MathTex,
                   NumberPlane, Rotate, Text, Transform, ValueTracker, VGroup,
                   VMobject, Write, always_redraw, linear)

from wavekit import (BODY_SIZE, HIGHLIGHT, IMAG, MATH_SIZE, MUTED, REAL, TEXT_COLOR,
                     VECTOR_A, VECTOR_B, WAVE, WaveScene)

# ---------------------------------------------------------------------------
# Parameters of the example wave used in section 1 (screen units and seconds)
# ---------------------------------------------------------------------------
WAVELENGTH = 4.0                       # λ: distance between crests [units]
PERIOD = 2.0                           # T: time of one oscillation [s]
AMPLITUDE = 0.7                        # A: largest displacement [units]
WAVENUMBER = 2 * PI / WAVELENGTH       # k = 2π/λ
ANGULAR_FREQ = 2 * PI / PERIOD         # ω = 2π/T


def harmonic(x: float | np.ndarray, t: float, amplitude: float = AMPLITUDE,
             k: float = WAVENUMBER, omega: float = ANGULAR_FREQ, phi: float = 0.0):
    """Harmonic travelling wave y(x, t) = A sin(kx - ωt + φ)."""
    return amplitude * np.sin(k * x - omega * t + phi)


def pulse(x: float | np.ndarray, width: float = 0.8):
    """A smooth bump of height 1 centred on x = 0 (a Gaussian)."""
    return np.exp(-(x / width) ** 2)


def smooth_curve(points: list[np.ndarray], color) -> VMobject:
    """A smooth line through a list of points (used to join the particles)."""
    curve = VMobject(color=color, stroke_width=2, stroke_opacity=0.6)
    curve.set_points_smoothly(points)
    return curve


class Episode01(WaveScene):
    """The whole episode: the sections are played one after the other."""

    def construct(self):
        self.intro()
        self.what_is_a_wave()
        self.vectors()
        self.complex_numbers()
        self.euler_relation()
        self.wave_functions()
        self.recap()

    # =========================================================================
    # 0. Intro
    # =========================================================================
    def intro(self):
        self.title_card("Episode 1", "Vectors, Complex Numbers and Wave Functions",
                        "The Mathematics of Waves")
        self.narrate("Hi, and welcome to the first episode of our journey into the "
                     "mathematics of waves.")
        self.narrate("Waves are everywhere: sound, light, ocean swells, even the electrons "
                     "in your phone behave like waves.")

        # A roadmap: four items, written one after the other.
        items = ["1.  What is a wave?", "2.  Vectors", "3.  Complex numbers and Euler",
                 "4.  Wave functions"]
        roadmap = VGroup(*(Text(item, font_size=BODY_SIZE, color=TEXT_COLOR) for item in items))
        roadmap.arrange(DOWN, aligned_edge=LEFT, buff=0.35)   # left-aligned list
        self.narrate("Before we can write down the wave equation, we need a small toolbox. "
                     "Here's today's plan.",
                     *(FadeIn(item, shift=RIGHT * 0.3) for item in roadmap), run_time=2.5)
        self.narrate("Let's go!")
        self.clear_screen()

    # =========================================================================
    # 1. What is a wave?
    # =========================================================================
    def what_is_a_wave(self):
        heading = self.section_title("1. What is a wave?")

        # --- a row of particles carrying a transverse wave ----------------------
        t = ValueTracker(0.0)                          # time [s], animated below
        x_positions = np.linspace(-6, 6, 31)           # rest positions of 31 particles
        tracked = 10                                   # index of the particle we follow

        def particle_updater(x0: float):
            """Return an updater moving a particle, at rest in x0, to y(x0, t).

            Why a function returning a function? Each particle needs its own x0.
            Writing `lambda d: d.move_to(... x0 ...)` directly inside the loop
            would capture the loop VARIABLE, and every particle would end up
            using the last value of x0 (a classic Python pitfall).
            """
            return lambda dot: dot.move_to([x0, harmonic(x0, t.get_value()), 0])

        particles = VGroup()
        for i, x0 in enumerate(x_positions):
            color = HIGHLIGHT if i == tracked else WAVE
            dot = Dot([x0, 0, 0], radius=0.09 if i == tracked else 0.06, color=color)
            dot.add_updater(particle_updater(x0))
            particles.add(dot)
        # a faint line through the particles helps to see the shape of the wave
        string = always_redraw(lambda: smooth_curve([d.get_center() for d in particles], WAVE))

        self.narrate("So, what is a wave? A good definition: a wave is a disturbance that "
                     "travels, carrying energy from one place to another.",
                     FadeIn(particles), FadeIn(string))
        self.add_subcaption("Look at this row of particles.", duration=2)
        self.play(t.animate.set_value(4 * PERIOD), run_time=8, rate_func=linear)
        self.narrate("The pattern moves to the right... but keep your eye on the orange "
                     "particle. It just bobs up and down!",
                     t.animate.set_value(7 * PERIOD), run_time=6, )
        self.narrate("The medium stays where it is. Only the disturbance, and its energy, "
                     "travels along.",
                     t.animate.set_value(9 * PERIOD), run_time=4)
        # Freeze everything that moves before fading it out (see WaveScene.clear_screen).
        particles.clear_updaters()                     # also clears each dot's updater
        string.clear_updaters()
        self.narrate("This is a transverse wave: the particles move perpendicular to the "
                     "direction in which the wave travels.",
                     FadeOut(particles), FadeOut(string))

        # --- a longitudinal wave: displacements ALONG the direction of travel ----
        # Thin vertical bars at rest positions x0, displaced horizontally by
        # 0.35 sin(kx0 - ωt): where the bars crowd together we have compressions.
        bar_positions = np.linspace(-6, 6, 61)

        def bars():
            displaced = bar_positions + 0.35 * np.sin(WAVENUMBER * bar_positions
                                                      - ANGULAR_FREQ * t.get_value())
            return VGroup(*(Line([x, -0.8, 0], [x, 0.8, 0], color=WAVE, stroke_width=3)
                            for x in displaced))

        longitudinal = always_redraw(bars)
        label = Text("longitudinal wave (like sound)", font_size=BODY_SIZE, color=MUTED)
        label.next_to(longitudinal, DOWN, buff=0.6)
        self.add(longitudinal)
        self.narrate("In a longitudinal wave, like sound in air, the particles jiggle back "
                     "and forth along the direction of travel, making compressions and "
                     "rarefactions.", FadeIn(label), t.animate.set_value(13 * PERIOD),
                     run_time=8)
        longitudinal.clear_updaters()                  # freeze the bars, then fade them out
        self.play(FadeOut(longitudinal), FadeOut(label))

        # --- snapshot (fixed t) and history (fixed x) ----------------------------
        snapshot_axes = Axes(x_range=[0, 3 * WAVELENGTH, WAVELENGTH], y_range=[-1, 1, 0.5],
                             x_length=6, y_length=2.4,
                             axis_config={"color": MUTED, "include_tip": False})
        history_axes = snapshot_axes.copy()
        VGroup(snapshot_axes, history_axes).arrange(RIGHT, buff=1.0).shift(DOWN * 0.3)
        snap_title = Text("snapshot: y versus x", font_size=26, color=MUTED)
        snap_title.next_to(snapshot_axes, UP, buff=1.0)   # leave room for the λ brace
        hist_title = Text("history: y versus t", font_size=26, color=MUTED)
        hist_title.next_to(history_axes, UP, buff=1.0)    # leave room for the T brace

        # y(x, 0) on the left, drawn as a cosine so that crests sit at x = 0, λ, 2λ
        # (convenient for the braces). The amplitude on screen is 0.8 units.
        snapshot = snapshot_axes.plot(lambda x: 0.8 * np.cos(WAVENUMBER * x), color=WAVE)
        crest_1 = snapshot_axes.c2p(WAVELENGTH, 0.8)
        crest_2 = snapshot_axes.c2p(2 * WAVELENGTH, 0.8)
        wavelength_brace = BraceBetweenPoints(crest_1, crest_2, direction=UP, color=HIGHLIGHT)
        wavelength_label = wavelength_brace.get_tex(r"\lambda").set_color(HIGHLIGHT)
        amplitude_arrow = DoubleArrow(snapshot_axes.c2p(2 * WAVELENGTH, 0), crest_2, buff=0,
                                      color=REAL, tip_length=0.15)
        amplitude_label = MathTex("A", color=REAL).next_to(amplitude_arrow, RIGHT, buff=0.1)

        self.narrate("Two pictures help a lot. First, freeze time and take a snapshot of the "
                     "wave: displacement y versus position x.",
                     Create(snapshot_axes), FadeIn(snap_title), Create(snapshot))
        self.narrate("The distance from one crest to the next is the wavelength, lambda. "
                     "The largest displacement is the amplitude, A.",
                     FadeIn(wavelength_brace), Write(wavelength_label),
                     GrowArrow(amplitude_arrow), Write(amplitude_label))

        # y(x0, t) on the right: same shape, but the horizontal axis is now time.
        history = history_axes.plot(lambda s: np.cos(2 * PI * s / WAVELENGTH) * 0.8, color=IMAG)
        period_brace = BraceBetweenPoints(history_axes.c2p(WAVELENGTH, 0.8),
                                          history_axes.c2p(2 * WAVELENGTH, 0.8),
                                          direction=UP, color=HIGHLIGHT)
        period_label = period_brace.get_tex("T").set_color(HIGHLIGHT)
        self.narrate("Now do the opposite: sit at one spot and record the displacement as "
                     "time goes by.",
                     Create(history_axes), FadeIn(hist_title), Create(history))
        self.narrate("The time for one full cycle is the period, T. Its inverse, one over T, "
                     "is the frequency f: cycles per second, or hertz.",
                     FadeIn(period_brace), Write(period_label))
        self.clear_screen()
        self.add(heading)

        # --- the basic relationships ------------------------------------------------
        speed = MathTex(r"v", "=", r"\frac{\lambda}{T}", "=", r"\lambda f", font_size=MATH_SIZE + 8)
        speed.shift(UP * 1.2)
        self.narrate("Here's the key connection. In one period, the wave moves forward by "
                     "exactly one wavelength.")
        self.narrate("So its speed is lambda over T, which is the same as lambda times f.",
                     Write(speed))
        self.key_idea(speed)
        definitions = MathTex(r"k = \frac{2\pi}{\lambda}", r"\qquad", r"\omega = 2\pi f",
                              r"\qquad", r"v = \frac{\omega}{k}", font_size=MATH_SIZE)
        definitions.next_to(speed, DOWN, buff=1.0)
        definitions[0].set_color(WAVE)
        definitions[2].set_color(IMAG)
        self.narrate("Physicists love radians, so we also define the wavenumber k, two pi "
                     "over lambda, and the angular frequency omega, two pi times f.",
                     Write(definitions[0]), Write(definitions[2]))
        self.narrate("In this language the speed is simply omega over k.",
                     Write(definitions[4]))
        self.clear_screen()

    # =========================================================================
    # 2. Vectors
    # =========================================================================
    def vectors(self):
        self.section_title("2. Vectors")

        # A NumberPlane is a pair of axes with a background grid.
        plane = NumberPlane(x_range=[-1, 6, 1], y_range=[-1, 5, 1], x_length=6.3, y_length=5.4,
                            background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.25},
                            axis_config={"stroke_color": MUTED})
        plane.to_edge(LEFT, buff=0.6).shift(DOWN * 0.4)
        origin = plane.c2p(0, 0)
        a_xy, b_xy = np.array([3.0, 2.0]), np.array([1.0, 2.5])   # components of A and B
        tip_a = plane.c2p(*a_xy)
        vec_a = Arrow(origin, tip_a, buff=0, color=VECTOR_A)
        label_a = MathTex(r"\vec A", color=VECTOR_A).next_to(vec_a.get_center(), DOWN + RIGHT * 0.5)

        self.narrate("Next up: vectors. A vector is a quantity with both a size and a "
                     "direction, like the velocity of a wave, or the displacement of a particle.",
                     Create(plane))
        self.narrate("We draw it as an arrow.", GrowArrow(vec_a), Write(label_a))

        # components: dashed projections on the axes
        corner = plane.c2p(a_xy[0], 0)
        comp_x = Arrow(origin, corner, buff=0, color=REAL, stroke_width=4)
        comp_y = Arrow(corner, tip_a, buff=0, color=IMAG, stroke_width=4)
        lab_x = MathTex(r"A_x\,\hat x", color=REAL, font_size=36).next_to(comp_x, DOWN, buff=0.1)
        lab_y = MathTex(r"A_y\,\hat y", color=IMAG, font_size=36).next_to(comp_y, RIGHT, buff=0.1)
        decomposition = MathTex(r"\vec A", "=", r"A_x\,\hat x", "+", r"A_y\,\hat y",
                                font_size=MATH_SIZE)
        decomposition[0].set_color(VECTOR_A)
        decomposition[2].set_color(REAL)
        decomposition[4].set_color(IMAG)
        decomposition.to_edge(RIGHT, buff=0.8).shift(UP * 2)
        self.narrate("Any vector in the plane can be built from two pieces, its components "
                     "along the x and y axes.", GrowArrow(comp_x), GrowArrow(comp_y),
                     Write(lab_x), Write(lab_y))
        self.narrate("With the unit vectors x hat and y hat, each of length one, we write: "
                     "A equals A x, x hat, plus A y, y hat.", Write(decomposition))

        magnitude = MathTex(r"|\vec A| = \sqrt{A_x^2 + A_y^2}", font_size=MATH_SIZE)
        direction = MathTex(r"\theta = \arctan\frac{A_y}{A_x}", font_size=MATH_SIZE)
        VGroup(magnitude, direction).arrange(DOWN, aligned_edge=LEFT, buff=0.4)\
            .next_to(decomposition, DOWN, buff=0.5).align_to(decomposition, LEFT)
        theta_arc = Arc(radius=0.8, start_angle=0, angle=np.arctan2(*a_xy[::-1]),
                        arc_center=origin, color=HIGHLIGHT)
        theta_label = MathTex(r"\theta", color=HIGHLIGHT, font_size=34)\
            .move_to(origin + 1.1 * np.array([np.cos(0.3), np.sin(0.3), 0]))
        self.narrate("Pythagoras gives its length, the magnitude. And a bit of trigonometry "
                     "gives its direction, the angle theta.",
                     Write(magnitude), Write(direction), Create(theta_arc), Write(theta_label))
        self.play(FadeOut(VGroup(comp_x, comp_y, lab_x, lab_y, theta_arc, theta_label,
                                 decomposition, magnitude, direction)))

        # addition, tip to tail
        vec_b = Arrow(origin, plane.c2p(*b_xy), buff=0, color=VECTOR_B)
        label_b = MathTex(r"\vec B", color=VECTOR_B).next_to(vec_b.get_end(), LEFT, buff=0.1)
        self.narrate("Adding vectors is easy. Here's a second vector, B.",
                     GrowArrow(vec_b), Write(label_b))
        moved_b = Arrow(tip_a, plane.c2p(*(a_xy + b_xy)), buff=0, color=VECTOR_B)
        total = Arrow(origin, plane.c2p(*(a_xy + b_xy)), buff=0, color=HIGHLIGHT)
        total_label = MathTex(r"\vec A + \vec B", color=HIGHLIGHT, font_size=36)\
            .next_to(total.get_end(), UP, buff=0.1)
        sum_rule = MathTex(r"\vec A + \vec B = (A_x + B_x)\,\hat x + (A_y + B_y)\,\hat y",
                           font_size=38).to_edge(RIGHT, buff=0.5).shift(UP * 2)
        sliding_b = vec_b.copy()                    # the copy slides; the original B stays
        self.narrate("Slide B so that its tail sits on the tip of A. The sum goes from the "
                     "start of A to the end of B.",
                     Transform(sliding_b, moved_b), GrowArrow(total), Write(total_label))
        self.narrate("In components you just add the x's together and the y's together.",
                     Write(sum_rule))
        self.play(FadeOut(VGroup(total, total_label, sum_rule, sliding_b)))

        # dot product
        angle_a = np.arctan2(a_xy[1], a_xy[0])
        angle_b = np.arctan2(b_xy[1], b_xy[0])
        between = Arc(radius=1.1, start_angle=angle_a, angle=angle_b - angle_a,
                      arc_center=origin, color=HIGHLIGHT)
        between_label = MathTex(r"\theta", color=HIGHLIGHT, font_size=34)\
            .move_to(origin + 1.45 * np.array([np.cos((angle_a + angle_b) / 2),
                                               np.sin((angle_a + angle_b) / 2), 0]))
        dot_rule = MathTex(r"\vec A \cdot \vec B", "=", r"A_x B_x + A_y B_y", font_size=MATH_SIZE)
        dot_geom = MathTex(r"=", r"|\vec A|\,|\vec B|\cos\theta", font_size=MATH_SIZE)
        VGroup(dot_rule, dot_geom).arrange(DOWN, aligned_edge=LEFT, buff=0.4)\
            .to_edge(RIGHT, buff=0.8).shift(UP * 1.2)
        dot_geom.align_to(dot_rule[1], LEFT)
        self.narrate("One more operation we'll need: the dot product. Multiply matching "
                     "components, then add them up.", Write(dot_rule))
        self.narrate("Geometrically, it's the two lengths times the cosine of the angle "
                     "between the vectors: it measures how much they point the same way.",
                     Write(dot_geom), Create(between), Write(between_label))
        why = Text("plane waves: the phase contains k · r", font_size=26, color=MUTED)
        why.next_to(dot_geom, DOWN, buff=0.8).to_edge(RIGHT, buff=0.6)
        self.narrate("Why bother? Because in three dimensions the phase of a plane wave "
                     "contains exactly such a dot product, k dot r. We'll get there!",
                     FadeIn(why))
        self.clear_screen()

    # =========================================================================
    # 3. Complex numbers
    # =========================================================================
    def complex_numbers(self):
        heading = self.section_title("3. Complex numbers")

        i_squared = MathTex(r"i^2 = -1", font_size=MATH_SIZE + 12)
        self.narrate("Now for something that sounds scary, but is actually a wave lover's "
                     "best friend: complex numbers.")
        self.narrate("It all starts with the imaginary unit i, defined so that i squared "
                     "equals minus one. No real number can do that, so we simply invent one.",
                     Write(i_squared))
        self.play(FadeOut(i_squared))

        # ComplexPlane = NumberPlane whose points are complex numbers;
        # plane.n2p(z) ("number to point") gives the screen position of z.
        plane = ComplexPlane(x_range=[-3, 4, 1], y_range=[-3, 3, 1], x_length=6.3, y_length=5.4,
                             background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.25},
                             axis_config={"stroke_color": MUTED})
        plane.to_edge(LEFT, buff=0.6).shift(DOWN * 0.4)
        re_label = Text("Re", font_size=26, color=REAL).next_to(plane.n2p(4), DOWN, buff=0.15)
        im_label = Text("Im", font_size=26, color=IMAG).next_to(plane.n2p(3j), RIGHT, buff=0.15)
        z = complex(3, 2)
        origin = plane.n2p(0)
        z_arrow = Arrow(origin, plane.n2p(z), buff=0, color=WAVE)
        z_dot = Dot(plane.n2p(z), color=WAVE)
        z_label = MathTex("z = a + ib", color=WAVE, font_size=38).next_to(z_dot, UR, buff=0.1)
        real_part = DashedLine(origin, plane.n2p(z.real), color=REAL)
        imag_part = DashedLine(plane.n2p(z.real), plane.n2p(z), color=IMAG)
        a_label = MathTex("a", color=REAL, font_size=36).next_to(real_part, DOWN, buff=0.1)
        b_label = MathTex("b", color=IMAG, font_size=36).next_to(imag_part, RIGHT, buff=0.1)

        self.narrate("A complex number z, a plus i b, has a real part, a, and an imaginary "
                     "part, b.", Create(plane), FadeIn(re_label), FadeIn(im_label))
        self.narrate("Use them as coordinates, and every complex number becomes a point, or "
                     "an arrow, in the complex plane.",
                     Create(real_part), Write(a_label), Create(imag_part), Write(b_label),
                     FadeIn(z_dot), Write(z_label))
        polar = MathTex(r"|z| = \sqrt{a^2 + b^2}", font_size=MATH_SIZE)
        phase = MathTex(r"\theta = \arctan\frac{b}{a}", font_size=MATH_SIZE)
        polar_form = MathTex(r"z = |z|\,(\cos\theta + i\sin\theta)", font_size=MATH_SIZE)
        VGroup(polar, phase, polar_form).arrange(DOWN, aligned_edge=LEFT, buff=0.45)\
            .to_edge(RIGHT, buff=0.7).shift(UP * 0.8)
        theta_arc = Arc(radius=0.7, start_angle=0, angle=np.angle(z), arc_center=origin,
                        color=HIGHLIGHT)
        self.narrate("Looks a lot like a 2D vector, right? The length of the arrow is the "
                     "magnitude of z, and its angle is called the phase.",
                     GrowArrow(z_arrow), Write(polar), Write(phase), Create(theta_arc))
        self.narrate("So we can also write z using its length and its angle: magnitude, "
                     "times cosine theta plus i sine theta.", Write(polar_form))
        self.clear_screen()
        self.add(heading)

        # --- multiplying by i is a rotation by 90 degrees -----------------------------
        plane = ComplexPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1], x_length=5, y_length=5,
                             background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.25},
                             axis_config={"stroke_color": MUTED}).shift(LEFT * 3 + DOWN * 0.4)
        origin = plane.n2p(0)
        arrow = Arrow(origin, plane.n2p(1), buff=0, color=WAVE)
        names = ["1", "i", "-1", "-i"]
        spots = [1, 1j, -1, -1j]
        labels = VGroup(*(MathTex(name, font_size=40).move_to(plane.n2p(1.35 * spot))
                          for name, spot in zip(names, spots, strict=True)))
        self.play(Create(plane), GrowArrow(arrow), Write(labels[0]))
        self.narrate("Here's the magic trick. Start from the number one, and multiply it by i.")
        for step in range(1, 5):
            # Rotate turns the arrow around `about_point` along an arc of 90 degrees.
            animations = [Rotate(arrow, angle=PI / 2, about_point=origin)]
            if step < 4:                                 # after 4 turns we are back to 1
                animations.append(Write(labels[step]))
            self.play(*animations, run_time=1.2)
        rule = MathTex(r"\times\, i \;=\; \text{rotation by } 90^\circ", font_size=MATH_SIZE)
        rule.to_edge(RIGHT, buff=0.8).shift(UP * 1.5)
        self.narrate("One, i, minus one, minus i, and back to one. Multiplying by i is a "
                     "rotation by 90 degrees!", Write(rule))
        self.key_idea(rule)

        # --- general product: magnitudes multiply, angles add -------------------------
        z1 = 1.4 * np.exp(1j * PI / 9)              # |z1| = 1.4, angle 20°
        z2 = 1.3 * np.exp(1j * 5 * PI / 18)         # |z2| = 1.3, angle 50°
        # product: |z1 z2| = 1.82, angle 70° (still inside the plane, which reaches 2)
        arrow_1 = Arrow(origin, plane.n2p(z1), buff=0, color=VECTOR_A)
        arrow_2 = Arrow(origin, plane.n2p(z2), buff=0, color=VECTOR_B)
        product = Arrow(origin, plane.n2p(z1 * z2), buff=0, color=HIGHLIGHT)
        product_rule = VGroup(MathTex(r"|z_1 z_2| = |z_1|\,|z_2|", font_size=MATH_SIZE),
                              MathTex(r"\theta_{z_1 z_2} = \theta_1 + \theta_2", font_size=MATH_SIZE))
        product_rule.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        product_rule.next_to(rule, DOWN, buff=0.9).align_to(rule, LEFT)
        self.play(FadeOut(arrow), FadeOut(labels))
        self.narrate("In general, when you multiply two complex numbers, their magnitudes "
                     "multiply and their angles add.",
                     GrowArrow(arrow_1), GrowArrow(arrow_2), Write(product_rule))
        # Transform with path_arc: the copy of z1 swings by the angle of z2 while stretching.
        self.narrate("Rotation and stretching, all in one operation. That's exactly why "
                     "complex numbers are so handy for anything that oscillates.",
                     Transform(arrow_1.copy(), product, path_arc=np.angle(z2)), run_time=2.5)
        self.clear_screen()
        self.add(heading)

        # --- complex conjugate ----------------------------------------------------------
        plane = ComplexPlane(x_range=[-1, 4, 1], y_range=[-3, 3, 1], x_length=4.5, y_length=5.4,
                             background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.25},
                             axis_config={"stroke_color": MUTED}).shift(LEFT * 3.3 + DOWN * 0.4)
        origin = plane.n2p(0)
        z_arrow = Arrow(origin, plane.n2p(z), buff=0, color=WAVE)
        conj_arrow = Arrow(origin, plane.n2p(z.conjugate()), buff=0, color=IMAG)
        z_name = MathTex("z", color=WAVE).next_to(z_arrow.get_end(), UR, buff=0.1)
        conj_name = MathTex("z^*", color=IMAG).next_to(conj_arrow.get_end(), DOWN + RIGHT * 0.3, buff=0.1)
        mirror = DashedLine(plane.n2p(z), plane.n2p(z.conjugate()), color=MUTED)
        conj_rules = VGroup(MathTex(r"z^* = a - ib", font_size=MATH_SIZE),
                            MathTex(r"z\,z^* = a^2 + b^2 = |z|^2", font_size=MATH_SIZE))
        conj_rules.arrange(DOWN, aligned_edge=LEFT, buff=0.5).to_edge(RIGHT, buff=1.0)
        self.play(Create(plane), GrowArrow(z_arrow), Write(z_name))
        self.narrate("Last tool: the complex conjugate, z star. It flips the sign of the "
                     "imaginary part: a mirror image across the real axis.",
                     Create(mirror), GrowArrow(conj_arrow), Write(conj_name), Write(conj_rules[0]))
        self.narrate("And z times z star is the magnitude squared: always a real number. "
                     "We'll use that a lot, for instance to compute intensities.",
                     Write(conj_rules[1]))
        self.clear_screen()

    # =========================================================================
    # 4. Euler's relation
    # =========================================================================
    def euler_relation(self):
        heading = self.section_title("4. Euler's relation")
        euler = MathTex(r"e^{i\theta}", "=", r"\cos\theta", "+", r"i\sin\theta",
                        font_size=MATH_SIZE + 8)
        euler[2].set_color(REAL)
        euler[4].set_color(IMAG)
        euler.to_edge(UP, buff=1.2)
        self.narrate("Now, perhaps the most beautiful formula you'll see in this course: "
                     "Euler's relation.", Write(euler))

        # --- geometry: a unit circle on the left, a graph on the right ----------------
        # The circle's radius on screen equals ONE unit of the graph's vertical axis,
        # so the horizontal dashed connector joins points at the same height.
        radius = 1.5                                 # screen units representing 1
        center = np.array([-4.6, -1.2, 0])
        circle = Circle(radius=radius, color=MUTED).move_to(center)
        real_axis = Line(center + LEFT * 2.1, center + RIGHT * 2.1, color=MUTED, stroke_width=2)
        imag_axis = Line(center + DOWN * 2.1, center + UP * 2.1, color=MUTED, stroke_width=2)
        graph_axes = Axes(x_range=[0, TAU, PI / 2], y_range=[-1.2, 1.2, 1],
                          x_length=7, y_length=2.4 * radius,
                          axis_config={"color": MUTED, "include_tip": False})
        # move the graph so that its origin is 3.7 units to the right of the circle
        # centre, at the same height: y = 0 of the graph is level with the real axis
        graph_axes.shift(center + RIGHT * 3.7 - graph_axes.c2p(0, 0))
        sine_label = MathTex(r"\sin\theta", color=IMAG, font_size=36)\
            .next_to(graph_axes.c2p(TAU, 1.0), RIGHT, buff=0.1)

        theta = ValueTracker(0.0)

        def point_on_circle() -> np.ndarray:
            """Screen position of e^{iθ}: (cos θ, sin θ) scaled by the radius."""
            angle = theta.get_value()
            return center + radius * np.array([np.cos(angle), np.sin(angle), 0])

        spinner = always_redraw(lambda: Arrow(center, point_on_circle(), buff=0, color=WAVE))
        cos_part = always_redraw(lambda: Line(center, [point_on_circle()[0], center[1], 0],
                                              color=REAL, stroke_width=6))
        sin_part = always_redraw(lambda: Line([point_on_circle()[0], center[1], 0],
                                              point_on_circle(), color=IMAG, stroke_width=6))
        # The graph of sin θ grows up to the current θ (at θ = 0 there is nothing to draw).
        sine_trace = always_redraw(
            lambda: graph_axes.plot(np.sin, x_range=[0, max(theta.get_value(), 1e-3)], color=IMAG)
        )
        connector = always_redraw(lambda: DashedLine(
            point_on_circle(), graph_axes.c2p(theta.get_value(), np.sin(theta.get_value())),
            color=MUTED, stroke_width=2))

        self.narrate("Since cos squared plus sin squared is one, e to the i theta always sits "
                     "on the unit circle, at angle theta from the real axis.",
                     Create(circle), Create(real_axis), Create(imag_axis), GrowArrow(spinner))
        self.add(cos_part, sin_part)
        self.narrate("As theta grows, the point goes round and round.",
                     theta.animate.set_value(PI / 3), run_time=2)
        self.add(graph_axes, sine_trace, connector)
        self.narrate("Now watch its shadows. The horizontal shadow, the real part, is cosine "
                     "theta. The vertical shadow, the imaginary part, is sine theta.",
                     FadeIn(graph_axes), theta.animate.set_value(TAU), FadeIn(sine_label),
                     run_time=8)
        self.narrate("A rotating arrow secretly contains two oscillations, ninety degrees "
                     "apart. Keep this picture in mind: it is the heart of the next episode.")
        self.clear_screen()
        self.add(heading)

        # --- compact polar form ---------------------------------------------------------
        compact = MathTex(r"z = |z|\,e^{i\theta}", font_size=MATH_SIZE + 12).shift(UP * 1.5)
        product = MathTex(r"z_1 z_2 = |z_1||z_2|\,e^{i(\theta_1 + \theta_2)}",
                          font_size=MATH_SIZE).next_to(compact, DOWN, buff=0.7)
        self.narrate("Euler gives the neatest way to write any complex number: magnitude "
                     "times e to the i theta.", Write(compact))
        self.narrate("And multiplying just adds the angles, simply because exponents add.",
                     Write(product))
        self.play(FadeOut(compact), FadeOut(product))

        # --- cos θ from two counter-rotating arrows ------------------------------------------
        center = np.array([-3.0, -0.6, 0])
        scale = 2.2                                  # screen units representing 1
        real_axis = Line(center + LEFT * 2.8, center + RIGHT * 2.8, color=MUTED, stroke_width=2)
        imag_axis = Line(center + DOWN * 2.5, center + UP * 2.5, color=MUTED, stroke_width=2)
        theta.set_value(PI / 4)

        def half_turns() -> VGroup:
            """e^{iθ}/2 from the origin, then e^{-iθ}/2 from its tip: their sum is cos θ."""
            angle = theta.get_value()
            first_tip = center + scale * 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
            sum_tip = center + scale * np.array([np.cos(angle), 0, 0])
            return VGroup(
                Arrow(center, first_tip, buff=0, color=WAVE, stroke_width=5),
                Arrow(first_tip, sum_tip, buff=0, color=VECTOR_B, stroke_width=5),
                Line(center, sum_tip, color=REAL, stroke_width=8),
            )

        arrows = always_redraw(half_turns)
        cos_formula = MathTex(r"\cos\theta", "=", r"\frac{e^{i\theta} + e^{-i\theta}}{2}",
                              font_size=MATH_SIZE).to_edge(RIGHT, buff=0.8).shift(UP * 1.0)
        cos_formula[0].set_color(REAL)
        sin_formula = MathTex(r"\sin\theta = \frac{e^{i\theta} - e^{-i\theta}}{2i}",
                              font_size=MATH_SIZE).next_to(cos_formula, DOWN, buff=0.6)
        self.play(Create(real_axis), Create(imag_axis), FadeIn(arrows))
        self.narrate("Euler also works backwards. Two half-length arrows spinning in "
                     "opposite directions...", theta.animate.set_value(PI / 4 + PI), run_time=4)
        self.narrate("...always add up to a point on the real axis: their sum is exactly "
                     "cosine theta.", Write(cos_formula), theta.animate.set_value(TAU + PI / 4),
                     run_time=4)
        self.narrate("A similar trick, with a minus sign, gives the sine.", Write(sin_formula))
        self.clear_screen()
        self.add(heading)
        bonus = MathTex(r"e^{i\pi} + 1 = 0", font_size=MATH_SIZE + 20)
        self.narrate("And a little bonus: set theta equal to pi. e to the i pi, plus one, "
                     "equals zero. Five of the most important numbers in maths, in one line.",
                     Write(bonus))
        self.clear_screen()

    # =========================================================================
    # 5. Wave functions
    # =========================================================================
    def wave_functions(self):
        heading = self.section_title("5. Wave functions")
        axes = Axes(x_range=[-7, 7, 1], y_range=[-0.4, 1.4, 0.5], x_length=12.5, y_length=3.2,
                    axis_config={"color": MUTED, "include_tip": False}).shift(DOWN * 1.2)
        x_label = MathTex("x", font_size=36).next_to(axes.c2p(7, 0), RIGHT, buff=0.1)
        t = ValueTracker(0.0)
        speed = 1.5                                  # pulse speed [units/s]
        start = -4.0                                 # pulse position at t = 0

        function = MathTex(r"y(x,t)", "=", r"f(x - vt)", font_size=MATH_SIZE).to_edge(UP, buff=1.2)
        function[2].set_color(WAVE)
        shape = axes.plot(lambda x: pulse(x - start), color=WAVE)
        self.narrate("Finally, let's put things together and describe a wave with a function. "
                     "The displacement y depends on position x and on time t.",
                     Create(axes), FadeIn(x_label))
        self.narrate("Take any shape f of x, like this pulse.", Create(shape))

        moving = always_redraw(lambda: axes.plot(
            lambda x: pulse(x - start - speed * t.get_value()), color=WAVE))
        peak = always_redraw(lambda: Dot(axes.c2p(start + speed * t.get_value(), 1.0),
                                         color=HIGHLIGHT))
        peak_note = MathTex(r"x - vt = \text{const}", color=HIGHLIGHT, font_size=34)
        peak_note.add_updater(lambda m: m.next_to(peak, UP, buff=0.15))
        self.remove(shape)
        self.add(moving)
        self.narrate("Now replace x by x minus v t. As time goes on, the whole shape slides "
                     "to the right at speed v, without changing its form.",
                     Write(function), t.animate.set_value(3.0), run_time=5)
        self.add(peak, peak_note)
        self.narrate("Why? Follow the peak. It sits where x minus v t keeps the same value, "
                     "that is, at a point moving with speed v.",
                     t.animate.set_value(5.0), run_time=4)
        peak_note.clear_updaters()
        self.remove(peak, peak_note)

        # the same pulse travelling left: f(x + vt)
        left_function = MathTex(r"y(x,t)", "=", r"f(x + vt)", font_size=MATH_SIZE)\
            .move_to(function)
        left_function[2].set_color(VECTOR_B)
        t_back = ValueTracker(0.0)
        moving_left = always_redraw(lambda: axes.plot(
            lambda x: pulse(x + start + speed * t_back.get_value()), color=VECTOR_B))
        self.remove(moving)
        self.add(moving_left)
        self.narrate("Flip the sign, f of x plus v t, and the pulse travels to the left.",
                     Transform(function, left_function), t_back.animate.set_value(5.0),
                     run_time=5)
        self.clear_screen()
        self.add(heading)

        # --- the harmonic wave -----------------------------------------------------------
        axes = Axes(x_range=[0, 4 * PI, PI], y_range=[-1.3, 1.3, 1], x_length=12, y_length=3,
                    axis_config={"color": MUTED, "include_tip": False}).shift(DOWN * 1.4)
        t.set_value(0.0)
        phi = ValueTracker(0.0)
        # A cosine (rather than a sine) so that it is exactly the real part of the
        # complex form shown below: Re[A e^{iα}] = A cos α. (A sine is the same wave
        # with φ shifted by π/2.) On screen: A = 1, k = 1, ω = 1.5.
        harmonic_wave = always_redraw(lambda: axes.plot(
            lambda x: np.cos(x - 1.5 * t.get_value() + phi.get_value()), color=WAVE))
        formula = MathTex("y(x,t)", "=", "A", r"\cos(", "kx", "-", r"\omega t", "+", r"\phi", ")",
                          font_size=MATH_SIZE + 4).to_edge(UP, buff=1.2)
        formula[2].set_color(REAL)
        formula[4].set_color(WAVE)
        formula[6].set_color(IMAG)
        formula[8].set_color(HIGHLIGHT)
        self.narrate("The most important example of all is the harmonic wave: A, times the "
                     "cosine of k x minus omega t, plus phi. A sine works just as well.",
                     Create(axes), Write(formula), FadeIn(harmonic_wave))
        same = MathTex(r"kx - \omega t = k\,(x - vt), \qquad v = \frac{\omega}{k}",
                       font_size=38).next_to(formula, DOWN, buff=0.4)
        self.narrate("It's f of x minus v t in disguise: k x minus omega t is k times x minus "
                     "v t, with v equal to omega over k.",
                     Write(same), t.animate.set_value(4.0), run_time=5)
        # A brace above the argument of the cosine (parts 4 to 8 of the formula).
        argument = VGroup(*formula[4:9])
        phase_brace = Brace(argument, UP, color=MUTED, buff=0.1)
        phase_brace_note = VGroup(phase_brace,
                                  phase_brace.get_text("phase").scale(0.7).set_color(MUTED))
        self.narrate("The whole argument of the cosine is called the phase. And phi, the phase "
                     "constant, slides the wave along: it fixes where the wave starts at time "
                     "zero.", FadeIn(phase_brace_note), phi.animate.set_value(PI / 2),
                     run_time=4)
        self.play(phi.animate.set_value(0.0), run_time=2)
        self.play(FadeOut(same), FadeOut(phase_brace_note))

        # --- the complex form: a teaser for phasors ----------------------------------------
        complex_form = MathTex(r"y(x,t)", "=", r"\mathrm{Re}\left[A\,e^{i(kx - \omega t)}\right]",
                               font_size=MATH_SIZE).next_to(formula, DOWN, buff=0.4)
        complex_form[2].set_color(REAL)
        self.narrate("And here's the payoff of Euler's relation: the same wave is the real "
                     "part of A, e to the i, k x minus omega t.", Write(complex_form))
        x_probe = 1.5 * PI                            # the point where we attach a phasor
        center = np.array([5.6, 1.45, 0])             # centre of the phasor's circle
        # Same scale as the graph: one unit of y on the axes = `radius` screen units,
        # so the real part of the phasor and the height of the wave have equal lengths.
        radius = axes.c2p(0, 1)[1] - axes.c2p(0, 0)[1]

        def phasor() -> VGroup:
            """The rotating arrow A e^{i(k x_probe - ω t)}, its real part, and the wave point.

            Re[A e^{iα}] = A cos α is exactly y(x_probe, t): the amber segment on the
            phasor and the amber segment on the graph always have the same length.
            """
            angle = x_probe - 1.5 * t.get_value()
            tip = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            on_wave = axes.c2p(x_probe, np.cos(angle))
            return VGroup(Circle(radius=radius, color=MUTED, stroke_width=2).move_to(center),
                          Arrow(center, tip, buff=0, color=WAVE, stroke_width=5),
                          Line(center, [tip[0], center[1], 0], color=REAL, stroke_width=7),
                          Line(axes.c2p(x_probe, 0), on_wave, color=REAL, stroke_width=7),
                          Dot(on_wave, color=HIGHLIGHT))

        phasor_group = always_redraw(phasor)
        name = Text("a phasor", font_size=28, color=WAVE).next_to(center + DOWN * radius, DOWN,
                                                                    buff=0.1)
        self.narrate("Behind every point of the wave, there's a little rotating arrow. That "
                     "arrow is called a phasor.", FadeIn(phasor_group), FadeIn(name),
                     t.animate.set_value(7.0), run_time=5)
        self.narrate("And phasors are exactly where we'll start next time.",
                     t.animate.set_value(9.0), run_time=3)
        self.clear_screen()

    # =========================================================================
    # 6. Recap
    # =========================================================================
    def recap(self):
        self.section_title("Recap")
        points = VGroup(
            MathTex(r"\text{a wave: a travelling disturbance}, \quad v = \lambda f = \omega / k"),
            MathTex(r"\text{vectors: } \vec A = A_x \hat x + A_y \hat y, \quad "
                    r"\vec A\cdot\vec B = |\vec A||\vec B|\cos\theta"),
            MathTex(r"\text{complex numbers: } z = a + ib = |z|\,e^{i\theta}, \quad "
                    r"\times i = \text{rotation by } 90^\circ"),
            MathTex(r"\text{Euler: } e^{i\theta} = \cos\theta + i\sin\theta"),
            MathTex(r"\text{travelling waves: } f(x \mp vt), \quad A\cos(kx - \omega t + \phi)"),
        )
        for point in points:
            point.scale(0.85)
        points.arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(DOWN * 0.2)
        self.narrate("Let's recap. A wave is a travelling disturbance, with speed lambda f.",
                     FadeIn(points[0], shift=RIGHT * 0.3))
        self.narrate("Vectors have components, a magnitude and a direction.",
                     FadeIn(points[1], shift=RIGHT * 0.3))
        self.narrate("Complex numbers live in a plane, and multiplying them rotates.",
                     FadeIn(points[2], shift=RIGHT * 0.3))
        self.narrate("Euler connects exponentials with cosines and sines.",
                     FadeIn(points[3], shift=RIGHT * 0.3))
        self.narrate("And any function of x minus v t is a wave travelling to the right.",
                     FadeIn(points[4], shift=RIGHT * 0.3))
        self.narrate("Thanks for watching, and see you in episode two, where the phasors "
                     "start spinning!")
        self.clear_screen()


# ---------------------------------------------------------------------------
# One scene per section, handy while developing: they reuse the methods above.
# (Subclassing Episode01 and overriding construct is all that is needed.)
# ---------------------------------------------------------------------------
class Ep01Intro(Episode01):
    def construct(self):
        self.intro()


class Ep01WhatIsAWave(Episode01):
    def construct(self):
        self.what_is_a_wave()


class Ep01Vectors(Episode01):
    def construct(self):
        self.vectors()


class Ep01ComplexNumbers(Episode01):
    def construct(self):
        self.complex_numbers()


class Ep01Euler(Episode01):
    def construct(self):
        self.euler_relation()


class Ep01WaveFunctions(Episode01):
    def construct(self):
        self.wave_functions()


class Ep01Recap(Episode01):
    def construct(self):
        self.recap()
