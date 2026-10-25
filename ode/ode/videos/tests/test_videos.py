"""Checks of the video code that need no rendering (skipped where Manim is not installed)."""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("manim")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # the episode modules

import ep02_logistic as ep02  # noqa: E402
import ep03_euler as ep03  # noqa: E402
import odekit  # noqa: E402


def test_speaking_time():
    assert odekit.speaking_time("one two three four five") == pytest.approx(5 / odekit.WORDS_PER_SECOND)
    assert odekit.speaking_time("Hi") == odekit.MIN_LINE_DURATION


def test_logistic_formula_used_in_episode_2():
    t = np.linspace(0, 20, 2001)
    x = np.array([ep02.logistic(s, 1.0) for s in t])
    residual = np.gradient(x, t) - ep02.K * x * (ep02.M - x)
    assert np.max(np.abs(residual[2:-2])) < 1e-4
    assert np.isnan(ep02.logistic(10.0, -0.5))           # beyond the blow-up time t* ~ 4.8


def test_euler_points_match_lebl_section_1_7():
    # Lebl, Section 1.7: two steps with h = 1 give 4/3 and 52/27
    pts = ep03.euler_points(ep03.f_lebl, 0, 1, 1.0, 2)
    assert pts[1][1] == pytest.approx(4 / 3)
    assert pts[2][1] == pytest.approx(52 / 27)
    # Table 1.7.1, h = 0.125
    assert ep03.euler_points(ep03.f_lebl, 0, 1, 0.125, 16)[-1][1] == pytest.approx(2.68034, abs=1e-5)


def test_every_section_has_a_standalone_scene():
    for module, base in ((ep02, ep02.Episode02), (ep03, ep03.Episode03)):
        standalone = [c for c in vars(module).values()
                      if isinstance(c, type) and issubclass(c, base) and c is not base]
        assert len(standalone) == 7
