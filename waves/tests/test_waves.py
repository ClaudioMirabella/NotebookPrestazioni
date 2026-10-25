"""Checks of the waves series code that do not need rendering (skipped without Manim)."""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("manim")
# the modules live one level up, in the series folder
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import ep01_vectors_complex_wavefunctions as ep01  # noqa: E402
import wavekit  # noqa: E402


def test_speaking_time_follows_word_count():
    ten_words = "one two three four five six seven eight nine ten"
    assert wavekit.speaking_time(ten_words) == pytest.approx(10 / wavekit.WORDS_PER_SECOND)
    assert wavekit.speaking_time("Hi") == wavekit.MIN_LINE_DURATION


def test_harmonic_wave_is_periodic_and_travels_at_lambda_f():
    x = np.linspace(0, 10, 101)
    # one wavelength later in space, one period later in time: same displacement
    assert np.allclose(ep01.harmonic(x + ep01.WAVELENGTH, 0.3), ep01.harmonic(x, 0.3))
    assert np.allclose(ep01.harmonic(x, 0.3 + ep01.PERIOD), ep01.harmonic(x, 0.3))
    # f(x - vt): the shape at time t is the shape at t = 0 shifted by v t, v = λ / T
    v, t = ep01.WAVELENGTH / ep01.PERIOD, 0.7
    assert np.allclose(ep01.harmonic(x + v * t, t), ep01.harmonic(x, 0.0))


def test_pulse_is_a_unit_bump():
    assert ep01.pulse(0.0) == 1.0
    assert ep01.pulse(5.0) < 1e-10


def test_every_section_has_a_standalone_scene():
    sections = ["intro", "what_is_a_wave", "vectors", "complex_numbers", "euler_relation",
                "wave_functions", "recap"]
    for name in sections:
        assert callable(getattr(ep01.Episode01, name))
    standalone = [cls for cls in vars(ep01).values()
                  if isinstance(cls, type) and issubclass(cls, ep01.Episode01) and cls is not ep01.Episode01]
    assert len(standalone) == len(sections)
