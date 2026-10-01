"""Geometry of the Manim animations: the drawn forces must satisfy equilibrium.

Skipped automatically where Manim is not installed (it is an optional dependency).
"""

import numpy as np
import pytest

manim = pytest.importorskip("manim")

import sys, pathlib  # noqa: E401,E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from animations.climb_descent import along_path, normal_to_path  # noqa: E402

ANGLES = np.radians([1.0, 5.0, 15.0, 30.0])


@pytest.mark.parametrize("climbing", [True, False])
@pytest.mark.parametrize("gamma", ANGLES)
def test_unit_vectors_are_orthonormal(gamma, climbing):
    u, n = along_path(gamma, climbing), normal_to_path(gamma, climbing)
    assert np.isclose(np.linalg.norm(u), 1) and np.isclose(np.linalg.norm(n), 1)
    assert np.isclose(u @ n, 0) and n[1] > 0          # lift points upwards


@pytest.mark.parametrize("gamma", ANGLES)
def test_climb_forces_in_equilibrium(gamma):
    W, D = 1.0, 0.3
    u, n = along_path(gamma), normal_to_path(gamma)
    L, T = W * np.cos(gamma), D + W * np.sin(gamma)    # as drawn in ClimbGeometry
    total = L * n + T * u - D * u + W * np.array([0, -1, 0])
    assert np.allclose(total, 0)


@pytest.mark.parametrize("gamma", ANGLES)
def test_glide_resultant_is_vertical(gamma):
    W = 1.0
    u, n = along_path(gamma, False), normal_to_path(gamma, False)
    R = W * np.cos(gamma) * n - W * np.sin(gamma) * u  # L + D as drawn in DescentGeometry
    assert np.allclose(R, [0, W, 0])
