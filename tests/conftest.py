"""Shared fixtures for the test suite."""

import os
import sys
import warnings

import pytest

# Make the package importable when the tests are run from a fresh checkout
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aircraft_performance import examples  # noqa: E402
from aircraft_performance import units as u  # noqa: E402
from aircraft_performance.errors import PerformanceWarning  # noqa: E402

#: Sea-level density used by Roskam in English units (0.002377 slug/ft^3)
RHO0_ROSKAM = 0.002377 * u.SLUG_FT3


@pytest.fixture(autouse=True)
def _silence_compressibility_warnings():
    """The Gulfstream IV-like polar deliberately ignores compressibility, as in Anderson.

    The warnings are tested explicitly in test_errors.py; here they would only add noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        yield


@pytest.fixture
def giv():
    return examples.business_jet_giv()


@pytest.fixture
def twin():
    return examples.roskam_light_twin()


@pytest.fixture
def single():
    return examples.light_single_piston()


@pytest.fixture
def turboprop():
    return examples.regional_turboprop()
