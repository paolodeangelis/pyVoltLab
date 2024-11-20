"""
Unit and regression test for the mc_mace package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest  # noqa: F401

import mc_mace  # noqa: F401


def test_mc_mace_imported() -> None:
    """Sample test, will always pass so long as import statement worked."""
    assert "mc_mace" in sys.modules
