"""
Unit and regression test for the mc_mace package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mc_mace


def test_mc_mace_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mc_mace" in sys.modules
