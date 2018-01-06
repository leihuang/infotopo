"""
"""

import pytest

import util


def test_(pred):
    """
    """
    assert pred([1,2]) == util.Series([1,2,3])
    assert pred((1,2)) == util.Series([1,2,3])
    assert pred(np.array([1,2])) == util.Series([1,2,3])
    assert pred(util.Series([1,2])) == util.Series([1,2,3])




