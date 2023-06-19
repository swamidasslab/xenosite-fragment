from xenosite.fragment import ops as mg
from hypothesis import strategies as st, given
from pytest import approx
import pytest
import numpy as np


@pytest.mark.parametrize(
    "input, output",
    [
        ([1, 2, 2, 4], (2, 3, 3, 5)),
        ([1, 2, 3, 4], (2, 3, 5, 7)),
        ([2, 3, 4], (2, 3, 5)),
        ([5, 1, 2, 3, 4], (11, 2, 3, 5, 7)),
        ([4], (2,)),
        ([], ()),
    ],
    ids=["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"],
)
def test_primes(input, output):
    assert mg.to_primes(np.array(input))[0] == approx(output)


@pytest.mark.parametrize(
    "input, output",
    [
        ([1, 2, 2, 4], (0, 1, 1, 2)),
        ([1, 2, 3, 4], (0, 1, 2, 3)),
        ([2, 3, 4], (0, 1, 2)),
        ([5, 1, 2, 3, 4], (4, 0, 1, 2, 3)),
        ([4], (0,)),
        ([], ()),
    ],
    ids=["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"],
)
def test_range(input, output):
    assert mg.to_range(np.array(input))[0] == approx(output)
