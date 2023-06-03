from __future__ import annotations

import numpy as np
import numba
from numba import jit, njit
import scipy.stats
from typing import Iterator, Sequence, Union


@njit(cache=True)
def to_primes(x) -> tuple[numba.uint64[:], numba.uint32]:  # type: ignore
    u = np.unique(x)
    d = {v: n for n, v in enumerate(sorted(u))}
    return np.array([_p[d[v]] for v in x]), len(u)


def to_range(x): # slower implementation of to_range_numba, but works without errors.
    r = np.array(scipy.stats.rankdata(x, 'dense'), dtype=np.uint32) - 1 
    n = len(np.unique(r))
    return r, n

@njit(cache=True) # sometimes has errors related to caching (?)
def to_range_numba(x : numba.uint64[:]) -> numba.uint32[:]:  # type: ignore
    u = np.unique(x)
    d = {v: n for n, v in enumerate(sorted(u))}
    return np.array([d[v] for v in x]), len(u)


@njit(cache=True)
def collect_product(
    v: numba.uint64[:], e1: numba.uint32[:], e2: numba.uint32[:]  # type: ignore
) -> numba.uint64[:]:  # type: ignore
    out = v.copy()
    for i, j in zip(e1, e2):
        out[i] *= v[j]
        out[j] *= v[i]
    return out


@njit(cache=True)
def _morgan(v, e1: numba.uint32[:], e2: numba.uint32[:]) -> numba.uint64[:]:  # type: ignore
    p, u = to_primes(v)
    history = [(u, p)]
    while True:
        p = collect_product(p, e1, e2)
        p, u = to_primes(p)
        if u == len(p):
            return p
        history.append((u, p))

        for v, q in history[:-1]:
            if v != u:
                continue
            if np.all(q == p):
                return p


def morgan(
    v: Union[Sequence, np.ndarray], e1: Sequence[np.uint32], e2: Sequence[np.uint32]
) -> np.ndarray[np.uint64]:
    v = np.asarray(v)  # type: ignore
    e1 = np.asarray(e1, dtype=np.uint32)  # type: ignore
    e2 = np.asarray(e2, dtype=np.uint32)  # type: ignore
    return _morgan(v, e1, e2)


# Sieve of Eratosthenes
# Code by David Eppstein, UC Irvine, 28 Feb 2002
# http://code.activestate.com/recipes/117119/
def gen_primes() -> Iterator[int]:
    """Generate an infinite sequence of prime numbers."""
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            #
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]

        q += 1


_p = np.array([p for p, _ in zip(gen_primes(), range(1000))], dtype=np.uint64)
