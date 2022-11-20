import numpy as np
import numba
from numba import jit, njit

from typing import Iterator, Sequence


@njit(cache=True)
def to_primes(x) -> tuple[numba.int64[:], numba.int64]:  # type: ignore
    u = np.unique(x)
    u = sorted(u)
    p = _p[: len(u)]
    d = {v: n for n, v in enumerate(u)}
    return np.array([p[d[v]] for v in x]), len(u)


@njit(cache=True)
def nself_prod(
    v: numba.int64[:], e1: numba.int64[:], e2: numba.int64[:]  # type: ignore
) -> numba.int64[:]:  # type: ignore
    out = v.copy()
    for i, j in zip(e1, e2):
        out[i] *= v[j]
        out[j] *= v[i]
    return out


@njit(cache=True)
def _morgan(v, e1, e2) -> numba.int64[:]:  # type: ignore
    p, u = to_primes(v)
    history = [(u, p)]
    while True:
        p = nself_prod(p, e1, e2)
        p, u = to_primes(p)
        if u == len(p):
            return p
        history.append((u, p))

        for v, q in history[:-1]:
            if v != u:
                continue
            if np.all(q == p):
                return p


def morgan(v: Sequence, e1: Sequence[int], e2: Sequence[int]) -> np.ndarray[np.int64]:
    v = np.asarray(v)  # type: ignore
    e1 = np.asarray(e1, dtype=np.int64)  # type: ignore
    e2 = np.asarray(e2, dtype=np.int64)  # type: ignore
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


_p = np.array([p for p, _ in zip(gen_primes(), range(1000))])
