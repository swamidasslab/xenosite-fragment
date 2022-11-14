from typing import Union, NamedTuple, Optional, Sequence
from enum import Enum
from .morgan import morgan
from . import serialize
import numpy as np


class Graph:
    def __init__(
        self,
        n: int,
        edge: tuple[Sequence[np.int64], Sequence[np.int64]],
        nlabel: Optional[list[str]] = None,
        elabel: Optional[list[str]] = None,
    ):

        self.n = n
        self.edge: tuple[np.ndarray[np.int64], np.ndarray[np.int64]] = (
            np.asarray(edge[1], dtype=np.int64),  # type: ignore
            np.asarray(edge[0], dtype=np.int64),  # type: ignore
        )
        self.nlabel = nlabel
        self.elabel = elabel

    def __repr__(self):
        return f"Graph(n={self.n},\n\tedge={self.edge},\n\tnlabel={self.nlabel},\n\telabel={self.elabel})"

    def edge_to_node(self) -> "Graph":
        if self.nlabel and not self.elabel:
            raise ValueError("edge_to_node: edge labels missing.")

        if self.elabel and not self.nlabel:
            raise ValueError("edge_to_node: node labels missing.")

        nlabel = self.nlabel + self.elabel if self.nlabel and self.elabel else []

        e1 = self.edge[0]
        e2 = self.edge[1]
        ne = len(e1)

        eid = np.arange(ne, dtype=np.int64) + self.n

        e1 = np.concatenate([e1.ravel(), e2.ravel()])
        e2 = np.concatenate([eid, eid])

        return Graph(n=self.n + ne, edge=(e1, e2), nlabel=nlabel)  # type: ignore

    def morgan(self) -> np.ndarray[np.int64]:
        if self.elabel:
            return self.edge_to_node().morgan()[: self.n]

        e1 = self.edge[0]
        e2 = self.edge[1]

        return morgan(self.nlabel, e1, e2)  # type: ignore

    def serialize(self, canonize=True) -> serialize.Serialized:
        return serialize.serialize(self, canonize)

    def subgraph(self, nidx: list[int], eidx: Optional[list[int]] = None) -> "Graph":
        ns = set(nidx)
        assert len(nidx) == len(ns)

        n = len(ns)
        nlabel = [self.nlabel[i] for i in nidx] if self.nlabel else None

        eidx = eidx or [
            e for e, (i, j) in enumerate(zip(*self.edge)) if i in ns and j in ns
        ]

        renum = -np.ones(self.n, dtype=np.int64)
        for i, nid in enumerate(nidx):
            renum[nid] = i

        elabel = [self.elabel[i] for i in eidx] if self.elabel else None

        e1 = renum[self.edge[0][eidx]]
        e2 = renum[self.edge[1][eidx]]

        return Graph(n=n, edge=(e1, e2), nlabel=nlabel, elabel=elabel)  # type: ignore

    @classmethod
    def from_molecule(cls, molecule, smiles=False) -> "Graph":
        from . import (
            chem,
        )  # lazy import to prevent load of rdkit unless needed

        if smiles:
            return chem.MolToSmilesGraph(molecule)
        return chem.MolToSmartsGraph(molecule)


class DFS_TYPE(Enum):
    TREE = 0
    RING = 1


class DFS_EDGE(NamedTuple):
    i: int
    j: int
    t: DFS_TYPE


def _dfs(
    s: int,
    neighbors: dict[int, list[int]],
    visited: Optional[set[Union[int, tuple[int, int]]]] = None,
    out: Optional[list[DFS_EDGE]] = None,
) -> list[DFS_EDGE]:

    visited = visited or set()
    out = out or []
    visited.add(s)
    ns = neighbors[s]

    for n in ns:
        e = tuple((n, s) if n < s else (s, n))
        if e in visited:
            continue

        if n in visited:
            visited.add(e)
            out.append(DFS_EDGE(s, n, DFS_TYPE.RING))

        else:
            visited.add(e)
            out.append(DFS_EDGE(s, n, DFS_TYPE.TREE))
            _dfs(n, neighbors, visited, out)
    return out


def dfs_ordered(G: Graph, canonize=True) -> list[DFS_EDGE]:
    N = neighbors(G)
    start = 0

    if canonize:
        M = G.morgan()
        start = int(np.argmin(M))
        for n in N:
            N[n] = sorted(N[n], key=lambda x: M[x])
    else:
        for n in N:
            N[n] = sorted(N[n])
    return _dfs(start, N)


def neighbors(G: Graph) -> dict[int, list[int]]:
    N = {n: [] for n in range(G.n)}

    for i, j in zip(G.edge[0], G.edge[1]):
        N[i].append(j)
        N[j].append(i)
    return N


def ring_graph(n: int) -> Graph:
    e1 = np.arange(n)
    e2 = np.roll(e1, 1)  # type: ignore
    return Graph(n=n, edge=(e1, e2), nlabel=["*"] * n, elabel=[""] * n)  # type: ignore


def star_graph(n: int) -> Graph:
    e1 = np.zeros(n - 1)
    e2 = np.arange(n - 1) + 1  # type: ignore
    return Graph(n=n, edge=(e1, e2), nlabel=["*"] * n, elabel=[""] * (n - 1))  # type: ignore
