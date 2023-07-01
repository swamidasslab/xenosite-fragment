from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Optional, Sequence, Union, Any
from numba import njit
import numpy as np

from . import serialize
from .ops import morgan


class BaseGraph:
    def __init__(
        self,
        n: int,
        edge: tuple[Sequence[np.uint32], Sequence[np.uint32]],
        nlabel: Optional[list[str]] = None,
        elabel: Optional[list[str]] = None,
        nprops: Optional[dict[str, Sequence[Any]]] = None,
        eprops: Optional[dict[str, Sequence[Any]]] = None,
        info: Optional[dict[str, Any]] = None,
    ):

        self.n = n
        self.edge: tuple[np.ndarray[np.uint32], np.ndarray[np.uint32]] = (
            np.asarray(edge[1], dtype=np.uint32),  # type: ignore
            np.asarray(edge[0], dtype=np.uint32),  # type: ignore
        )
        self.nlabel = nlabel
        self.elabel = elabel
        self.nprops = nprops
        self.eprops = eprops
        self.info = info

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n},\n\tedge={self.edge},\n\tnlabel={self.nlabel},\n\telabel={self.elabel})"

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

        return self.__class__(n=n, edge=(e1, e2), nlabel=nlabel, elabel=elabel)  # type: ignore


class DirectedGraph(BaseGraph):
    pass


class Graph(BaseGraph):  # Undirected Graph
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
    
    def neighbors(self) -> list[list[int]]:
        N = [[] for _ in range(self.n)]

        for i, j in zip(self.edge[0], self.edge[1]):
            N[i].append(j)
            N[j].append(i)

        return N
    


    def morgan(self) -> np.ndarray[np.int64]: #type: ignore
        try:
            _ = self._morgan
        except:

          if self.elabel:
              self._morgan = self.edge_to_node().morgan()[: self.n]
          else:

              e1 = self.edge[0]
              e2 = self.edge[1]

              self._morgan = morgan(self.nlabel, e1, e2)  # type: ignore
    
        return self._morgan



    def serialize(self, canonize=True) -> serialize.Serialized:
        return serialize.serialize(self, canonize)

    @classmethod
    def from_molecule(cls, molecule, smiles=False) -> "Graph":
        from . import chem  # lazy import to prevent load of rdkit unless needed

        if smiles:
            return chem.MolToSmilesGraph(molecule)
        return chem.MolToSmartsGraph(molecule)

@njit
def _invert_mapping(x):
    y = np.zeros_like(x)
    for i in range(len(x)): y[x[i]] = i
    return y


class DFS_TYPE(int, Enum):
    TREE = 0
    RING = 1


class DFS_EDGE(NamedTuple):
    i: int
    j: int
    t: DFS_TYPE


def _dfs(
    start: int,
    neighbors: dict[int, list[int]],
    visited: Optional[set[Union[int, tuple[int, int]]]] = None,
    out: Optional[list[DFS_EDGE]] = None,
) -> list[DFS_EDGE]:

    visited = visited if visited is not None else set()
    out = out if out is not None else []
    visited.add(start)

    for n in neighbors[start]:
        e = tuple((n, start) if n < start else (start, n))
        if e in visited:
            continue

        if n in visited:
            visited.add(e)
            out.append(DFS_EDGE(start, n, DFS_TYPE.RING))

        else:
            visited.add(e)
            out.append(DFS_EDGE(start, n, DFS_TYPE.TREE))
            _dfs(n, neighbors, visited, out)
    return out


def dfs_ordered(G: Graph, canonize=True) -> list[DFS_EDGE]:
    N = neighbors(G)
    start = 0

    if G.n <= 1:
        return []

    if canonize:
        M = G.morgan()
        start = int(np.argmin(M))
        for n in N:
            N[n] = sorted(N[n], key=lambda x: M[x])
    else:
        for n in N:
            N[n] = sorted(N[n])

    return _dfs(start, N)

def neighbors(G: Graph) -> dict[int,list[int]]:
    N = [[] for _ in range(G.n)]

    for i, j in zip(G.edge[0], G.edge[1]):
        N[i].append(j)
        N[j].append(i)

    N = {i: N[i] for i in range(G.n)}
    return N
