
import networkx as nx
from typing import Sequence, Optional
import rdkit.Chem
from .base import FragmentNetworkBase
from ..graph import neighbors
from ..graph import Graph

class SubGraphFragmentNetwork(FragmentNetworkBase):

    def _remap_ids(self, ids: Sequence[int], id_network: nx.DiGraph) -> Sequence[int]:
        return ids

    def _subgraph_network_ids(self, rdmol: rdkit.Chem.Mol, mol: Graph) -> nx.DiGraph:  # type: ignore
        return subgraph_network_ids(mol, self.max_size)



def subgraph_network_ids(
    G: Graph,
    max_size: int = 10,
) -> nx.DiGraph:

    network = nx.DiGraph()

    N = neighbors(G)
    subgraphs = subgraphs_gen(N, max_size)
    subgraphs_set = set(subgraphs)

    for parent in subgraphs:
        network.add_node(parent)

        if len(parent) == 1:
            continue

        for n in parent:
            child = frozenset(parent - {n})
            if child in subgraphs_set:
                network.add_edge(parent, child)

    return network


def subgraphs_gen(
    neighbors: dict[int, list[int]],
    max_size: int = 10,
    start_nodes: Optional[list[int]] = None,
) -> list[frozenset[int]]:

    visited: set[int] = set()  # type: ignore
    emitted = set()

    Ns = {k: set(v) for k, v in neighbors.items()}

    assert max_size > 1

    for n in start_nodes or range(len(neighbors)):
        # each interation yields all subgraphs that contain n, but not in ignore
        frontier: set[int] = set()

        # yield all subgraphs that include n
        _extend_fast(Ns, n, visited, frontier, emitted, max_size)

        # remove n from neighbors
        for i in Ns[n]:
            Ns[i].remove(n)
        del Ns[n]

    return list(emitted)


def _extend_fast(
    Ns: dict[int, set[int]],
    start: int,
    visited: set[int],
    frontier: set[int],
    emitted: set[frozenset[int]],
    max_size: int,
) -> None:

    # represents the current substructure
    visited = frozenset(visited | {start})  # type: ignore

    # stop if this substructure has been emitted already
    if visited in emitted:
        return

    emitted.add(visited)  # type: ignore

    # stop if this substructure is maxsize
    if len(visited) >= max_size:
        return

    # update the frontier
    neighbors = Ns[start]
    frontier = (frontier | neighbors) - visited - {start}

    frontier_left = set(frontier)

    # visit each node on the frontier
    for n in frontier:
        frontier_left.remove(n)
        _extend_fast(Ns, n, visited, frontier_left, emitted, max_size)

