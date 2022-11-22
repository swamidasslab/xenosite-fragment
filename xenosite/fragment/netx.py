import numpy as np
from typing import Optional, Union
from .graph import Graph, neighbors
from .chem import Fragment
from collections import defaultdict
from functools import reduce
import gzip
import pickle
import networkx as nx


class FragmentNetwork:
    max_size: int = 10
    agg_attr = ["count", "marked_count", "marked_ids"]

    def __init__(
        self,
        mol: Optional[Union[Graph, str]] = None,
        marked: Optional[set[int]] = None,
        max_size: Optional[int] = None,
    ):
        if not mol:
            self.network = nx.DiGraph()
            self.molrefs = nx.DiGraph()
            return

        mol = Fragment(mol).graph
        marked = marked or set()

        network = nx.DiGraph()

        if max_size:
            self.max_size = max_size

        id_network = subgraph_network_ids(mol, self.max_size)
        frag2ids = defaultdict(lambda: [])

        for ids in id_network.nodes:
            serial = Fragment(mol, ids).canonical(remap=True)
            frag = serial.string  # type: ignore
            id_network.nodes[ids]["frag"] = frag
            frag2ids[frag].append(serial.reordering)

        amarked = np.array(list(marked))[None, None, :]

        for frag, ids in frag2ids.items():

            # normalized count of marked ids by position in fragment
            marked_ids = (np.array(ids)[:, :, None] == amarked).sum(axis=0).sum(axis=1)
            marked_ids = marked_ids / max(marked_ids.sum(), 1) * len(marked)  # type: ignore

            ids = [set(i) for i in ids]
            size = len(ids[0])
            count = len(reduce(lambda x, y: x | y, ids)) / size

            marked_count = (
                len(
                    reduce(
                        lambda x, y: x | y,
                        [i for i in ids if marked & i],
                        set(),
                    )
                )
                / size
            )

            network.add_node(
                frag, count=count, marked_count=marked_count, marked_ids=marked_ids
            )

        for u, v in id_network.edges:
            network.add_edge(id_network.nodes[u]["frag"], id_network.nodes[v]["frag"])

        self.network = network

    def save(self, filename: str):
        with gzip.GzipFile(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "FragmentNetwork":
        with gzip.GzipFile(filename, "rb") as f:
            return pickle.load(f)

    def update(self, other: "FragmentNetwork"):
        new_frags = []

        for frag in other.network.nodes:
            if frag in self.network:
                for att in self.agg_attr:
                    self.network.nodes[frag][att] = (
                        self.network.nodes[frag][att] + other.network.nodes[frag][att]
                    )
            else:
                new_frags.append(frag)

        for frag in new_frags:
            self.network.add_node(frag, **other.network.nodes[frag])
            for child in other.network[frag]:
                self.network.add_edge(frag, child)


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
