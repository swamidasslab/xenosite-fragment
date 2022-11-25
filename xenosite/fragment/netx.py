import numpy as np
from typing import Optional, Union, Sequence
from .graph import Graph, neighbors
from .chem import Fragment
from collections import defaultdict
from scipy.stats import hypergeom
from functools import reduce
import gzip
import pickle
import networkx as nx
import rdkit
import logging

logger = logging.getLogger(__name__)


class FragmentNetwork:
    max_size: int = 10
    agg_attr = ["count", "marked_count", "marked_ids", "exp", "obs", "n"]
    _version: int = 1

    def __init__(
        self,
        smiles: Optional[str] = None,
        marked: Optional[set[int]] = None,
        max_size: Optional[int] = None,
    ):
        self.version: int = self._version

        if not smiles:
            self.network = nx.DiGraph()
            self.molrefs = nx.DiGraph()
            return

        rdmol: rdkit.Chem.Mol = rdkit.Chem.MolFromSmiles(smiles)  # type: ignore
        assert rdmol, f"Not a valid SMILES: ${smiles}"

        mol: Graph = Fragment(rdmol).graph
        marked = marked or set()

        network = nx.DiGraph()

        if max_size:
            self.max_size = max_size

        id_network = self._subgraph_network_ids(rdmol, mol)
        frag2reordering = defaultdict(lambda: [])
        # frag2ids = defaultdict(lambda: [])

        for ids in id_network.nodes:
            full_ids = self._remap_ids(ids, id_network)

            serial = Fragment(mol, full_ids).canonical(remap=True)
            frag = serial.string  # type: ignore
            id_network.nodes[ids]["frag"] = frag

            frag2reordering[frag].append(serial.reordering)
            # frag2ids[frag].append(ids)

        amarked = np.array(list(marked))[None, None, :]

        for frag, ids in frag2reordering.items():

            # normalized count of marked ids by position in fragment
            marked_ids = (np.array(ids)[:, :, None] == amarked).sum(axis=0).sum(axis=1)
            marked_ids = np.where(marked_ids > 0, 1, 0)  # type: ignore

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

            # exp = probability of fragment overlapping with at least one marked atom
            # given: size of molecule, number of atoms matching fragment, number of marked atoms
            exp = 1 - hypergeom.cdf(0, mol.n, int(count * size), len(marked))
            obs = 1 if marked_count else 0
            n = 1

            network.add_node(
                frag,
                count=count,
                marked_count=marked_count,
                marked_ids=marked_ids,
                n=n,
                exp=exp,
                obs=obs,
            )

        for u, v in id_network.edges:
            network.add_edge(id_network.nodes[u]["frag"], id_network.nodes[v]["frag"])

        self.network = network

    def to_pandas(self):
        import pandas as pd

        df = pd.DataFrame.from_dict(
            [
                {
                    "frag": frag,
                    "count": self.network.nodes[frag]["count"],
                    "count_marked": self.network.nodes[frag]["marked_count"],
                    "marked_ids": self.network.nodes[frag]["marked_ids"],
                    "size": len(self.network.nodes[frag]["marked_ids"]),
                    "n": self.network.nodes[frag]["n"],
                    "exp": self.network.nodes[frag]["exp"],
                    "obs": self.network.nodes[frag]["obs"],
                }
                for frag in self.network
            ]  # type: ignore
        )
        return df.set_index("frag")

    def _remap_ids(self, ids: Sequence[int], id_network: nx.DiGraph) -> Sequence[int]:
        return ids

    def _subgraph_network_ids(self, rdmol: rdkit.Chem.Mol, mol: Graph) -> nx.DiGraph:  # type: ignore
        return subgraph_network_ids(mol, self.max_size)

    def save(self, filename: str):
        with gzip.GzipFile(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "FragmentNetwork":
        with gzip.GzipFile(filename, "rb") as f:
            network = pickle.load(f)

        if not isinstance(network, cls):
            logger.warning(
                f" Loaded object is not the required class: {type(network)} != {cls}"
            )
            return network

        net_version = network.__dict__.get("version", 0)
        if net_version != cls._version:
            logger.warning(
                f" Network version does not match library: v{net_version} != v{cls._version}"
            )

        return network

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


import rdkit

from xenosite.fragment.chem import MolToSmartsGraph


class RingFragmentNetwork(FragmentNetwork):
    def _remap_ids(self, ids: Sequence[int], id_network: nx.DiGraph) -> Sequence[int]:
        mapping = id_network.mapping  # type: ignore
        return list(
            set(reduce(lambda x, y: x + y, [mapping[x] for x in sorted(ids)], []))
        )

    def _subgraph_network_ids(self, rdmol: rdkit.Chem.Mol, mol: Graph) -> nx.DiGraph:  # type: ignore
        graph = ring_graph(rdmol, mol)
        assert graph.nprops

        out = subgraph_network_ids(graph, self.max_size)
        out.mapping = graph.nprops["mapping"]  # type: ignore
        return out


def ring_graph(rdmol: rdkit.Chem.Mol, mol: Optional[Graph] = None, max_ring_size=8):  # type: ignore
    mol = mol or MolToSmartsGraph(rdmol)

    # sort rings and remove macro-cycles larger than max_ring_size
    rings = sorted(
        sorted(r) for r in rdmol.GetRingInfo().AtomRings() if len(r) <= max_ring_size
    )
    rings_set = [set(r) for r in rings]

    ring_atoms = reduce(lambda x, y: x | y, (set(r) for r in rings), set())
    non_ring_atoms = [x for x in range(mol.n) if x not in ring_atoms]

    mapping = rings + [[x] for x in non_ring_atoms]

    mapping_inverted = {x: n + len(rings) for n, x in enumerate(non_ring_atoms)}

    N = neighbors(mol)
    N = {k: set(v) for k, v in N.items()}

    ring_N = {
        n: reduce(lambda x, y: x | y, (N[a] for a in r), set())
        for n, r in enumerate(rings)
    }

    edges = []

    # ring-ring edges
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if ring_N[i] & rings_set[j]:
                edges.append((i, j))

    non_ring_atoms_set = set(non_ring_atoms)

    # atom-atom edges
    for u, v in zip(*mol.edge):
        if u in non_ring_atoms_set and v in non_ring_atoms_set:
            edges.append((mapping_inverted[u], mapping_inverted[v]))

    # ring-atom edges
    for i in range(len(rings)):
        for j in (ring_N[i] - rings_set[i]) & non_ring_atoms_set:
            edges.append((i, mapping_inverted[j]))

    u = [i for i, _ in edges]
    v = [j for _, j in edges]

    return Graph(
        n=len(mapping), edge=(u, v), nprops={"mapping": mapping}  # type: ignore
    )
