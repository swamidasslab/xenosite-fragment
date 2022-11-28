import contextlib
import numpy as np
import pandas as pd
from typing import Optional, Union, Sequence, Generator
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
from .stats import FragmentStatistics
import rdkit
from xenosite.fragment.chem import MolToSmartsGraph

logger = logging.getLogger(__name__)


class FragmentNetwork:
    max_size: int = 10
    _version: int = 3

    def __init__(
        self,
        smiles: Optional[str] = None,
        marked: Optional[set[int]] = None,
        max_size: Optional[int] = None,
        include_mol_ref : bool = False,
    ):
        self.version: int = self._version
        self.stats = FragmentStatistics()

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

        self._frag2id = frag2reordering

        for frag, ids in frag2reordering.items():
            self.stats.add(frag, ids, marked, mol.n)
            network.add_node(frag)

        for u, v in id_network.edges:
            fu = id_network.nodes[u]["frag"]
            fv = id_network.nodes[v]["frag"]
            if fu != fv:
                network.add_edge(fu, fv)


        if include_mol_ref:
          top = [node for node, degree in network.in_degree() if degree == 0]  # type: ignore
          mol_key = (smiles, "ref")  # type: ignore
          nx.add_star(network, [mol_key] + top)

        self.network = network

    def contains_fragment(self, frag: str) -> Generator[str, None, None]:
        with contextlib.suppress(Exception):
            frag = Fragment(frag).canonical().string
            
        for n in nx.dfs_predecessors(self.network.reverse(False), frag):
            if isinstance(n, tuple):
                yield n[0]

    def to_pandas(self) -> pd.DataFrame:
        df = self.stats.to_pandas()
        df["size"] =(df["n_cover"] / df["count"]).astype(int)
        return df

    def _remap_ids(self, ids: Sequence[int], id_network: nx.DiGraph) -> Sequence[int]:
        return ids

    def _subgraph_network_ids(self, rdmol: rdkit.Chem.Mol, mol: Graph) -> nx.DiGraph:  # type: ignore
        return subgraph_network_ids(mol, self.max_size)

    def add(self, smiles: str) -> None:
      raise NotImplemented

    def copy_stats(self, other: "FragmentNetwork") -> "FragmentNetwork":
      #TODO change to shallow copy of self to avoid clobbering
      self.stats = self.stats.copy_from(other.stats) 
      return self

    def molecule_shading(self, mol : str) -> np.ndarray:
      raise NotImplemented
      
      N = type(self)(mol, max_size=self.max_size) # make 
      frag2ids = N._frag2id

      shade = zeros(12) 
      for frag in N.network.nodes:
        if not isinstance(frag, str): continue
        if frag not in self.stats._lookup: continue

        ids = frag2ids[frag]
        n = self.stats._lookup[frag]
        frag_shade = self.stats._stats["marked_ids"][n] / self.stats._stats["n_mol"]
        for match in ids:
          shade[match] = np.where(shade[match]> frag_shade, shade[match], frag_shade)

      



# How a molecule is shaded
# def display(info):
#   frag = info.name
#   #frag = re.sub(":", "", frag)
#   # frag = re.sub(r"\[nH\]", "n", frag)
#   try:
#     m = Chem.MolFromSmarts(frag)
#     assert m, "Fragment did not produce mol: " + frag

#     x = xenopict.Xenopict(m)
#     x.shade(np.minimum(info["marked_ids"] / info["n_mol"], 1))
#     return x
#   except Exception as e:
#     return e



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
        self.stats.update(other.stats)

        with contextlib.suppress(Exception):
          del self._frag2id 

        for frag in other.network.nodes:
            if frag not in self.network.nodes:
                self.network.add_node(frag)
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
    mol = mol or MolToSmartsGraph(rdmol)  # type: ignore
    assert mol

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
        edges.extend((i, j) for j in range(i + 1, len(rings)) if ring_N[i] & rings_set[j])

    non_ring_atoms_set = set(non_ring_atoms)

    # atom-atom edges
    for u, v in zip(*mol.edge):
        if u in non_ring_atoms_set and v in non_ring_atoms_set:
            edges.append((mapping_inverted[u], mapping_inverted[v]))

    # ring-atom edges
    for i in range(len(rings)):
        edges.extend((i, mapping_inverted[j]) for j in (ring_N[i] - rings_set[i]) & non_ring_atoms_set)

    u = [i for i, _ in edges]
    v = [j for _, j in edges]

    return Graph(
        n=len(mapping), edge=(u, v), nprops={"mapping": mapping}  # type: ignore
    )
