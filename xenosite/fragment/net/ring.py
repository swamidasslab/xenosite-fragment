
import networkx as nx
from typing import Sequence, Optional
import rdkit.Chem
from .base import FragmentNetworkBase
from ..graph import neighbors, Graph
from .subgraph import subgraph_network_ids
from functools import reduce

class RingFragmentNetwork(FragmentNetworkBase):
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
        edges.extend(
            (i, j) for j in range(i + 1, len(rings)) if ring_N[i] & rings_set[j]
        )

    non_ring_atoms_set = set(non_ring_atoms)

    # atom-atom edges
    for u, v in zip(*mol.edge):  # type: ignore 
        if u in non_ring_atoms_set and v in non_ring_atoms_set:
            edges.append((mapping_inverted[u], mapping_inverted[v]))

    # ring-atom edges
    for i in range(len(rings)):
        edges.extend(
            (i, mapping_inverted[j])
            for j in (ring_N[i] - rings_set[i]) & non_ring_atoms_set
        )

    u = [i for i, _ in edges]
    v = [j for _, j in edges]

    return Graph(
        n=len(mapping), edge=(u, v), nprops={"mapping": mapping}  # type: ignore
    )
