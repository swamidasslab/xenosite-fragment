import contextlib
import numpy as np
import pandas as pd
from typing import Optional, Union, Sequence, Generator
from ..graph import Graph
from ..chem import Fragment
from collections import defaultdict

import gzip
import pickle
import networkx as nx
import rdkit
import logging
from ..stats import FragmentStatistics
import rdkit
from xenosite.fragment.chem import MolToSmartsGraph

logger = logging.getLogger(__name__)

class FragmentNetworkBase:
    max_size: int = 8
    _version: int = 6

    def __init__(
        self,
        smiles: Optional[str] = None,
        marked: Optional[set[int]] = None,
        max_size: Optional[int] = None,
        include_mol_ref: bool = True,
        fragment_input: bool = False,
    ):
        self.version: int = self._version
        self.stats = FragmentStatistics()

        if max_size:
            self.max_size = max_size

        if not smiles:
            self.network = nx.DiGraph()
            self.molrefs = nx.DiGraph()
            return
        
        if fragment_input:
            rdmol: rdkit.Chem.Mol = rdkit.Chem.MolFromSmarts(smiles)
        else:
            rdmol: rdkit.Chem.Mol = rdkit.Chem.MolFromSmiles(smiles)  # type: ignore

        assert rdmol, f"Not a valid SMILES or SMARTS: ${smiles}" 
        self._mol: rdkit.Chem.Mol  = rdmol # type: ignore

        mol: Graph = Fragment(rdmol).graph
        marked = marked or set()

        network = nx.DiGraph()

        id_network = self._subgraph_network_ids(rdmol, mol)
        frag2reordering = defaultdict(lambda: [])
        # frag2ids = defaultdict(lambda: [])

        frag2fragment = {}

        for ids in id_network.nodes:
            full_ids = self._remap_ids(ids, id_network)
            fragment = Fragment(mol, full_ids)
            serial = fragment.canonical(remap=True)
            frag = serial.string  # type: ignore
            id_network.nodes[ids]["frag"] = frag

            frag2reordering[frag].append(serial.reordering)
            # frag2ids[frag].append(ids)

            frag2fragment[frag] = fragment

        self._frag2id = frag2reordering

        for frag, ids in frag2reordering.items():
            fragment = frag2fragment[frag]
            self.stats.add(frag, fragment, ids, marked, mol.n)
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
        return df

    def _remap_ids(self, ids: Sequence[int], id_network: nx.DiGraph) -> Sequence[int]:
        raise NotImplementedError

    def _subgraph_network_ids(self, rdmol: rdkit.Chem.Mol, mol: Graph) -> nx.DiGraph:  # type: ignore
        raise NotImplementedError

    def add(self, smiles: str, **kwargs):
      F = self.__class__(smiles, max_size=self.max_size, **kwargs)
      self.update(F)

    def copy_stats(self, other: "FragmentNetworkBase") -> "FragmentNetworkBase":
      #TODO change to shallow copy of self to avoid clobbering
      self.stats = self.stats.copy_from(other.stats)
      return self

    def molecule_shading(self,
      mol : str,
      beta_prior : float = 0.0,
      hold_out : bool = False,
      marked: Optional[set[int]] = None
    ) -> np.ndarray:
    
        assert(type(self).version == self.version)
        N = type(self)(mol, max_size=self.max_size, marked=marked)
        
        frag2ids = N._frag2id

        rdmol = rdkit.Chem.MolFromSmiles(mol)
        mol_size = rdmol.GetNumAtoms()
        shade = np.zeros(mol_size)
        
        for frag in N.network.nodes:
            if not isinstance(frag, str): continue
            if frag not in self.stats._lookup: continue
            
            ids = frag2ids[frag]
            n = self.stats._lookup[frag]
            marked_ids = self.stats._stats["marked_ids"][n]
            n_mol = self.stats._stats["n_mol"][n]

            if hold_out:
                n = N.stats._lookup[frag]
                marked_ids = marked_ids - N.stats._stats["marked_ids"][n]
                n_mol -= 1
                if not n_mol: continue

            frag_shade = marked_ids / (n_mol + beta_prior)

            for match in ids:
                shade[match] = np.where(shade[match] > frag_shade, shade[match], frag_shade)

        return shade

    # Get fragments of specific molecule with specific atom id
    def fragments_by_id(self, mol : Union[str, "FragmentNetworkBase"], atom_id : Union[int, Sequence[int]] ) -> Generator[tuple[str,np.ndarray], None, None]:
    
      if isinstance(mol, str):
        # Create new ring fragment network from specific molecule
        n_mol = type(self)(mol, max_size=8, include_mol_ref=True)
      elif isinstance(mol, FragmentNetworkBase):
        assert type(mol) == type(self), f"FragmentNetwork class mismatch: ${type(self)} ${type(mol)}"       
        n_mol = mol

      # Get fragments with specific atom id only
      frag2ids = n_mol._frag2id
        
      if isinstance(atom_id, int): atom_id = [atom_id]
      
      atom_set = set(atom_id)

      for frag in n_mol.network.nodes:
        if not isinstance(frag, str): continue
        if frag not in self.stats._lookup: continue

        ids = frag2ids[frag]
        for i in ids:
          if atom_set & set(i):
            # Save fragments and stats
            yield frag, i

    def save(self, filename: str):
        with gzip.GzipFile(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "FragmentNetworkBase":
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

    def update(self, other: "FragmentNetworkBase"):
        self.stats.update(other.stats)

        with contextlib.suppress(Exception):
          del self._frag2id

        for frag in other.network.nodes:
            if frag not in self.network.nodes:
                self.network.add_node(frag)
            for child in other.network[frag]:
                self.network.add_edge(frag, child)

