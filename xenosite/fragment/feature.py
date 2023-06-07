import pandas as pd
import numpy as np
from typing import Sequence, NamedTuple
from xenosite.fragment.net.base import FragmentNetworkBase
from xenosite.fragment import Fragment


class FragmentVectors(NamedTuple):
    frag : np.ndarray
    site : np.ndarray
    site_len : np.ndarray
    mol_frag_len : np.ndarray
    mol_site_len : np.ndarray

    def to_ragged(self):
        import tensorflow as tf

        lengths = (self.mol_site_len, self.site_len)
        values = tf.transpose(tf.constant([self.site, self.frag], dtype=tf.uint32))
        r = tf.RaggedTensor.from_nested_row_lengths(values, lengths)
        return r[0]

class FragmentVectorSizes(NamedTuple):
    frag : int
    frag_site : int
    frag_len : np.ndarray



class FragmentVectorize:
    """Class that vectorizes all sites in a molecule with a fragment representation.
    Each site of a molecule is represented as a bag of positions in fragments, all
    encoded as integers.
    """
    def __init__(self, fragments : Sequence[str]):
        """
        Initialize with list of fragments (represented as strings).
        """
        self.index = pd.Index(fragments).unique()

        frags =  [Fragment(f) for f in  self.index]
        equiv =  [f.equivalence() for f in frags]

        # list of maps from atom in fragment to equivalence group, one for each fragment.
        self.map : list[np.ndarray] = [e[0] for e in equiv]  #type: ignore

        # the number of equivalence groups in each fragment
        self.nequiv : np.ndarray = np.array([e[1] for e in equiv])

        # the index where each fragment begins in the embedding vector
        self.frag2start = np.cumsum(self.nequiv) - self.nequiv

    def get_loc(self, frag, idx=None, equiv_group=None) -> np.ndarray:  #type: ignore
        i = self.index.get_loc(frag)
        start = self.frag2start[i]

        if idx is None and equiv_group is None:
            return np.arange(self.nequiv[i]) + start
        
        if not idx is None:
            return np.array(start + self.map[i][idx])

        if not equiv_group is None:
            if equiv_group >= self.nequiv[i]: raise KeyError
            return np.array(start + equiv_group)    
        
    def sizes(self) -> FragmentVectorSizes:
        """
        Gives size information for embedding using this vecdtorization.

        :return: Returns the number of fragments and the number of fragment-equivalence groups, which
        can be used to paramterized embedding layers with these lenghts corresponding to IDs output
        by this vecdtorization.
        :rtype: FragmentVectorSizes
        """
        return FragmentVectorSizes(len(self.index), np.sum(self.nequiv), self.nequiv)

    def features(self, network : FragmentNetworkBase) -> FragmentVectors:
        """
        Convert a fragment network (computed from a *single* molecule) into vector form.
        Fragments not in the intialization are skipped.

        :param network: Network showing all mined fragments in single molecule
        :type network: FragmentNetworkBase
        :return: Vectors encoding the molecule in segemented vectors. 
          - "frag" is the fragment ID
          - "site" is the group ID, which correspondes to a topological equivalence group within a fragment
          - "site_len" is the length of each segment, where each segement corresponds to an atom of the input molecule.
          - "mol_len" is the length of each segment, where each segement corresponds to a single molecule
        :rtype: FragmentVectors
        """
        num_atoms = network._mol.GetNumAtoms()

        # initialize each site 
        v = [set() for _ in range(num_atoms)]

        # records how each fragment matches to the molecule
        frag2id = network._frag2id

        for f in frag2id: # for each fragment in the network
            matches = frag2id[f]  # get the matches to the original molecule
            try:  # get the integer id of the fragment, or skip if not considered.
              frag_id = self.index.get_loc(f)
            except KeyError:
              continue
            frag_start = self.get_loc(f, 0) # Get the starting frag-site id 
            frag_map = self.map[frag_id] # get the map of this fragment

            for match in matches: # for each map
                #iterate over the match positions and corrsponding frag_map to get the equivalence group.
                for s, e in zip(match, frag_map): #type: ignore
                    # to this site id, add a tuple of frag_id and equivalence group corresponding to this 
                    v[s].add((frag_id, frag_start + e))

        site_len = np.array([len(x) for x in v]) # frags per site
        mol_frag_len = np.array([site_len.sum()]) # frags per mol
        mol_site_len = np.array([len(site_len)]) # sites (atoms) per mol

        # convert sets to sorted lists to ensure stable output
        v = [list(sorted(x)) for x in v]

        # stack and transpose organize for output
        v = [np.array(x, np.uint32).reshape((-1, 2)) for x in v]
        vv = np.vstack(v).T  #type: ignore
        return FragmentVectors(vv[0], vv[1], site_len, mol_frag_len, mol_site_len)
    
    def embedding_frame(self, frag_embedding, site_embedding):
        import pandas as pd

        sections = np.cumsum(self.nequiv)[:-1]
        se = np.split(site_embedding, sections)
        ae = [np.take(e,m) for m,e in zip(self.map, se)]
        frag_embedding = np.array(frag_embedding)

        return pd.DataFrame({"frag_value": frag_embedding, "atom_values": ae, "site_values": se}, index=self.index)

    def resolve_features(self, vectors : FragmentVectors) -> tuple[int, str, int]:
        frags = self.index[vectors.frag]
        
        fvs = self.sizes()
        frag_start =  np.cumsum(fvs.frag_len) - fvs.frag_len
        site = vectors.site - np.take(frag_start, vectors.frag)

        mols = np.repeat(np.arange(len(vectors.mol_frag_len)), vectors.mol_frag_len)  #type: ignore

        return list(zip(mols, frags, site)) #type: ignore
    
    @staticmethod
    def batch(*examples : tuple[FragmentVectors]):
        batched = [np.concatenate(x) for x in zip(*examples)]  #type: ignore
        return FragmentVectors(*batched)
        