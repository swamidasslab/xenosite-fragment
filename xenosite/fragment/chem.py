from rdkit import Chem
from .graph import Graph
from .serialize import Serialized
from typing import Callable, Optional, Union, Generator, Sequence, NamedTuple
from .ops import to_range 
import networkx as nx
import numpy as np

Mol = Chem.rdchem.Mol
Atom = Chem.rdchem.Atom
Bond = Chem.rdchem.Bond


class FragmentEquivalence(NamedTuple):
    group : Sequence[int]
    num : int

def MolToGraph(
    mol: Mol,
    label_node: Callable[[Atom], str] = lambda x: "*",
    label_edge: Callable[[Bond], str] = lambda x: "",
) -> Graph:
    """
    Convert a molecule to a Graph with configurable labels.

    :param mol: RDK molecule input.
    :type mol: Mol

    :param label_node: Function to label nodes using the atom, defaults to lambdax:"*"
    :type label_node: _type_, optional

    :param label_edge: unction to label nodes using the bond, defaults to lambdax:""
    :type label_edge: _type_, optional

    :return: Labeled graph output
    :rtype: Graph
    """
    n = mol.GetNumAtoms()

    bonds = list(mol.GetBonds())
    e1 = [b.GetBeginAtomIdx() for b in bonds]
    e2 = [b.GetEndAtomIdx() for b in bonds]

    nlabel = [label_node(a) for a in mol.GetAtoms()]
    elabel = [label_edge(b) for b in bonds]

    return Graph(n=n, edge=(e1, e2), nlabel=nlabel, elabel=elabel)  # type: ignore


def MolToSmilesGraph(mol: Mol) -> Graph:
    """
    Convert rdkit molecule to Graph, labeled so as to match smiles output on serialization.

    :param mol: Rdkit molecule input.
    :type mol: Mol

    :return: Smiles labeled graph.
    :rtype: Graph
    """
    label_node = lambda x: x.GetSmarts(isomericSmiles=False)
    label_edge = lambda x: x.GetSmarts()

    return MolToGraph(mol, label_node=label_node, label_edge=label_edge)


_bond_symbol = {1.0: "-", 1.5: ":", 2.0: "=", 3.0: "#"}


def _get_smarts_symbol(atom):
  out = atom.GetSmarts(isomericSmiles=False)

  # Next line is necessary to prevent broken SMARTS like:
  #    n1:c:[nH]:c:c:c:1
  # which has a valence violation and cannot be depicted. 
  # It is unclear which molecules yield fragments with this problem.
  out = "n" if out=="[nH]" else out 

  return out



def MolToSmartsGraph(mol: Mol) -> Graph:
    """
    Convert rdkit molecule to Graph, labeled so as to be a SMARTS string on serialization.
    This is the default labeling of Graphs.from_molecule. Substantial reconfiguration
    is possible here.

    :param mol: Rdkit molecule input.
    :type mol: Mol

    :return: Smarts labeled graph.
    :rtype: Graph
    """
    label_node = _get_smarts_symbol
    label_edge = lambda x: _bond_symbol[x.GetBondTypeAsDouble()]

    return MolToGraph(mol, label_node=label_node, label_edge=label_edge)


class Fragment:
    """
    Fragment object to optimize matching of a fragment to multiple molecules.
    """

    def __init__(
        self,
        frag: Union[Graph, Mol, str, Serialized],
        nidx: Optional[Sequence[int]] = None,
        eidx: Optional[Sequence[int]] = None,
    ):
        """
        Fragment object to optimze matching of a fragment to multiple molecules.

        :param frag: The fragmenet as a string, Rdkit Mol or Graph. Pass as Graph to optimize performance.
        :type frag: Union[Graph, Mol, str, Serialized]

        :param nidx: If provided, fragment will be initalized to a subgraph of frag, defaults to None
        :type nidx: Optional[list[int]], optional

        :param eidx: If nidx is provided, use these edge ids to determine subgraph, defaults to None
        :type eidx: Optional[list[int]], optional
        """

        if isinstance(frag, Serialized):
            frag = frag.string

        if isinstance(frag, str):
            frag = Chem.MolFromSmarts(frag)  # type: ignore
            if not frag:
                raise ValueError("Not a valid Smiles string")

        if isinstance(frag, Mol):
            fragment = Graph.from_molecule(frag, smiles=False)

        if isinstance(frag, Graph):
            fragment = frag

        nidx = list(nidx) if nidx else nidx

        fragment: Graph = fragment.subgraph(nidx, eidx) if nidx else fragment  # type: ignore

        self.graph = fragment

        self._nidx : list[int] = nidx or list(range(fragment.n)) # type: ignore

        self.serial_canonized = fragment.serialize(canonize=True)
        self.serial = fragment.serialize(canonize=False)
        self.smarts_mol = Chem.MolFromSmarts(self.serial.string)  # type: ignore

    def __str__(self) -> str:
        """
        The input-ordered SMARTS representation of fragment (not canonical).

        :return: String representation of fragment.
        :rtype: str
        """
        return self.serial.string
    
    def __repr__(self) -> str:
        """
        Uses the cannonical-ordered SMARTS representation of fragment (not canonical).

        :return: String representation of fragment.
        :rtype: str
        """
        return f'{self.__class__.__name__}({repr(self.serial_canonized.string)})' 

    def equivalence(self)  -> FragmentEquivalence: 
        """
        Determine which atoms in a fragment are topologically equivalent.

        :return: Tuple of numpy array with group assignment, and the integer number of distinct topological groups.
        :rtype: FragmentEquivalence
        """        

        o = self.graph.morgan()
        o = to_range(o)
        return FragmentEquivalence(*o) #type: ignore

    def canonical(self, remap=False) -> Serialized:
        """
        Canonical representation of Fragment, with reordering from input molecule.

        :return: Cannonical string representation of fragment.
        :rtype: Serialized
        """

        if remap:
            return Serialized(
                self.serial_canonized.string,
                self.remap_ids(self.serial_canonized.reordering),
            )

        return self.serial_canonized


    def __eq__(self, other : Union[str, "Fragment"]):
        if isinstance(other, Fragment):
            return other == self.serial_canonized.string
        return self.serial_canonized.string == other
    
    def __hash__(self):
        return self.serial_canonized.string.__hash__()
    
    def remap_ids(self, ids: Sequence[int]) -> list[int]:
        """
        Remap a list of ID in the same space as the Fragment to
        match the input representation's IDs.

        :param ids: Sequence of integer atom ids.
        :type ids: Sequence[int]

        :return: Remaped list of IDS
        :rtype: list[int]
        """
        return [self._nidx[x] for x in ids]

    def matches(
        self,
        mol: Union[Mol, str],
        mol_graph: Optional[Graph] = None,
    ) -> Generator[list[int], None, None]:
        """
        Generator of all matches between molecule and this fragment. Ordering of IDs in match
        corresponds to the Fragment.

        :param mol: Molecule to match with fragment. Pass as rdkit mol for optimal efficiency.
        :type mol: Union[Mol, str]

        :param mol_graph: Pass the graph of the molecule to optimize efficiency, defaults to None
        :type mol_graph: Optional[Graph], optional

        :yield: Matches between fragment and the molecule as integer list of atom indexes.
        :rtype: Generator[list[int], None, None]
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmarts(mol)  # type: ignore

        assert isinstance(mol, Mol)

        mol_graph = mol_graph or Graph.from_molecule(mol)

        for match in mol.GetSubstructMatches(self.smarts_mol):  # type: ignore
            
            #TODO: optimize by checking atom and bond number equality before canonnization
            match_str = mol_graph.subgraph(match).serialize(canonize=True).string
            if match_str == self.serial_canonized.string:
                yield match




#TODO: implment a version/option of this function uses the edge count instead of just the atoms.
#TODO: add convenience conversion from Fragment or str to rdkit.Mol
def smarts_tanimoto(m1 : Chem.Mol, m2 : Chem.Mol, cutoff : float = 0, allow_disconnected : bool = True) -> float: #type: ignore
    """Tanimoto similarity of two SMARTS rdkit mols. Overlap is not mesured using a fingerprint, but 
    by computing the max-common-structure overlap (in atoms) of the two, so this can be slow.

    For example, these two molecules/SMARTS overlap in 3 out of 4 molecules, yielding a tanimoto of 3/4. 

    >>> from rdkit import Chem
    >>> m1 = Chem.MolFromSmarts("CCC")
    >>> m2 = Chem.MolFromSmarts("CCCC")
    >>> smarts_tanimoto(m1, m2)
    0.75

    These two molecules yield the same overlap because the SMARTS string allows for either
    a C or an N in the third position.

    >>> m1 = Chem.MolFromSmarts("CCN")
    >>> m2 = Chem.MolFromSmarts("CC[C,N]C")
    >>> smarts_tanimoto(m1, m2)
    0.75

    Cutoff is the minimum similarity we care about, which can speed computation by avoiding the MCS
    computation in some cases. When this threshold can't be met, the similarity is reported as zero.

    >>> smarts_tanimoto(m1, m2, cutoff=0.8)
    0.0

    Note that the function will still report similarity, if it exists and even if it is below cutoff, when
    the MSC step is run. In the following case, 3 out of 5 atoms match, but based on size of the framents 
    alone 4 out of 4 atoms could have matched. So the MCS was run, and the 0.6 similarity was reported even
    though it was below the cutoff.

    >>> m1 = Chem.MolFromSmarts("OCCN")
    >>> m2 = Chem.MolFromSmarts("CC[C,N]C")
    >>> smarts_tanimoto(m1, m2, cutoff=0.8)
    0.6

    """
    A = m1.GetNumAtoms()
    B = m2.GetNumAtoms()
    
    if min(A,B) / max(A,B) < cutoff: return 0.0
    AnB = _maximum_common_subgraph(m1, m2, allow_disconnected)

    return AnB / (A + B - AnB) 


#TODO: implment a version/option of this function uses the edge count instead of just the atoms.
#TODO: add convenience conversion from Fragment or str to rdkit.Mol
def smarts_edit_distance(m1 : Chem.Mol, m2 : Chem.Mol, cutoff : float = 10, allow_disconnected : bool = True) -> float: #type: ignore
    """Vertex edit distance between two SMARTS rdkit mols. Overlap is not mesured using a fingerprint, but 
    by computing the max-common-structure overlap (in atoms) of the two, so this can be slow.

    For example, these two molecules/SMARTS overlap in 3 out of 4 molecules, yielding an edit of 1. 

    >>> from rdkit import Chem
    >>> m1 = Chem.MolFromSmarts("CCC")
    >>> m2 = Chem.MolFromSmarts("CCCC")
    >>> smarts_edit_distance(m1, m2)
    1

    These two molecules yield the same edit distance because the SMARTS string allows for either
    a C or an N in the third position.

    >>> m1 = Chem.MolFromSmarts("CCN")
    >>> m2 = Chem.MolFromSmarts("CC[C,N]C")
    >>> smarts_edit_distance(m1, m2)
    1

    Cutoff is the maximum edit-distance we care about, which can speed computation by avoiding the MCS
    computation in some cases. When this threshold can't be met, the edit is reported as the cutoff value.

    >>> smarts_edit_distance(m1, m2, cutoff=0.5)
    0.5

    Note that the function will still report edit distance, if it exists and even if it is above cutoff, when
    the MSC step is run. In the following case, 3 out of 5 atoms match, but based on size of the framents 
    alone 4 out of 4 atoms could have matched. So the MCS was run, and the 2 edit distance was reported even
    though it was above the cutoff.

    >>> m1 = Chem.MolFromSmarts("OCCN")
    >>> m2 = Chem.MolFromSmarts("CC[C,N]C")
    >>> smarts_edit_distance(m1, m2, cutoff=0.5)
    2

    """
    A = m1.GetNumAtoms()
    B = m2.GetNumAtoms()
    
    if max(A,B) - min(A,B) > cutoff: return cutoff
    AnB = _maximum_common_subgraph(m1, m2, allow_disconnected)

    return A + B - 2 * AnB

def _modular_product_graph(m1 : Chem.Mol, m2: Chem.Mol) -> nx.Graph: #type: ignore
    """This function converts two molecules into a modular product graph.
    Cliques in a modular product graph correspond to the maximum-common substructure.
    Uses rdkit mols to make use of SMARTS matching logic.
    
    https://en.wikipedia.org/wiki/Modular_product_of_graphs
    """
    g = nx.Graph()

    #nodes are tuple of atomids (i,j), one from each input, representing a pairing 
    for a1 in m1.GetAtoms():
        for a2 in m2.GetAtoms():
            if a1.Match(a2) or a2.Match(a1):
                g.add_node((a1.GetIdx(), a2.GetIdx()))

    # edges indicate connectivity between parring is compatible
    for n1 in g.nodes():
        for n2 in g.nodes():
            # no edge if pairs refer to same node (in either graph)
            if n1[0] == n2[0]: continue
            if n1[1] == n2[1]: continue
            
            b1 = m1.GetBondBetweenAtoms(n1[0],n2[0])
            b2 = m2.GetBondBetweenAtoms(n1[1],n2[1])
            
            # no edge if one pair connected and the other isn't
            if b1 and not b2: continue
            if not b1 and b2: continue
            
            # if both pairs aren't connected, add edge
            if not b1 and not b2:
                g.add_edge(n1, n2)
                continue
            
            # if both pairs are connected, and edge matches, add edge
            if b1.Match(b2) or b2.Match(b1):
                g.add_edge(n1, n2)
                continue

            # otherwise, no edge

    return g


def _mol_graph(m) -> nx.Graph:
    # convert to a graph ignoring all labels
    g = nx.Graph()
    for b in m.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        g.add_edge(i,j)
    return g


def _maximum_common_subgraph(m1 : Chem.Mol, m2 : Chem.Mol, allow_disconnected : bool = True) -> int: #type: ignore
    """
    Return the size (in atoms) of the maximum match between
    two RDkit molecules, using smarts matching rules.

    :param m1: molecule 2
    :type m1: Chem.Mol
    :param m2: molecule 1
    :type m2: Chem.Mol
    :return: size of maximum match in atoms
    :rtype: int
    """

    g = _modular_product_graph(m1, m2)

    # need to consider all cliques
    # TODO: benchmark against igraph library
    cliques = nx.clique.find_cliques(g)

    if allow_disconnected:  
      max_clique =  max(cliques, key=len)

    else: #TODO: Optimize this branch
      # we are only interested in connected components of the match
      # so we get the sugraph corresponding to the clique in g1
      # alternatively, g2 could be used
      g1 = _mol_graph(m1)
          
      to_subgraph = lambda clique : g1.subgraph([c[0] for c in clique])
      to_components = lambda clique : nx.connected_components(to_subgraph(clique))
      max_connected_clique = lambda clique: max(to_components(clique), key=len)
      max_clique = max((max_connected_clique(clique) for clique in cliques), key=len)
    
    return len(max_clique)
      
