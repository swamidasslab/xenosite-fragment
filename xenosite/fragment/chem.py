from rdkit import Chem
from .graph import Graph
from .serialize import Serialized
from typing import Callable, Optional, Union, Generator, Sequence, NamedTuple
from .morgan import to_range 

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

        self._nidx = nidx = nidx or list(range(fragment.n))

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

        m = self.graph.morgan()
        i,ni = to_range(m)
        return FragmentEquivalence(i, ni) #type: ignore

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
