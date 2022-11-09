from rdkit import Chem
from .graph import Graph
from .serialize import Serialized
from typing import Callable, Optional, Union, Generator, Sequence

Mol = Chem.rdchem.Mol
Atom = Chem.rdchem.Atom
Bond = Chem.rdchem.Bond


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
    label_node = lambda x: x.GetSmarts(isomericSmiles=False)
    label_edge = lambda x: _bond_symbol[x.GetBondTypeAsDouble()]

    return MolToGraph(mol, label_node=label_node, label_edge=label_edge)


class Fragment:
    """
    Fragment object to optimze matching of a fragment to multiple molecules.

    Create a fragment from a SMIELS or a fragment SMILES string:

    >>> from xenosite.fragment import Fragment
    >>> str(Fragment("ccCC") # not a valid SMILES
    'c:c-C-C'

    Optionally, create a fragment of a molecule from a string and (optionally) a list of nodes
    in the fragment. If IDs are provided, they MUST select a connected fragment.

    >>> from xenosite.fragment import Fragment
    >>> F = Fragment("CCCCCCOc1ccccc1", [0,1,2,3,4,5])
    >>> str(F)  # hexane
    'C-C-C-C-C-C'

    Get the canonical representation of a fragment:

    >>> Fragment("OC").canonical().string
    'C-O'

    Get the reordering of nodes used to create the canonical
    string representaiton. If remap=True, then the ID are remapped to the input
    representation used to initalize the Fragment.

    >>> Fragment("COC", [1,2]).canonical(remap=True).reordering
    [2, 1]
    >>> Fragment("COC", [1,2]).canonical().reordering
    [1, 0]

    Match fragment to a molecule. By default, the ID
    correspond with fragment IDs. If remap=True, the ID
    corresponds to the input representation when the Fragment
    was initialized.

    >>> smiles = "CCCC1CCOCN1"
    >>> F = Fragment("C1CCCCC1") # hexane as a string
    >>> list(F.match(smiles)) # smiles string (least efficient)
    [(0, 1, 2, 3, 4, 5)]

    >>> mol = rdkit.Chem.MolFromSmiles(smiles)
    >>> list(F.match(mol))  # RDKit mol
    [(0, 1, 2, 3, 4, 5)]

    >>> mol_graph = Graph.from_molecule(mol)
    >>> list(F.match(mol_graph)) # Graph (most efficient)
    [(0, 1, 2, 3, 4, 5)]

    Matches ensure that the fragment string of matches is the same as
    the fragment. This is different than standards SMARTS matching,
    and *prevents* rings from matching unclosed ring patterns:

    >>> str(Fragment("C1CCCCC1")) # cyclohexane
    'C1-C-C-C-C-C-1'

    >>> assert(str(Fragment("C1CCCCC1")) != str(F)) # cyclohexane is not hexane
    >>> F.match("C1CCCCC1") # Unlike SMARTS, no match!
    []

    Efficiently create multiple fragments by reusing a
    precomputed graph:

    >>> from xenosite.fragment import Graph
    >>> import rdkit
    >>>
    >>> mol = rdkit.Chem.MolFromSmiles("c1ccccc1OCCC")
    >>> mol_graph = Graph.from_molecule(mol)
    >>>
    >>> f1 = Fragment(mol_graph, [0])
    >>> f2 = Fragment(mol_graph, [6,5,4])

    Find matches to fragments:

    >>> list(f1.matches(mol))
    [(0,), (1,), (2,), (3,), (4,), (5,)]

    >>> list(f2.matches(mol))
    [(6, 5, 4), (6, 5, 0)]
    """

    def __init__(
        self,
        frag: Union[Graph, Mol, str],
        nidx: Optional[list[int]] = None,
        eidx: Optional[list[int]] = None,
    ):
        """
        Fragment object to optimze matching of a fragment to multiple molecules.

        :param frag: The fragmenet as a string, Rdkit Mol or Graph. Pass as Graph to optimize performance.
        :type frag: Union[Graph, Mol, str]

        :param nidx: If provided, fragment will be initalized to a subgraph of frag, defaults to None
        :type nidx: Optional[list[int]], optional

        :param eidx: If nidx is provided, use these edge ids to determine subgraph, defaults to None
        :type eidx: Optional[list[int]], optional
        """
        if isinstance(frag, str):
            frag = Chem.MolFromSmarts(frag)  # type: ignore
            if not frag:
                raise ValueError("Not a valid Smiles string")

        if isinstance(frag, Mol):
            frag = Graph.from_molecule(frag, smiles=False)

        fragment: Graph = frag.subgraph(nidx, eidx) if nidx else frag  # type: ignore

        self.graph = fragment

        self._nidx = nidx = nidx or list(range(fragment.n))

        self.serial_canonized = fragment.serialize(canonize=True)
        self.serial = fragment.serialize(canonize=False)
        self.graph_mol = Chem.MolFromSmarts(self.serial.string)  # type: ignore

    def __str__(self) -> str:
        """
        The input-ordered SMARTS representation of fragment (not canonical).

        :return: String representation of fragment.
        :rtype: str
        """
        return self.serial.string

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
        remap: bool = False,
    ) -> Generator[list[int], None, None]:
        """
        Generator of all matches between molecule and this fragment. Ordering of IDs in match
        corresponds to the input order of atoms when Fragment was initialized.

        :param mol: Molecule to match with fragment. Pass as rdkit mol for optimal efficiency.
        :type mol: Union[Mol, str]

        :param mol_graph: Pass the graph of the molecule to optimize efficiency, defaults to None
        :type mol_graph: Optional[Graph], optional

        :yield: Matches between fragment and the molecule as integer list of atom indexes.
        :rtype: Generator[list[int], None, None]
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmarts(mol)  # type: ignore

        mol_graph = mol_graph or Graph.from_molecule(mol)

        for match in mol.GetSubstructMatches(self.graph_mol):  # type: ignore
            match_str = mol_graph.subgraph(match).serialize(canonize=True).string
            if match_str == self.serial_canonized.string:
                if remap:
                    yield self.remap_ids(match)
                else:
                    yield match
