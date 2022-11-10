
"""
Library for processing molecule fragments. 

Create a fragment from a SMILES or a fragment SMILES string:

>>> from xenosite.fragment import Fragment
>>> str(Fragment("ccCC")) # not a valid SMILES
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
>>> F = Fragment("CCCCCC") # hexane as a string
>>> list(F.matches(smiles)) # smiles string (least efficient)
[(0, 1, 2, 3, 4, 5)]

>>> import rdkit
>>> mol = rdkit.Chem.MolFromSmiles(smiles)
>>> list(F.matches(mol))  # RDKit mol
[(0, 1, 2, 3, 4, 5)]

>>> mol_graph = Graph.from_molecule(mol)
>>> list(F.matches(mol, mol_graph)) # RDKit mol and Graph (most efficient)
[(0, 1, 2, 3, 4, 5)]

Matches ensure that the fragment string of matches is the same as
the fragment. This is different than standards SMARTS matching,
and *prevents* rings from matching unclosed ring patterns:

>>> str(Fragment("C1CCCCC1")) # cyclohexane
'C1-C-C-C-C-C-1'

>>> assert(str(Fragment("C1CCCCC1")) != str(F)) # cyclohexane is not hexane
>>> list(F.matches("C1CCCCC1")) # Unlike SMARTS, no match!
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


from .graph import Graph
from .chem import Fragment, MolToSmartsGraph
from ._version import __version__

