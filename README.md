# Xenosite Fragment

A library for processing molecule fragments. 

Install from pypi:

```
pip install xenosite-fragment
```

Create a fragment from a SMILES or a fragment SMILES string:

```
>>> str(Fragment("CCCC")) # Valid smiles
'C-C-C-C'

>>> str(Fragment("ccCC")) # not a valid SMILES
'c:c-C-C'
```

Optionally, create a fragment of a molecule from a string and (optionally) a list of nodes
in the fragment. 

```
>>> F = Fragment("CCCCCCOc1ccccc1", [0,1,2,3,4,5])
>>> str(F)  # hexane
'C-C-C-C-C-C'
```

If IDs are provided, they MUST select a connected fragment.

```
>>> F = Fragment("CCCCCCOc1ccccc1", [0,10]) 
Traceback (most recent call last):
  ...
ValueError: Multiple components in graph are not allowed.
```

Get the canonical representation of a fragment:

```
>>> Fragment("O-C").canonical().string
'C-O'
>>> Fragment("OC").canonical().string
'C-O'
>>> Fragment("CO").canonical().string
'C-O'
```

Get the reordering of nodes used to create the canonical
string representation. If remap=True, then the ID are remapped to the input
representation used to initalize the Fragment.

```
>>> Fragment("COC", [1,2]).canonical(remap=True).reordering
[2, 1]
>>> Fragment("COC", [1,2]).canonical().reordering
[1, 0]
```

Match fragment to a molecule. By default, the ID
correspond with fragment IDs. If remap=True, the ID
corresponds to the input representation when the Fragment
was initialized.

```
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
```

Matches ensure that the fragment string of matches is the same as
the fragment. This is different than standards SMARTS matching,
and *prevents* rings from matching unclosed ring patterns:

```
>>> str(Fragment("C1CCCCC1")) # cyclohexane
'C1-C-C-C-C-C-1'

>>> assert(str(Fragment("C1CCCCC1")) != str(F)) # cyclohexane is not hexane
>>> list(F.matches("C1CCCCC1")) # Unlike SMARTS, no match!
[]
```

Efficiently create multiple fragments by reusing a
precomputed graph:

```
>>> import rdkit
>>>
>>> mol = rdkit.Chem.MolFromSmiles("c1ccccc1OCCC")
>>> mol_graph = Graph.from_molecule(mol)
>>>
>>> f1 = Fragment(mol_graph, [0])
>>> f2 = Fragment(mol_graph, [6,5,4])
```

Find matches to fragments:

```
>>> list(f1.matches(mol))
[(0,), (1,), (2,), (3,), (4,), (5,)]

>>> list(f2.matches(mol))
[(6, 5, 4), (6, 5, 0)]
```

Fragments know how to report if they are canonically the same as each other or strings.

```
>>> Fragment("CCO") == Fragment("OCC")
True
>>> Fragment("CCO") == "C-C-O"
True
```

Note, however, that strings are not converted to canonical form. Therefore,

```
>>> Fragment("CCO") == "CCO"
False
```

Enumerate and compute statistics on all the subgraphs in a molecule:

```
>>> from xenosite.fragment.net import SubGraphFragmentNetwork
>>> N = SubGraphFragmentNetwork("CC1COC1")
>>> fragments = N.to_pandas()
>>> list(fragments.index)
['C-C', 'C', 'C-O-C', 'C-O', 'O', 'C-C1-C-O-C-1', 'C1-C-O-C-1', 'C-C-C-O', 'C-C(-C)-C', 'C-C-O', 'C-C-C']
>>> fragments["size"].to_numpy()
array([2, 1, 3, 2, 1, 5, 4, 4, 4, 3, 3])
```

Better fragments can be enumerated by collapsing all atoms in a ring into a single node
during subgraph enumeration. 

```
>>> from xenosite.fragment.net import RingFragmentNetwork
>>> N = RingFragmentNetwork("CC1COC1")
>>> fragments = N.to_pandas()
>>> list(fragments.index)
['C-C1-C-O-C-1', 'C', 'C1-C-O-C-1']
>>> fragments["size"].to_numpy()
array([5, 1, 4])
```