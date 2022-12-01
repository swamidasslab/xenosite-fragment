from typing import Sequence, Iterable, Generator, Union
from rdkit import Chem
import ast
from .serialize import Serialized


def rdkit_serialize(
    mol: Union[Chem.Mol, str], canonical=True, isomeric: bool = False  # type: ignore
) -> Serialized:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)  # type: ignore

    smi = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)  # type: ignore

    # slow version
    # reorder = list(mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])  # type: ignore
    # fast version
    reorder = ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder"))  # type: ignore

    return Serialized(string=smi, reordering=reorder)


def remap_ids(
    ids: Iterable[int], reordering: Sequence[int], is_inverse=False
) -> Generator[int, None, None]:
    """
    Remap ids given a reordering.

    Inputs:
      ids: Iterable of int IDs to remap.
      reordering: The new order of the IDs.
      is_inverse: Flag that to avoid inverting reordering if it is already reordered.

    Output:
      Generator of ints reordered correctly.

    This can be used to remap ids after canonization by rdkit or Fragment.
    For example, consider this molecule's canonization.

    >>> smi, reordering = rdkit_serialize("OCCCCCC")
    >>> smi
    'CCCCCCO'

    The original atoms appear in the new string in this order.

    >>> reordering
    [6, 5, 4, 3, 2, 1, 0]

    The first two atoms in the original molecule can be remapped
    to the new is using this reorderingarray.

    >>> ids = [0,1]
    >>> tuple(remap_ids(ids, reordering))
    (6, 5)

    A more complicated example works using the same API.

    >>> smi, reordering = rdkit_serialize('O1C(CC2CO2)C1')
    >>> smi
    'C1OC1CC1CO1'
    >>> reordering
    [4, 5, 3, 2, 1, 6, 0]

    The ids here correspond to the oxygen atoms (O).

    >>> ids = [0, 5]
    >>> tuple(remap_ids(ids, reordering))
    (6, 1)
    """

    remap = reordering if is_inverse else {r: n for n, r in enumerate(reordering)}

    for i in ids:
        yield remap[i]


def serialize_and_remap(
    mol: Union[Chem.Mol, str],  # type: ignore
    ids: Iterable[int],
    canonical=True,
    isomeric: bool = False,
) -> tuple[Serialized, tuple[int]]:

    """
    Serialize a molecule, remapping ids associated with the molecule
    to the new ordering.

    Inputs:
      mol : String or rdkit Molecule.
      ids : Iterable of integer ids to remap
      canonical : Boolean flag for canonization (default: True)
      isomeric: : Bolean flag for isomeric smiles (default: False)

    Outputs:
      Seralized molecule (tuple of string and reordering)
      Tuple of remapped_ids

    >>> (smi, reordering), ids = serialize_and_remap("OCCCCCC", [0,1])
    >>> smi
    'CCCCCCO'
    >>> reordering
    [6, 5, 4, 3, 2, 1, 0]
    >>> ids
    (6, 5)

    >>> (smi, reordering), ids  = serialize_and_remap('O1C(CC2CO2)C1', [0, 5])
    >>> smi
    'C1OC1CC1CO1'
    >>> reordering
    [4, 5, 3, 2, 1, 6, 0]
    >>> ids
    (6, 1)
    """
    serial = rdkit_serialize(mol, canonical, isomeric)
    remapped_ids = tuple(remap_ids(ids, serial.reordering))
    return serial, remapped_ids
