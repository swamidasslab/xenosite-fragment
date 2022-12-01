import pytest
from xenosite.fragment import Fragment
from hypothesis import given, settings
from .util import random_smiles_som_pair

from xenosite.fragment.remap import remap_ids
from rdkit import Chem


@settings(deadline=None)
@given(random_smiles_som_pair())
def test_fragment_som_reorder(data):
    (smi1, smi2), (som1, som2) = data

    csmi1, remap1 = Fragment(smi1).canonical()
    csmi2, remap2 = Fragment(smi2).canonical()

    assert csmi1 == csmi2

    r_som1 = set(remap_ids(som1, remap1))
    r_som2 = set(remap_ids(som2, remap2))

    assert r_som1 == r_som2


@settings(deadline=None)
@given(random_smiles_som_pair())
def test_rdkit_som_reorder(data):
    (smi1, smi2), (som1, som2) = data

    m1 = Chem.MolFromSmiles(smi1)  # type: ignore
    csmi1 = Chem.MolToSmiles(m1)  # type: ignore
    remap1 = list(m1.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])

    m2 = Chem.MolFromSmiles(smi2)  # type: ignore
    csmi2 = Chem.MolToSmiles(m2)  # type: ignore
    remap2 = list(m2.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])

    assert csmi1 == csmi2

    r_som1 = set(remap_ids(som1, remap1))
    r_som2 = set(remap_ids(som2, remap2))

    assert r_som1 == r_som2
