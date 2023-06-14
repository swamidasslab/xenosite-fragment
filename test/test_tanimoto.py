import pytest

from xenosite.fragment import smarts_tanimoto
from rdkit import Chem


def test_epoxide():
    """Failure case for early implmentation"""
    m1 = Chem.MolFromSmarts("C1OC1") #type: ignore
    m2 = Chem.MolFromSmarts("C1[O,N,S]C1") #type: ignore
    assert smarts_tanimoto(m1, m2) == 1.0
    assert smarts_tanimoto(m2, m1) == 1.0

def test_phenol():
    """Failure case for early implmentation"""
    m1 = Chem.MolFromSmarts("c1ccccc1O") #type: ignore
    m2 = Chem.MolFromSmarts("c1ccccc1") #type: ignore
    assert smarts_tanimoto(m1, m2) == 6/7
    assert smarts_tanimoto(m2, m1) == 6/7

def test_bond_sensitivity():
    """Ensure that bond type impacts match."""
    m1 = Chem.MolFromSmarts("CC") #type: ignore
    m2 = Chem.MolFromSmarts("C-C") #type: ignore
    assert smarts_tanimoto(m1, m2) == 1.0

    m1 = Chem.MolFromSmarts("C=C") #type: ignore
    m2 = Chem.MolFromSmarts("C-C") #type: ignore
    assert smarts_tanimoto(m1, m2) != 1.0

    m1 = Chem.MolFromSmarts("C=C") #type: ignore
    m2 = Chem.MolFromSmarts("CC") #type: ignore
    assert smarts_tanimoto(m1, m2) != 1.0




