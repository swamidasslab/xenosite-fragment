import pytest
from xenosite.fragment import Fragment
from hypothesis import strategies as st, settings, given, assume
from .util import random_smiles_pair
from rdkit import Chem

# FIXED @pytest.mark.xfail(reason="serialization can't use ring ids > 9 yet")
def test_many_rings():
    smi = r"OCC1OC(OC2C(CO)OC(OC3C(CO)OC(OC4C(CO)OC(OC5C(CO)OC(OC6C(CO)OC(OC7C(CO)OC(OC8C(CO)OC(OC9C(CO)OC(OC%10C(CO)OC(OC%11C(CO)OC(OC%12C(CO)OC(O)C(O)C%12O)C(O)C%11O)C(O)C%10O)C(O)C9O)C(O)C8O)C(O)C7O)C(O)C6O)C(O)C5O)C(O)C4O)C(O)C3O)C(O)C2O)C(O)C(O)C1O"
    F = Fragment(smi)


@given(random_smiles_pair())  # type: ignore
def test_order_independence(smiles_pair):
    smiles, rsmiles = smiles_pair
    assert Fragment(smiles).canonical().string == Fragment(rsmiles).canonical().string


def test_boundary_1atom():
    F = Fragment("C")
    assert F.canonical().string == "C"


def test_boundary_0atom():
    F = Fragment("")
    assert F.canonical().string == ""


def test_boundary_two_components():
    with pytest.raises(ValueError):
        F = Fragment("C.C")


def test_eq():
    # Identical fragments with different ordering are equal
    assert Fragment("CCO") == Fragment("OCC")

    # Different fragments are not equal
    assert Fragment("CCO") != Fragment("CCC")

    # Fragment equal to canonical string of itself.
    assert Fragment("OCC") == "C-C-O"

    # Fragment not equal to non-canonical string of itsefl
    assert Fragment("OCC") != "O-C-C"
    assert Fragment("OCC") != "OCC"


def _test_canonization(smi, seed=1):
    m = Chem.MolFromSmarts(smi) #type: ignore
    f = Fragment(m)
    
    for s in Chem.MolToRandomSmilesVect(m, 10, seed): #type: ignore
        assert f == Fragment(s)


def test_case1():
    """Will fail if atom labels not used."""
    _test_canonization(r"Cc1ccccc1O")


def test_case2():
    """Will fail if bond labels not used."""
    _test_canonization(r"C1=CCCCC1")


@pytest.mark.xfail
def test_case_hard1():
    """From fig. 4, DOI: 10.1021/acs.jcim.5b00543
    
    Fails when using morgan labels for canonization.
    """
    _test_canonization(r"C12C3C4C5C6C1C6C(C23)C45")
    
@pytest.mark.xfail
def test_case_hard2():
    """From fig. 4, DOI: 10.1021/acs.jcim.5b00543
    
    Fails when using morgan labels for canonization."""
    _test_canonization(r"C1=CC2=CC=C1\C=C/C1=CC=C(C=C1)\C=C/2")


