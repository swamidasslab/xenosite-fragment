import pytest
from xenosite.fragment import Fragment


# FIXED @pytest.mark.xfail(reason="serialization can't use ring ids > 9 yet")
def test_many_rings():
    smi = r"OCC1OC(OC2C(CO)OC(OC3C(CO)OC(OC4C(CO)OC(OC5C(CO)OC(OC6C(CO)OC(OC7C(CO)OC(OC8C(CO)OC(OC9C(CO)OC(OC%10C(CO)OC(OC%11C(CO)OC(OC%12C(CO)OC(O)C(O)C%12O)C(O)C%11O)C(O)C%10O)C(O)C9O)C(O)C8O)C(O)C7O)C(O)C6O)C(O)C5O)C(O)C4O)C(O)C3O)C(O)C2O)C(O)C(O)C1O"
    F = Fragment(smi)
