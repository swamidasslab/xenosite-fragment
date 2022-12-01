from rdkit import Chem
from hypothesis import strategies as st, given, assume, settings

TEST_SMILES = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "CN(CC=CC#CC(C)(C)C)Cc1cccc2ccccc12",
    "C1OC1CC1OC1",
]


@st.composite
def random_smiles_pair(draw):
    seed = draw(st.integers(min_value=1e2, max_value=1e7))  # type: ignore
    smiles = draw(st.sampled_from(TEST_SMILES))
    out = Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles), 2, randomSeed=seed)  # type: ignore
    assume(out[0] != out[1])
    return out


@st.composite
def random_smiles(draw):
    seed = draw(st.integers(min_value=1e2, max_value=1e7))  # type: ignore
    smiles = draw(st.sampled_from(TEST_SMILES))
    return Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles), 1, randomSeed=seed)[0]  # type: ignore


@st.composite
def random_smiles_som(draw):
    smiles = draw(random_smiles())
    site_smarts = Chem.MolFromSmarts(draw(st.sampled_from(["O", "N"])))  # type: ignore
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    soms = {m[0] for m in mol.GetSubstructMatches(site_smarts)}
    return smiles, soms


@st.composite
def random_smiles_som_pair(draw):
    smiles_pair = draw(random_smiles_pair())

    site_smarts = Chem.MolFromSmarts(draw(st.sampled_from(["O", "N"])))  # type: ignore

    mols = [Chem.MolFromSmiles(s) for s in smiles_pair]  # type: ignore

    soms = [{m[0] for m in mol.GetSubstructMatches(site_smarts)} for mol in mols]

    return smiles_pair, soms
