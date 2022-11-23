from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX
from hypothesis import strategies as st, given, assume, settings
from rdkit import Chem

def test_propane():
  F = FragmentNetworkX("CCC")

  assert set(F.network) == {"C", "C-C", "C-C-C"}

  assert F.network.nodes["C"]["count"] == 3
  assert F.network.nodes["C-C"]["count"] == 3/2
  assert F.network.nodes["C-C-C"]["count"] == 1


def test_propane_ring():
  F = RingFragmentNetworkX("CCC")

  assert set(F.network) == {"C", "C-C", "C-C-C"}

  assert F.network.nodes["C"]["count"] == 3
  assert F.network.nodes["C-C"]["count"] == 3/2
  assert F.network.nodes["C-C-C"]["count"] == 1


@given(st.integers(max_value=2, min_value=0))
def test_cyclopropane(rid):
  F = FragmentNetworkX("C1CC1",  {rid})

  assert set(F.network) == {"C", "C-C", "C1-C-C-1"}

  assert F.network.nodes["C"]["count"] == 3
  assert F.network.nodes["C-C"]["count"] == 3/2
  assert F.network.nodes["C1-C-C-1"]["count"] == 1

  assert F.network.nodes["C"]["marked_count"] == 1
  assert F.network.nodes["C-C"]["marked_count"] == 3/2
  assert F.network.nodes["C1-C-C-1"]["marked_count"] == 1



@given(st.integers(max_value=2, min_value=0))
def test_cyclopropane_ring(rid):
  F = RingFragmentNetworkX("C1CC1",  {rid})

  assert set(F.network) == {"C1-C-C-1"}

  assert F.network.nodes["C1-C-C-1"]["count"] == 1
  assert F.network.nodes["C1-C-C-1"]["marked_count"] == 1

def test_epoxide():
  F = FragmentNetworkX("C1OC1")
  
  assert set(F.network) == {"C", "C-C", "C1-C-O-1", "C-O", "O"}

  assert F.network.nodes["C"]["count"] == 2
  assert F.network.nodes["C-C"]["count"] == 1
  assert F.network.nodes["C1-C-O-1"]["count"] == 1
  assert F.network.nodes["C-O"]["count"] == 3/2
  assert F.network.nodes["O"]["count"] == 1


def test_epoxide_ring():
  F = RingFragmentNetworkX("C1OC1")
  
  assert set(F.network) == {"C1-C-O-1"}

  assert F.network.nodes["C1-C-O-1"]["count"] == 1



@given(st.integers(max_value=5, min_value=0))
def test_hexane(rid):
  F = FragmentNetworkX("C1CCCCC1", {rid})
  
  assert set(F.network) == {"C", "C-C", "C-C-C", "C-C-C-C", "C-C-C-C-C", "C1-C-C-C-C-C-1"}

  assert F.network.nodes["C"]["count"] == 6
  assert F.network.nodes["C-C"]["count"] == 3
  assert F.network.nodes["C-C-C"]["count"] == 2
  assert F.network.nodes["C-C-C-C"]["count"] == 6/4
  assert F.network.nodes["C-C-C-C-C"]["count"] == 6/5
  assert F.network.nodes["C1-C-C-C-C-C-1"]["count"] == 1

  assert F.network.nodes["C"]["marked_count"] == 1
  assert F.network.nodes["C-C"]["marked_count"] == 3/2
  assert F.network.nodes["C-C-C"]["marked_count"] == 5/3
  assert F.network.nodes["C-C-C-C"]["marked_count"] == 6/4
  assert F.network.nodes["C-C-C-C-C"]["marked_count"] == 6/5
  assert F.network.nodes["C1-C-C-C-C-C-1"]["marked_count"] == 1



@given(st.integers(max_value=5, min_value=0))
def test_hexane_ring(rid):
  F = RingFragmentNetworkX("C1CCCCC1", {rid})
  
  assert set(F.network) == {"C1-C-C-C-C-C-1"}

  assert F.network.nodes["C1-C-C-C-C-C-1"]["count"] == 1
  assert F.network.nodes["C1-C-C-C-C-C-1"]["marked_count"] == 1



TEST_SMILES = ["CC(C)Cc1ccc(C(C)C(=O)O)cc1", "CN(CC=CC#CC(C)(C)C)Cc1cccc2ccccc12", "C1OC1CC1OC1"]

@st.composite
def random_smiles_pair(draw):
  seed = draw(st.integers(min_value=1e2, max_value=1e7)) 
  smiles = draw(st.sampled_from(TEST_SMILES))
  return Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles),2,randomSeed=seed)


@settings(max_examples=20)
@given(random_smiles_pair())  # type: ignore
def test_order_independence_fragnetwork(smiles_pair):
  smiles, rsmiles = smiles_pair
  assume(smiles != rsmiles)

  F = FragmentNetworkX(smiles, max_size=5).network
  rF = FragmentNetworkX(rsmiles, max_size=5).network

  assert set(F) == set(rF)
  assert set(F.edges) == set(rF.edges) 

  for frag in F:
    assert F.nodes[frag]["count"] == rF.nodes[frag]["count"]


@settings(max_examples=20)
@given(random_smiles_pair())  # type: ignore
def test_order_independence_ringfragnetwork(smiles_pair):
  smiles, rsmiles = smiles_pair
  assume(smiles != rsmiles)

  F = RingFragmentNetworkX(smiles, max_size=5).network
  rF = RingFragmentNetworkX(rsmiles, max_size=5).network

  assert set(F) == set(rF)
  assert set(F.edges) == set(rF.edges) 

  for frag in F:
    assert F.nodes[frag]["count"] == rF.nodes[frag]["count"]