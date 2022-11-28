from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX
from hypothesis import strategies as st, given, assume, settings
from .util import random_smiles_pair, random_smiles
import networkx as nx
import pytest
import pandas as pd
import numpy as np

def fragment_data(N: FragmentNetworkX) -> pd.DataFrame:
    return N.to_pandas()


def fragment_view(N: FragmentNetworkX) -> nx.DiGraph:
    return nx.subgraph_view(N.network, filter_node=lambda x: isinstance(x, str))  # type: ignore


@pytest.mark.parametrize(
    "smiles, frags, frag_data",
    ids=["propane", "cyclopropane", "epoxide", "hexane"],
    argvalues=[
        (
            "CCC",
            {"C", "C-C", "C-C-C"},
            {
                "C": {"count": 3},
                "C-C": {"count": 3 / 2},
                "C-C-C": {"count": 1},
            },
        ),
        (
            "C1CC1",
            {"C", "C-C", "C1-C-C-1"},
            {
                "C": {
                    "count": 3,
                    "marked_count": 1,
                },
                "C-C": {
                    "count": 3 / 2,
                    "marked_count": 3 / 2,
                },
                "C1-C-C-1": {
                    "count": 1,
                    "marked_count": 1,
                },
            },
        ),
        (
            "C1OC1",
            {"C", "C-C", "C1-C-O-1", "C-O", "O"},
            {
                "C": {"count": 2},
                "C-C": {"count": 1},
                "C1-C-O-1": {"count": 1},
                "C-O": {"count": 3 / 2},
                "O": {"count": 1},
            },
        ),
        (
            "C1CCCCC1",
            {"C", "C-C", "C-C-C", "C-C-C-C", "C-C-C-C-C", "C1-C-C-C-C-C-1"},
            {
                "C": {
                    "count": 6,
                    "marked_count": 1,
                },
                "C-C": {
                    "count": 3,
                    "marked_count": 3 / 2,
                },
                "C-C-C": {
                    "count": 2,
                    "marked_count": 5 / 3,
                },
                "C-C-C-C": {
                    "count": 6 / 4,
                    "marked_count": 6 / 4,
                },
                "C-C-C-C-C": {
                    "count": 6 / 5,
                    "marked_count": 6 / 5,
                },
                "C1-C-C-C-C-C-1": {
                    "count": 1,
                    "marked_count": 1,
                },
            },
        ),
    ],
)
def test_net_ex(
    smiles: str,
    frags: set[str],
    frag_data: dict[str, dict[str, float]],
):
    F = FragmentNetworkX(smiles, {0})
    assert nx.is_directed_acyclic_graph(F.network)

    data = fragment_data(F)

    assert set(data.index) == frags

    for f in frag_data:
        for k in frag_data[f]:
            assert frag_data[f][k] == data.loc[f][k]


@pytest.mark.parametrize(
    "smiles, frags, frag_data",
    ids=["propane", "cyclopropane", "epoxide", "hexane"],
    argvalues=[
        (
            "CCC",
            {"C", "C-C", "C-C-C"},
            {
                "C": {"count": 3},
                "C-C": {"count": 3 / 2},
                "C-C-C": {"count": 1},
            },
        ),
        (
            "C1CC1",
            {"C1-C-C-1"},
            {
                "C1-C-C-1": {
                    "count": 1,
                    "marked_count": 1,
                },
            },
        ),
        (
            "C1OC1",
            {"C1-C-O-1"},
            {
                "C1-C-O-1": {
                    "count": 1,
                    "marked_count": 1,
                },
            },
        ),
        (
            "C1CCCCC1",
            {"C1-C-C-C-C-C-1"},
            {
                "C1-C-C-C-C-C-1": {
                    "count": 1,
                    "marked_count": 1,
                },
            },
        ),
    ],
)
def test_ring_net_ex(
    smiles: str,
    frags: set[str],
    frag_data: dict[str, dict[str, float]],
):
    F = RingFragmentNetworkX(smiles, {0})
    assert nx.is_directed_acyclic_graph(F.network)

    data = fragment_data(F)

    assert set(data.index) == frags

    for f in frag_data:
        for k in frag_data[f]:
            assert frag_data[f][k] == data.loc[f][k]



@settings(max_examples=15, deadline=None)
@given(
  random_smiles_pair(), 
  st.booleans(),
  st.sampled_from([RingFragmentNetworkX, RingFragmentNetworkX]),
)  # type: ignore
def test_order_independence_network(
  smiles_pair, 
  include_mol_ref, 
  cls
):
    smiles, rsmiles = smiles_pair

    F = cls(smiles, max_size=5, include_mol_ref=include_mol_ref)
    rF = cls(rsmiles, max_size=5, include_mol_ref=include_mol_ref)

    F_pd = F.to_pandas()
    rF_pd = rF.to_pandas()

    F = fragment_view(F)
    rF = fragment_view(rF)

    assert set(F) == set(rF)
    assert set(F.edges) == set(rF.edges)

    for frag in F:
        assert F_pd.loc[frag]["count"] == rF_pd.loc[frag]["count"]



@settings(max_examples=5, deadline=None)
@given(random_smiles(), st.booleans())  # type: ignore
def test_update1(smiles, include_mol_ref):
    X = RingFragmentNetworkX(max_size=5, include_mol_ref=include_mol_ref)
    F = RingFragmentNetworkX(smiles, max_size=5, include_mol_ref=include_mol_ref)

    X.update(F)

    assert X.to_pandas().equals(F.to_pandas())  # type: ignore



@settings(max_examples=5, deadline=None)
@given(random_smiles(), st.booleans())  # type: ignore
def test_update_twice(smiles, include_mol_ref):
    X = RingFragmentNetworkX()

    F = RingFragmentNetworkX(smiles, max_size=5, include_mol_ref=include_mol_ref)
    X.update(F)

    F = RingFragmentNetworkX(smiles, max_size=5, include_mol_ref=include_mol_ref)
    X.update(F)

    Xpd = X.to_pandas()
    Fpd = F.to_pandas() * 2

    for col in ["count", "marked_count", "n_mol"]:
      assert np.all(Xpd[col] == Fpd[col]) # type: ignore
     

@settings(max_examples=5, deadline=None)
@given(random_smiles(),random_smiles(), st.booleans())  # type: ignore
def test_update_diff_mols(smi1, smi2, include_mol_ref):

    X = RingFragmentNetworkX()

    F1 = RingFragmentNetworkX(smi1, max_size=5, include_mol_ref=include_mol_ref)
    X.update(F1)

    F2 = RingFragmentNetworkX(smi2, max_size=5, include_mol_ref=include_mol_ref)
    X.update(F2)

    Xpd = X.to_pandas()

    F1frags = set(F1.to_pandas().index) 
    F2frags = set(F2.to_pandas().index)

    assert set(Xpd.index) == F1frags | F2frags

    for frag in F1frags & F2frags:
      assert Xpd["n_mol"].loc[frag] == 2

    for frag in F1frags ^ F2frags:
      assert Xpd["n_mol"].loc[frag] == 1

     