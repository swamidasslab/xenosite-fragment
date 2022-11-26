from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX
from hypothesis import strategies as st, given, assume, settings
from test.util import random_smiles_pair
import networkx as nx


def fragment_data(N: FragmentNetworkX) -> dict[str, dict]:
    return {k: v for k, v in N.network.nodes.items() if isinstance(k, str)}


def fragment_view(N: FragmentNetworkX) -> nx.DiGraph:
    return nx.subgraph_view(N.network, filter_node=lambda x: isinstance(x, str))  # type: ignore


def test_propane():
    F = FragmentNetworkX("CCC")

    nodes = fragment_data(F)

    assert set(nodes) == {"C", "C-C", "C-C-C"}

    assert nodes["C"]["count"] == 3
    assert nodes["C-C"]["count"] == 3 / 2
    assert nodes["C-C-C"]["count"] == 1


def test_propane_ring():
    F = RingFragmentNetworkX("CCC")

    nodes = fragment_data(F)

    assert set(nodes) == {"C", "C-C", "C-C-C"}

    assert nodes["C"]["count"] == 3
    assert nodes["C-C"]["count"] == 3 / 2
    assert nodes["C-C-C"]["count"] == 1


@given(st.integers(max_value=2, min_value=0))
def test_cyclopropane(rid):
    F = FragmentNetworkX("C1CC1", {rid})

    nodes = fragment_data(F)

    assert set(nodes) == {"C", "C-C", "C1-C-C-1"}

    assert nodes["C"]["count"] == 3
    assert nodes["C-C"]["count"] == 3 / 2
    assert nodes["C1-C-C-1"]["count"] == 1

    assert nodes["C"]["marked_count"] == 1
    assert nodes["C-C"]["marked_count"] == 3 / 2
    assert nodes["C1-C-C-1"]["marked_count"] == 1


@given(st.integers(max_value=2, min_value=0))
def test_cyclopropane_ring(rid):
    F = RingFragmentNetworkX("C1CC1", {rid})

    nodes = fragment_data(F)

    assert set(nodes) == {"C1-C-C-1"}

    assert nodes["C1-C-C-1"]["count"] == 1
    assert nodes["C1-C-C-1"]["marked_count"] == 1


def test_epoxide():
    F = FragmentNetworkX("C1OC1")

    nodes = fragment_data(F)

    assert set(nodes) == {"C", "C-C", "C1-C-O-1", "C-O", "O"}

    assert nodes["C"]["count"] == 2
    assert nodes["C-C"]["count"] == 1
    assert nodes["C1-C-O-1"]["count"] == 1
    assert nodes["C-O"]["count"] == 3 / 2
    assert nodes["O"]["count"] == 1


def test_epoxide_ring():
    F = RingFragmentNetworkX("C1OC1")

    nodes = fragment_data(F)

    assert set(nodes) == {"C1-C-O-1"}

    assert nodes["C1-C-O-1"]["count"] == 1


@given(st.integers(max_value=5, min_value=0))
def test_hexane(rid):
    F = FragmentNetworkX("C1CCCCC1", {rid})

    nodes = fragment_data(F)

    assert set(nodes) == {
        "C",
        "C-C",
        "C-C-C",
        "C-C-C-C",
        "C-C-C-C-C",
        "C1-C-C-C-C-C-1",
    }

    assert nodes["C"]["count"] == 6
    assert nodes["C-C"]["count"] == 3
    assert nodes["C-C-C"]["count"] == 2
    assert nodes["C-C-C-C"]["count"] == 6 / 4
    assert nodes["C-C-C-C-C"]["count"] == 6 / 5
    assert nodes["C1-C-C-C-C-C-1"]["count"] == 1

    assert nodes["C"]["marked_count"] == 1
    assert nodes["C-C"]["marked_count"] == 3 / 2
    assert nodes["C-C-C"]["marked_count"] == 5 / 3
    assert nodes["C-C-C-C"]["marked_count"] == 6 / 4
    assert nodes["C-C-C-C-C"]["marked_count"] == 6 / 5
    assert nodes["C1-C-C-C-C-C-1"]["marked_count"] == 1


@given(st.integers(max_value=5, min_value=0))
def test_hexane_ring(rid):
    F = RingFragmentNetworkX("C1CCCCC1", {rid})

    nodes = fragment_data(F)

    assert set(nodes) == {"C1-C-C-C-C-C-1"}

    assert nodes["C1-C-C-C-C-C-1"]["count"] == 1
    assert nodes["C1-C-C-C-C-C-1"]["marked_count"] == 1


@settings(max_examples=20)
@given(random_smiles_pair())  # type: ignore
def test_order_independence_fragnetwork(smiles_pair):
    smiles, rsmiles = smiles_pair
    assume(smiles != rsmiles)

    F = fragment_view(RingFragmentNetworkX(smiles, max_size=5))
    rF = fragment_view(RingFragmentNetworkX(rsmiles, max_size=5))

    F = nx.subgraph_view(F, filter_node=lambda x: isinstance(x, str))  # type: ignore
    rF = nx.subgraph_view(rF, filter_node=lambda x: isinstance(x, str))  # type: ignore

    assert set(F) == set(rF)
    assert set(F.edges) == set(rF.edges)

    for frag in F:
        assert F.nodes[frag]["count"] == rF.nodes[frag]["count"]


@settings(max_examples=20)
@given(random_smiles_pair())  # type: ignore
def test_order_independence_ringfragnetwork(smiles_pair):
    smiles, rsmiles = smiles_pair
    assume(smiles != rsmiles)

    F = fragment_view(RingFragmentNetworkX(smiles, max_size=5))
    rF = fragment_view(RingFragmentNetworkX(rsmiles, max_size=5))

    assert set(F) == set(rF)
    assert set(F.edges) == set(rF.edges)

    for frag in F:
        assert F.nodes[frag]["count"] == rF.nodes[frag]["count"]
