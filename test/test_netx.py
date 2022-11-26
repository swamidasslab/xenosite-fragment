from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX
from hypothesis import strategies as st, given, assume, settings
from test.util import random_smiles_pair
import networkx as nx
import pytest


def fragment_data(N: FragmentNetworkX) -> dict[str, dict]:
    return {k: v for k, v in N.network.nodes.items() if isinstance(k, str)}


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

    assert set(data) == frags

    for f in frag_data:
        for k in frag_data[f]:
            assert frag_data[f][k] == data[f][k]


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

    assert set(data) == frags

    for f in frag_data:
        for k in frag_data[f]:
            assert frag_data[f][k] == data[f][k]


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
