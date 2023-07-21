from __future__ import annotations

import typing
from enum import Enum

import heapq
import numpy as np

from . import graph


class Serialized(typing.NamedTuple):
    string: str
    reordering: list[int]


def serialize(G: "graph.Graph", canonize=True) -> Serialized:
    if G.nlabel is None:
        raise ValueError("Need node labels.")
    
    if G.elabel is None:
        if G.edge[0]:
          raise ValueError("Need edge labels.")

    dfs = graph.dfs_ordered(G, canonize)

    return smiles_serialize(dfs, G.n, G.nlabel, G.elabel, G.edge)


class NodeInfo(typing.TypedDict):
    ring_forward: list["graph.DFS_EDGE"]
    ring_backward: list["graph.DFS_EDGE"]
    tree_forward: list["graph.DFS_EDGE"]
    tree_backward: list["graph.DFS_EDGE"]


def _collect_node_info(
    dfs: list["graph.DFS_EDGE"], nodes: list[int]
) -> dict[int, NodeInfo]:

    # collect DFS info for all nodes
    node_info: dict[int, NodeInfo] = {
        i: NodeInfo(
            ring_forward=[], ring_backward=[], tree_forward=[], tree_backward=[]
        )
        for i in nodes
    }

    for EDGE in dfs:
        i, j, t = EDGE
        ni = node_info[i]
        nj = node_info[j]

        if t == graph.DFS_TYPE.TREE:
            # outgoing tree edges
            ni["tree_forward"].append(EDGE)

            # incoming tree edge
            assert nj["tree_backward"] == []
            nj["tree_backward"] = [EDGE]

        if t == graph.DFS_TYPE.RING:
            # ring closure to earlier atoms (backrefs)
            ni["ring_backward"].append(EDGE)

            # ring closure to later atoms (forward refs)
            nj["ring_forward"].append(EDGE)
    return node_info


def smiles_serialize(
    dfs: list["graph.DFS_EDGE"],
    n: int,
    nlabel: list[str],
    elabel: list[str],
    edge: tuple[np.ndarray[np.int64], np.ndarray[np.int64]],
) -> Serialized:
    # sourcery skip: use-fstring-for-concatenation

    if n == 0:
        return Serialized("", [])
    if n == 1:
        return Serialized(nlabel[0], [0])

    if not dfs:
        raise ValueError("Multiple components in graph are not allowed.")

    # output-ordered nodes
    ordered_nodes: list[int] = [dfs[0][0]] + [
        j for _, j, t in dfs if t != graph.DFS_TYPE.RING
    ]

    if len(ordered_nodes) != n:
        raise ValueError("Multiple components in graph are not allowed.")

    node_info = _collect_node_info(dfs, ordered_nodes)

    ring_labels = _label_rings(node_info, ordered_nodes)

    dfs_labels = _label_dfs(dfs, elabel, edge)

    return _serialize(node_info, ordered_nodes, nlabel, dfs_labels, ring_labels)


def _label_dfs(
    dfs: list["graph.DFS_EDGE"],
    elabel: list[str],
    edge: tuple[np.ndarray[np.int64], np.ndarray[np.int64]],
) -> dict["graph.DFS_EDGE", str]:

    # look up for edge labels
    L: dict[tuple[int, int], str] = {
        (i, j): l for i, j, l in zip(edge[0], edge[1], elabel)
    }

    dfs_labels: dict["graph.DFS_EDGE", str] = {}
    for E in dfs:
        i, j, _ = E

        if (i, j) in L:
            dfs_labels[E] = L[(i, j)]
            continue

        if (j, i) in L:
            dfs_labels[E] = L[(j, i)]

    return dfs_labels


class RingIds:
    def __init__(self):
        self.active = set()
        self.inactive = []
        self.n = 0

    def open_ring(self) -> int:
        if not self.inactive:
            ring_id = self._add_id()
        else:
            ring_id = heapq.heappop(self.inactive)

        self.active.add(ring_id)
        return ring_id

    def close_ring(self, ring_id: int):
        self.active.remove(ring_id)
        heapq.heappush(self.inactive, ring_id)

    @classmethod
    def format(cls, n):
        s = str(n)
        s = "%" + s if len(s) > 1 else s
        return s

    def _add_id(self):
        self.n += 1
        return self.n


def _label_rings(
    node_info: dict[int, NodeInfo], ordered_nodes: list[int]
) -> dict["graph.DFS_EDGE", str]:
    R = RingIds()

    # assign a non-overlapping number to each ring edge
    ring: dict[graph.DFS_EDGE, int] = {}

    for n in ordered_nodes:
        info = node_info[n]

        for e in info["ring_forward"]:
            ring[e] = R.open_ring()

        for e in info["ring_backward"]:
            R.close_ring(ring[e])

    return {k: R.format(v) for k, v in ring.items()}


def _serialize(
    node_info: dict[int, NodeInfo],
    ordered_nodes: list[int],
    nlabel: list[str],
    dfs_labels: dict["graph.DFS_EDGE", str],
    ring_labels: dict["graph.DFS_EDGE", str],
) -> Serialized:

    # sourcery skip: use-fstring-for-concatenation

    out: list[str] = []

    # generate output
    for n in ordered_nodes:
        info = node_info[n]

        # start with node label
        o = nlabel[n]

        # prepend edge label of incoming tree edge
        if e := info["tree_backward"]:
            o = dfs_labels[e[0]] + o

        # append edge label and ID of backref rings
        for e in info["ring_backward"]:
            o += dfs_labels[e] + ring_labels[e]

        # append ID of forwardref rings
        for e in info["ring_forward"]:
            o += ring_labels[e]

        # prepend open and close parenthesis for branches
        if e := info["tree_backward"]:  # incoming tree edge
            # source of incoming edge is e[0]
            fe = node_info[e[0][0]][
                "tree_forward"
            ]  # list of outgoing tree edges from source

            if e[0] != fe[-1]:  # open parenth if not last outgoing
                o = "(" + o
            if e[0] != fe[0]:  # close parenth if not first outgoing
                o = ")" + o

        out.append(o)

    return Serialized("".join(out), ordered_nodes)
