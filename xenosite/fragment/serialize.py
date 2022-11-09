from . import graph
import numpy as np
import typing
from enum import Enum


class Serialized(typing.NamedTuple):
    string: str
    reordering: list[int]


class SERIAL(Enum):
    RING_FORWARD = "rf"
    RING_BACKWARD = "rb"
    TREE_FORWARD = "tf"
    TREE_BACKWARD = "tb"


def serialize(G: "graph.Graph", canonize=True) -> Serialized:
    dfs = graph.dfs_ordered(G, canonize)
    return smiles_serialize(dfs, **G.dict())  # type: ignore


def smiles_serialize(
    dfs: list[tuple[int, int, "graph.DFS_TYPE"]],
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

    # look up for edge labels
    L = {((i, j) if i < j else (j, i)): l for i, j, l in zip(edge[0], edge[1], elabel)}

    # output ordered nodes
    ordered_nodes = [dfs[0][0]] + [j for _, j, t in dfs if t != graph.DFS_TYPE.RING]

    if len(ordered_nodes) != n:
        ValueError(
            "serlialize: DFS does not cover whole molecule. Multiple components in graph are not allowed."
        )

    # collect DFS info for all nodes
    node_info = {}
    for i, j, t in dfs:
        node_info[i] = ni = node_info.get(i, {})
        node_info[j] = nj = node_info.get(j, {})
        e = (i, j)

        if t == graph.DFS_TYPE.TREE:
            # outgoing tree edges
            ni[SERIAL.TREE_FORWARD] = ni.get(SERIAL.TREE_FORWARD, [])
            ni[SERIAL.TREE_FORWARD].append(e)

            # incoming tree edge
            nj[SERIAL.TREE_BACKWARD] = e

        if t == graph.DFS_TYPE.RING:
            # ring closure to earlier atoms (backrefs)
            ni[SERIAL.RING_BACKWARD] = ni.get(SERIAL.RING_BACKWARD, [])
            ni[SERIAL.RING_BACKWARD].append(e)

            # ring closure to laters atoms (forward refs)
            nj[SERIAL.RING_FORWARD] = nj.get(SERIAL.RING_FORWARD, [])
            nj[SERIAL.RING_FORWARD].append(e)

    # assign a non-overlapping number to each ring edge
    ring = {}
    rids = set("123456789")
    active_rids = set()
    for n in ordered_nodes:
        info = node_info[n]
        ids = sorted(rids - active_rids)

        for i, e in zip(ids, info.get(SERIAL.RING_FORWARD, [])):
            ring[e] = i
            active_rids.add(i)

        for e in info.get(SERIAL.RING_FORWARD, []):
            active_rids.remove(ring[e])

    out = []

    # generate output
    for n in ordered_nodes:
        info = node_info[n]

        # start with node label
        o = nlabel[n]

        # prepend edge label of incoming tree edge
        if SERIAL.TREE_BACKWARD in info:
            e = info[SERIAL.TREE_BACKWARD]
            _e = e if e[0] < e[1] else (e[1], e[0])
            o = L[_e] + o

        # append edge label and ID of backref rings
        for e in info.get(SERIAL.RING_BACKWARD, []):  #
            _e = e if e[0] < e[1] else (e[1], e[0])
            o += L[_e] + ring[e]

        # append ID of forwardref rings
        for e in info.get(SERIAL.RING_FORWARD, []):
            o += ring[e]

        # prepend open and close parenthesis for branches
        if SERIAL.TREE_BACKWARD in info:
            e = info[SERIAL.TREE_BACKWARD]  # incoming tree edge
            source = e[0]  # source of incoming edge
            fe = node_info[source][
                SERIAL.TREE_FORWARD
            ]  # list of outgoing tree edges from source

            if e != fe[-1]:  # open parenth if not last outgoing
                o = "(" + o
            if e != fe[0]:  # close parenth if not first outgoing
                o = ")" + o

        out.append(o)

    return Serialized("".join(out), ordered_nodes)
