import typing
from enum import Enum

import numpy as np

from . import graph


class Serialized(typing.NamedTuple):
    string: str
    reordering: list[int]


class SERIAL(Enum):
    RING_FORWARD =  4
    RING_BACKWARD = 6 # "rb"
    TREE_FORWARD = 7 # "tf"
    TREE_BACKWARD = 8 # "tb"


def serialize(G: "graph.Graph", canonize=True) -> Serialized:
    if G.nlabel is None:
        raise ValueError("Need node labels.")
    if G.elabel is None:
        print(G)
        raise ValueError("Need edge labels.")

    dfs = graph.dfs_ordered(G, canonize)

    return smiles_serialize(dfs, G.n, G.nlabel, G.elabel, G.edge)


class NodeInfo(typing.TypedDict):
  ring_forward : list["graph.DFS_EDGE"]
  ring_backward : list["graph.DFS_EDGE"]
  tree_forward : list["graph.DFS_EDGE"]
  tree_backward : typing.Optional["graph.DFS_EDGE"]


def collect_node_info(dfs: list["graph.DFS_EDGE"], nodes : list[int]) -> dict[int,  NodeInfo]:

    # collect DFS info for all nodes
    node_info : dict[int,  NodeInfo]= {
      i: NodeInfo(ring_forward=[], ring_backward=[], tree_forward=[], tree_backward=None)
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
            nj["tree_backward"] = EDGE

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
    ordered_nodes : list[int] = [dfs[0][0]] + [j for _, j, t in dfs if t != graph.DFS_TYPE.RING]

    if len(ordered_nodes) != n:
        raise ValueError("Multiple components in graph are not allowed.")

    node_info = collect_node_info(dfs, ordered_nodes)

    ring_labels = label_rings(node_info, ordered_nodes)

    edge_labels = label_edges(dfs, elabel, edge)
    
    return _serialize(node_info, ordered_nodes, ring_labels, edge_labels, nlabel)



def label_edges(
    dfs: list["graph.DFS_EDGE"],
    elabel: list[str],
    edge: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]
  ) -> dict["graph.DFS_EDGE", str]:

    # look up for edge labels
    L : dict[tuple[int, int], str] = {(i, j): l for i, j, l in zip(edge[0], edge[1], elabel)}

    edge_labels : dict["graph.DFS_EDGE", str] = {}
    for E in dfs:
      i, j, _  = E

      if (i, j) in L:
        edge_labels[E] = L[(i,j)]
        continue

      if (j, i) in L: edge_labels[E] = L[(j,i)]

    return edge_labels
  


def label_rings(node_info: dict[int,  NodeInfo], ordered_nodes : list[int]) -> dict["graph.DFS_EDGE", str]:

    # assign a non-overlapping number to each ring edge
    ring : dict[graph.DFS_EDGE, str] = {}
    rids : set[str] = set("123456789")
    active_rids : set[str]= set()

    for n in ordered_nodes:
        info = node_info[n]
        ids = sorted(rids - active_rids)

        for i, e in zip(ids, info["ring_forward"]): #TODO: Check??? or "forward"??
            ring[e] = i
            active_rids.add(i)

        for e in info["ring_backward"]:
            active_rids.remove(ring[e])

    return ring


def _serialize(
  node_info: dict[int,  NodeInfo],
  ordered_nodes: list[int],
  ring_labels : dict["graph.DFS_EDGE", str],
  edge_labels:  dict["graph.DFS_EDGE", str],
  nlabel: list[str]
) -> Serialized:

    # sourcery skip: use-fstring-for-concatenation

    out : list[str] = []

    # generate output
    for n in ordered_nodes:
        info = node_info[n]

        # start with node label
        o = nlabel[n]

        # prepend edge label of incoming tree edge
        if e := info["tree_backward"]:
            o = edge_labels[e] + o

        # append edge label and ID of backref rings
        for e in info["ring_backward"]: 
            o += edge_labels[e] + ring_labels[e]

        # append ID of forwardref rings
        for e in  info["ring_forward"]:
            o += ring_labels[e]

        # prepend open and close parenthesis for branches
        if e := info["tree_backward"]: # incoming tree edge
            # source of incoming edge is e[0] 
            fe = node_info[e[0]][
                "tree_forward"
            ]  # list of outgoing tree edges from source

            if e != fe[-1]:  # open parenth if not last outgoing
                o = "(" + o
            if e != fe[0]:  # close parenth if not first outgoing
                o = ")" + o

        out.append(o)

    return Serialized("".join(out), ordered_nodes)
