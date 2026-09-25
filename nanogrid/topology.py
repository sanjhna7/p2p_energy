"""
Community topology.

PROVISIONAL ASSUMPTION. The Ausgrid dataset contains no network
connectivity, and customer ids carry no spatial meaning whatsoever -
customer 1 and customer 2 are not neighbours in any real sense. The
topology below is our own simulation assumption about how the houses
share the community DC bus, and it is deliberately stored apart from the
time series so it can be swapped without regenerating any data.
"""

import json
from typing import List, Tuple


def build_edges(n_nodes: int, topology: str) -> List[Tuple[int, int]]:
    """Edges for the configured community layout.

    chain : House0 - House1 - ... - HouseN-1   (default; matches the
            single shared community bus the hardware team is building)
    ring  : chain with the ends joined
    star  : House0 at the centre, every other house connected to it
    full  : every house connected to every other house
    """

    if n_nodes < 1:
        raise ValueError("a community needs at least one house")

    if topology == "chain":
        return [(i, i + 1) for i in range(n_nodes - 1)]

    if topology == "ring":
        if n_nodes < 3:
            return [(i, i + 1) for i in range(n_nodes - 1)]
        return [(i, (i + 1) % n_nodes) for i in range(n_nodes)]

    if topology == "star":
        return [(0, i) for i in range(1, n_nodes)]

    if topology == "full":
        return [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    raise ValueError(
        f"unknown topology {topology!r}; expected chain, ring, star or full"
    )


def build_topology(house_ids, customer_ids, topology: str) -> dict:
    nodes = list(range(len(house_ids)))
    return {
        "provenance": "provisional_simulation",
        "note": (
            "Preliminary community layout. NOT derived from Ausgrid - the "
            "source dataset has no network topology, and customer ids are "
            "not spatial. Replace with the Simulink/hardware layout."
        ),
        "topology": topology,
        "nodes": nodes,
        "edges": [list(e) for e in build_edges(len(nodes), topology)],
        "node_attributes": [
            {"node": n, "house_id": h, "ausgrid_customer_id": c}
            for n, h, c in zip(nodes, house_ids, customer_ids)
        ],
    }


def save_topology(topology: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(topology, fh, indent=2)


def load_topology(path) -> dict:
    with open(path) as fh:
        return json.load(fh)
