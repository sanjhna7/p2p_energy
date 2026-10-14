"""IEEE European Low Voltage Test Feeder -> graph for the GAT.

Source: IEEE PES Distribution Test Feeders Working Group,
European_LV_Test_Feeder_v2.zip (documentation dated May 19, 2015).
  - radial, 50 Hz, 11/0.416 kV 800 kVA transformer
  - 906 LV buses, 905 line segments, 55 single-phase residential loads
  - line codes give positive/zero-sequence R and X in ohm/km

This is an AC three-phase feeder. We keep it as published here; the DC
re-rating (dropping X, loop resistance, DC bus voltage) is a modelling
step done later in the environment, not in the data layer.

Most of the 906 buses are drawing vertices only centimetres apart, so the
graph is reduced to the nodes that matter: the transformer LV bus, the 55
house connection buses and the branching junctions. Series segments between
them are merged by summing length, R and X (exact for series impedances).
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd


def _read(csv_dir, name):
    df = pd.read_csv(Path(csv_dir) / name, comment="#", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def read_feeder(csv_dir):
    lines = _read(csv_dir, "Lines.csv")
    codes = _read(csv_dir, "LineCodes.csv")
    loads = _read(csv_dir, "Loads.csv")
    coords = _read(csv_dir, "Buscoords.csv")
    transformer = _read(csv_dir, "Transformer.csv")

    for df, cols in ((lines, ["Bus1", "Bus2"]), (loads, ["Bus"]), (coords, ["Busname"])):
        for c in cols:
            df[c] = df[c].astype(str).str.strip()

    if set(lines["Units"].str.strip()) != {"m"}:
        raise ValueError("Unexpected line length units in Lines.csv")
    if set(codes["Units"].str.strip()) != {"km"}:
        raise ValueError("Unexpected line code units in LineCodes.csv")

    lines = lines.merge(codes[["Name", "R1", "X1", "R0", "X0"]], left_on="LineCode", right_on="Name",
                        suffixes=("", "_code"), how="left")
    if lines["R1"].isna().any():
        raise ValueError("Line references an unknown line code")

    length_km = lines["Length"] / 1000.0
    lines["r1_ohm"] = lines["R1"] * length_km
    lines["x1_ohm"] = lines["X1"] * length_km
    lines["r0_ohm"] = lines["R0"] * length_km
    lines["x0_ohm"] = lines["X0"] * length_km
    lines["length_m"] = lines["Length"]

    root_bus = str(transformer.loc[0, "bus2"]).strip()
    return lines, loads, coords, transformer, root_bus


def full_edges(lines):
    return lines[["Name", "Bus1", "Bus2", "LineCode", "length_m", "r1_ohm", "x1_ohm", "r0_ohm", "x0_ohm"]].rename(
        columns={"Name": "segment_name", "Bus1": "from_bus", "Bus2": "to_bus", "LineCode": "line_code"}
    )


def prune_dead_ends(lines, loads, root_bus):
    """Drop cable stubs that end without a load; they carry no power flow."""
    protected = set(loads["Bus"]) | {root_bus}
    active = lines.copy()
    while True:
        degree = pd.concat([active["Bus1"], active["Bus2"]]).value_counts()
        leaves = set(degree.index[degree == 1]) - protected
        if not leaves:
            return active
        active = active[~(active["Bus1"].isin(leaves) | active["Bus2"].isin(leaves))]


def reduce_feeder(lines, loads, coords, root_bus):
    """Merge chains of pass-through buses; keep root, load and junction buses.

    `lines` should already be pruned of dead ends (see prune_dead_ends)."""
    adjacency = defaultdict(list)
    for row in lines.itertuples(index=False):
        adjacency[row.Bus1].append((row.Bus2, row))
        adjacency[row.Bus2].append((row.Bus1, row))

    load_buses = set(loads["Bus"])
    kept = {root_bus} | load_buses | {b for b, nbrs in adjacency.items() if len(nbrs) != 2}

    edges = []
    visited_segments = set()
    for start in kept:
        for nxt, seg in adjacency[start]:
            if seg.Name in visited_segments:
                continue
            path_segments = [seg]
            visited_segments.add(seg.Name)
            prev, cur = start, nxt
            while cur not in kept:
                (a, sa), (b, sb) = adjacency[cur]
                step, step_seg = (b, sb) if a == prev else (a, sa)
                path_segments.append(step_seg)
                visited_segments.add(step_seg.Name)
                prev, cur = cur, step
            edges.append(
                {
                    "from_node": start,
                    "to_node": cur,
                    "length_m": sum(s.length_m for s in path_segments),
                    "r1_ohm": sum(s.r1_ohm for s in path_segments),
                    "x1_ohm": sum(s.x1_ohm for s in path_segments),
                    "r0_ohm": sum(s.r0_ohm for s in path_segments),
                    "x0_ohm": sum(s.x0_ohm for s in path_segments),
                    "n_segments": len(path_segments),
                    "line_codes": ";".join(sorted({s.LineCode for s in path_segments})),
                }
            )

    edges = pd.DataFrame(edges)
    if len(visited_segments) != len(lines):
        raise ValueError("Reduction did not cover every line segment")
    if len(edges) != len(kept) - 1:
        raise ValueError("Reduced graph is not a tree; the feeder should be radial")

    # Orient every edge away from the transformer so from_node is upstream.
    tree = defaultdict(list)
    for i, e in edges.iterrows():
        tree[e.from_node].append((e.to_node, i))
        tree[e.to_node].append((e.from_node, i))
    depth, parent_edge, stack = {root_bus: 0}, {}, [root_bus]
    while stack:
        u = stack.pop()
        for v, i in tree[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                parent_edge[v] = i
                if edges.at[i, "from_node"] != u:
                    edges.loc[i, ["from_node", "to_node"]] = [u, v]
                stack.append(v)

    load_by_bus = loads.set_index("Bus")
    coord_by_bus = coords.set_index("Busname")
    neighbours = defaultdict(list)
    for e in edges.itertuples(index=False):
        neighbours[e.from_node].append(e.to_node)
        neighbours[e.to_node].append(e.from_node)

    distance_m = {root_bus: 0.0}
    resistance_ohm = {root_bus: 0.0}
    for v in sorted(depth, key=depth.get):
        if v == root_bus:
            continue
        e = edges.loc[parent_edge[v]]
        distance_m[v] = distance_m[e.from_node] + e.length_m
        resistance_ohm[v] = resistance_ohm[e.from_node] + e.r1_ohm

    nodes = []
    for bus in sorted(kept, key=lambda b: (depth[b], int(b))):
        is_load = bus in load_buses
        nodes.append(
            {
                "source_bus": bus,
                "node_type": "substation_lv" if bus == root_bus else ("house" if is_load else "junction"),
                "load_name": load_by_bus.at[bus, "Name"] if is_load else "",
                "load_phase": load_by_bus.at[bus, "phases"] if is_load else "",
                "x": coord_by_bus.at[bus, "x"],
                "y": coord_by_bus.at[bus, "y"],
                "depth": depth[bus],
                "distance_from_transformer_m": distance_m[bus],
                "path_r1_ohm_from_transformer": resistance_ohm[bus],
                "neighbor_source_buses": ";".join(sorted(neighbours[bus], key=int)),
            }
        )
    nodes = pd.DataFrame(nodes)
    nodes.insert(0, "node_id", range(len(nodes)))

    bus_to_node = dict(zip(nodes["source_bus"], nodes["node_id"]))
    nodes["neighbor_ids"] = nodes["neighbor_source_buses"].apply(
        lambda s: ";".join(str(bus_to_node[b]) for b in s.split(";") if b)
    )
    edges.insert(0, "edge_id", range(len(edges)))
    edges.insert(1, "from_node_id", edges["from_node"].map(bus_to_node))
    edges.insert(2, "to_node_id", edges["to_node"].map(bus_to_node))
    edges = edges.rename(columns={"from_node": "from_source_bus", "to_node": "to_source_bus"})
    return nodes, edges
