"""Scenario loading: the only way the rest of the code reads data.

Reads the processed files written by `python -m data_pipeline.build`
(never the raw CSVs) and cuts out one experiment:

  houses  the n houses of the smallest feeder branch that holds n,
          taken in depth-first order so they sit next to each other
          on the cable, not the first n by id
  graph   the selected houses plus every junction and cable between
          them and the transformer, so the electrical path is complete
  window  `days` days from `start`

SIGN CONVENTION
    net_power_kw = pv_power_kw - load_power_kw   (> 0 surplus, < 0 deficit)
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2p.config import Config


@dataclass(frozen=True)
class Scenario:
    timestamps: pd.DatetimeIndex    # interval start, local wall-clock as recorded
    dt_hours: float
    houses: pd.DataFrame            # one row per agent, in agent order
    load_kw: np.ndarray             # (T, n_houses), REAL
    pv_kw: np.ndarray               # (T, n_houses), REAL
    nodes: pd.DataFrame             # subgraph, `local_id` 0..M-1, transformer first
    edges: pd.DataFrame             # subgraph, endpoints as `from_local` / `to_local`

    @property
    def net_power_kw(self) -> np.ndarray:
        return self.pv_kw - self.load_kw

    @property
    def edge_index(self) -> np.ndarray:
        """(2, 2E) undirected edge list over local ids, the PyG convention."""
        a, b = self.edges["from_local"].to_numpy(), self.edges["to_local"].to_numpy()
        return np.stack([np.concatenate([a, b]), np.concatenate([b, a])])

    @property
    def house_local_ids(self) -> np.ndarray:
        """Local graph id of each agent, in agent order."""
        local = dict(zip(self.nodes["node_id"], self.nodes["local_id"]))
        return self.houses["node_id"].map(local).to_numpy()


def select_branch_houses(nodes: pd.DataFrame, edges: pd.DataFrame, n: int) -> list:
    """Node ids of n houses from the smallest subtree that holds at least n."""

    children = defaultdict(list)
    for e in edges.itertuples(index=False):
        children[e.from_node_id].append(e.to_node_id)
    is_house = dict(zip(nodes["node_id"], nodes["node_type"] == "house"))

    def houses_under(v):
        order, stack = [], [v]
        while stack:
            u = stack.pop()
            if is_house[u]:
                order.append(u)
            stack.extend(sorted(children[u], reverse=True))
        return order

    n_total = int(sum(is_house.values()))
    if not 1 <= n <= n_total:
        raise ValueError(f"n_houses must lie in [1, {n_total}], got {n}")

    # The transformer holds every house, so there is always a candidate.
    candidates = [(len(h), v, h) for v in nodes["node_id"] if len(h := houses_under(v)) >= n]
    _, _, order = min(candidates, key=lambda c: (c[0], c[1]))
    return order[:n]


def path_subgraph(nodes: pd.DataFrame, edges: pd.DataFrame, house_ids: list):
    """Nodes and edges on the paths from the chosen houses to the transformer."""

    parent_edge = {e.to_node_id: e for e in edges.itertuples(index=False)}
    keep_nodes, keep_edges = set(), set()
    for v in house_ids:
        while True:
            keep_nodes.add(v)
            e = parent_edge.get(v)
            if e is None:
                break
            keep_edges.add(e.edge_id)
            v = e.from_node_id

    sub_nodes = nodes[nodes["node_id"].isin(keep_nodes)].sort_values("node_id").reset_index(drop=True)
    sub_nodes.insert(0, "local_id", range(len(sub_nodes)))
    local = dict(zip(sub_nodes["node_id"], sub_nodes["local_id"]))
    sub_edges = edges[edges["edge_id"].isin(keep_edges)].sort_values("edge_id").reset_index(drop=True)
    sub_edges.insert(1, "from_local", sub_edges["from_node_id"].map(local))
    sub_edges.insert(2, "to_local", sub_edges["to_node_id"].map(local))
    return sub_nodes, sub_edges


def load_scenario(cfg: Config) -> Scenario:
    d = cfg.processed_dir
    if not (d / "houses_timeseries.csv.gz").exists():
        raise FileNotFoundError(f"{d} has no processed data; run `python -m data_pipeline.build` first")

    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    mapping = pd.read_csv(d / "house_mapping.csv")

    chosen = select_branch_houses(nodes, edges, cfg.scenario.n_houses.value)
    houses = mapping.set_index("node_id").loc[chosen].reset_index()
    sub_nodes, sub_edges = path_subgraph(nodes, edges, chosen)

    start = pd.Timestamp(cfg.scenario.start.value)
    end = start + pd.Timedelta(days=cfg.scenario.days.value)
    series = pd.read_csv(
        d / "houses_timeseries.csv.gz",
        usecols=["timestamp", "house_id", "load_power_kw", "pv_power_kw"],
        parse_dates=["timestamp"],
    )
    series = series[(series["timestamp"] >= start) & (series["timestamp"] < end)
                    & series["house_id"].isin(houses["house_id"])]
    if series.empty:
        raise ValueError(f"no data between {start} and {end}")

    order = houses["house_id"].tolist()
    load = series.pivot(index="timestamp", columns="house_id", values="load_power_kw")[order]
    pv = series.pivot(index="timestamp", columns="house_id", values="pv_power_kw")[order]
    if load.isna().any().any() or pv.isna().any().any():
        raise ValueError("scenario window has missing intervals")

    steps = np.diff(load.index.values).astype("timedelta64[s]").astype(float) / 3600.0
    if not np.allclose(steps, steps[0]):
        raise ValueError("timestamps are not evenly spaced")
    dt_hours = float(steps[0])

    return Scenario(
        timestamps=load.index,
        dt_hours=dt_hours,
        houses=houses,
        load_kw=load.to_numpy(),
        pv_kw=pv.to_numpy(),
        nodes=sub_nodes,
        edges=sub_edges,
    )
