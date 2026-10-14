"""DC power flow on the scenario's feeder subgraph.

MODEL (our DC abstraction of an AC benchmark feeder)
  - Topology and cable resistance from the IEEE European LV feeder.
  - Two-wire DC: current goes out on one conductor and returns on another,
    so each edge has loop resistance R = 2 x R1 (R1 = per-conductor
    resistance of the published cable). Reactance plays no part in DC.
  - The grid-tie converter at the transformer node holds the bus at
    `bus_voltage_v` (the slack) and supplies or absorbs the imbalance.
  - Each house injects its residual power at its node: > 0 export into
    the network, < 0 draw from it. Junctions inject nothing.

SOLVER
  Backward/forward sweep, the standard method for radial networks:
    1. node current  I_n = P_n / V_n
    2. backward: edge current = everything drawn below it
    3. forward:  V_child = V_parent - R_edge * I_edge
  repeated until voltages stop changing. DC has no angles, so this is a
  fixed-point iteration on real numbers. Voltage collapse (too much power
  for the bus voltage) shows up as non-convergence: the result is then
  flagged and its voltages, currents and losses are NaN, never numbers
  from a diverged iteration.

SIGN CONVENTION
  edge current > 0  flows away from the transformer (feeding load downstream)
  edge current < 0  flows towards the transformer (exports going upstream)
  grid power   > 0  the grid supplies the community
"""

from dataclasses import dataclass

import numpy as np

from p2p.scenario import Scenario

W_PER_KW = 1000.0
LOOP_FACTOR = 2.0


@dataclass(frozen=True)
class DCNetwork:
    parent: np.ndarray        # (M,) parent local id, -1 for the transformer
    parent_edge: np.ndarray   # (M,) edge index to the parent, -1 for the transformer
    r_loop_ohm: np.ndarray    # (E,)
    order: np.ndarray         # (M,) local ids, every parent before its children
    house_nodes: np.ndarray   # (N,) local id of each agent
    v_slack: float

    @classmethod
    def from_scenario(cls, sc: Scenario, v_slack: float) -> "DCNetwork":
        m = len(sc.nodes)
        parent = np.full(m, -1)
        parent_edge = np.full(m, -1)
        for i, e in enumerate(sc.edges.itertuples(index=False)):
            parent[e.to_local] = e.from_local
            parent_edge[e.to_local] = i
        roots = np.flatnonzero(parent == -1)
        if len(roots) != 1 or sc.nodes.loc[roots[0], "node_type"] != "substation_lv":
            raise ValueError("subgraph must be a tree rooted at the transformer")

        depth = np.zeros(m, dtype=int)
        for _ in range(m):
            depth = np.where(parent >= 0, depth[np.maximum(parent, 0)] + 1, 0)
        return cls(
            parent=parent,
            parent_edge=parent_edge,
            r_loop_ohm=LOOP_FACTOR * sc.edges["r1_ohm"].to_numpy(),
            order=np.argsort(depth, kind="stable"),
            house_nodes=sc.house_local_ids,
            v_slack=v_slack,
        )


@dataclass(frozen=True)
class FlowResult:
    v_node: np.ndarray        # (M,) volts
    i_edge: np.ndarray        # (E,) amps, signed as above
    loss_kw: float
    grid_kw: float            # power the grid-tie converter supplies (> 0) or absorbs (< 0)
    converged: bool
    iterations: int

    def v_house(self, net: DCNetwork) -> np.ndarray:
        return self.v_node[net.house_nodes]


def solve(net: DCNetwork, house_injection_kw, tol_v: float = 1e-6, max_iter: int = 100) -> FlowResult:
    m = len(net.parent)
    p_node = np.zeros(m)
    np.add.at(p_node, net.house_nodes, np.asarray(house_injection_kw, dtype=float) * W_PER_KW)

    v = np.full(m, net.v_slack)
    i_edge = np.zeros(len(net.r_loop_ohm))
    converged = False
    for it in range(1, max_iter + 1):
        injected = p_node / v                       # A, > 0 into the network
        subtree = injected.copy()
        for n in net.order[::-1]:                   # leaves first
            if net.parent[n] >= 0:
                subtree[net.parent[n]] += subtree[n]
                i_edge[net.parent_edge[n]] = -subtree[n]

        v_new = v.copy()
        for n in net.order:                         # transformer first
            if net.parent[n] >= 0:
                e = net.parent_edge[n]
                v_new[n] = v_new[net.parent[n]] - net.r_loop_ohm[e] * i_edge[e]

        if not np.all(v_new > 0):
            v = v_new
            break
        delta = np.max(np.abs(v_new - v))
        v = v_new
        if delta < tol_v:
            converged = True
            break

    if not converged:
        nan = float("nan")
        return FlowResult(np.full(m, nan), np.full(len(i_edge), nan), nan, nan, False, it)
    loss_kw = float(np.sum(net.r_loop_ohm * i_edge**2)) / W_PER_KW
    grid_kw = -float(p_node.sum()) / W_PER_KW + loss_kw
    return FlowResult(v, i_edge.copy(), loss_kw, grid_kw, True, it)
