"""Reward: minus cost, minus penalties for unsafe or impossible proposals.

    reward_i = -cost_i - own_i - shared

    own_i   = w_own_v  * volts outside the band at house i's node
            + w_infeas * |requested - executed| battery energy of house i
    shared  = w_net_v  * volts outside the band summed over all nodes
            + w_line   * amps above the cable limit summed over all cables
            + w_grid   * kW above the grid-tie rating
            + w_coll   * (power flow does not converge)
            + w_block  * (safety layer replaced the action)

Every network term is evaluated on the PROPOSED action's power flow, i.e.
what would have happened had the safety layer not stepped in. Cost is
evaluated on what was executed. Penalty weights are in config `reward`.
"""

from dataclasses import dataclass

import numpy as np

from p2p import safety
from p2p.config import RewardConfig
from p2p.powerflow import DCNetwork, FlowResult


@dataclass(frozen=True)
class Penalty:
    own_aud: np.ndarray     # (N,)
    shared_aud: float
    parts: dict             # name -> AUD (shared) or (N,) AUD (own), for logging

    @property
    def total_aud(self) -> np.ndarray:
        return self.own_aud + self.shared_aud


def penalties(w: RewardConfig, net: DCNetwork, proposed: FlowResult, limits: safety.Limits,
              requested_kw, executed_kw, blocked: bool, dt_hours: float) -> Penalty:
    n = len(net.house_nodes)
    infeasible = w.infeasible_aud_per_kwh.value * np.abs(np.asarray(requested_kw) - executed_kw) * dt_hours

    if proposed.converged:
        v_excess = safety.voltage_excess(proposed.v_node, limits)
        own_v = w.own_voltage_aud_per_v.value * v_excess[net.house_nodes]
        net_v = w.network_voltage_aud_per_v.value * float(v_excess.sum())
        line = w.line_overload_aud_per_a.value * float(safety.line_excess(proposed.i_edge, limits).sum())
        grid = w.grid_overload_aud_per_kw.value * safety.grid_excess(proposed.grid_kw, limits)
        collapse = 0.0
    else:
        own_v, net_v, line, grid = np.zeros(n), 0.0, 0.0, 0.0
        collapse = w.collapse_aud.value
    block = w.blocked_aud.value if blocked else 0.0

    shared = net_v + line + grid + collapse + block
    parts = {"own_voltage": own_v, "infeasible": infeasible, "network_voltage": net_v,
             "line_overload": line, "grid_overload": grid, "collapse": collapse, "blocked": block}
    return Penalty(own_aud=own_v + infeasible, shared_aud=shared, parts=parts)
