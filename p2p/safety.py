"""Safety verification layer (slide 15, block 5).

Before a joint action is executed, its physical result is computed with
the DC power flow and checked against the network limits:

  - every node voltage within [voltage_min_v, voltage_max_v]
  - every cable current within line_current_max_a, when that limit is set
  - grid-tie power within grid_tie_max_kw, in either direction
  - the power flow converges (no voltage collapse)

Battery limits need no check here: p2p/battery.py cannot exceed them.

If the proposed action fails, it is blocked and replaced by the fallback
action, all batteries holding (0 kW). If the network still fails with no
battery action, the violation comes from household load and PV alone,
which the agents cannot fix; it is recorded as `inherent` so that it is
never counted against a policy.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from p2p.config import NetworkConfig
from p2p.powerflow import FlowResult


@dataclass(frozen=True)
class Limits:
    v_min: float
    v_max: float
    i_max: Optional[float]   # None = reported, not enforced
    grid_max_kw: float

    @classmethod
    def from_config(cls, cfg: NetworkConfig) -> "Limits":
        return cls(cfg.voltage_min_v.value, cfg.voltage_max_v.value,
                   cfg.line_current_max_a.value, cfg.grid_tie_max_kw.value)


def voltage_excess(v, limits: Limits) -> np.ndarray:
    """Volts outside the band at each node (0 inside it)."""
    return np.maximum(limits.v_min - v, 0.0) + np.maximum(v - limits.v_max, 0.0)


def line_excess(i_edge, limits: Limits) -> np.ndarray:
    """Amps above the cable limit on each edge (0 while no limit is set)."""
    if limits.i_max is None:
        return np.zeros_like(i_edge)
    return np.maximum(np.abs(i_edge) - limits.i_max, 0.0)


def grid_excess(grid_kw: float, limits: Limits) -> float:
    return max(abs(grid_kw) - limits.grid_max_kw, 0.0)


def violations(flow: FlowResult, limits: Limits) -> list:
    """Names of the limits this power-flow result breaks (empty = safe)."""
    if not flow.converged:
        return ["no_convergence"]
    found = []
    if flow.v_node.min() < limits.v_min:
        found.append("undervoltage")
    if flow.v_node.max() > limits.v_max:
        found.append("overvoltage")
    if line_excess(flow.i_edge, limits).any():
        found.append("overcurrent")
    if grid_excess(flow.grid_kw, limits) > 0:
        found.append("grid_overload")
    return found
