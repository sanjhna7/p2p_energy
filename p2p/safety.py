"""Safety verification layer (slide 15, block 5).

Before a joint action is executed, its physical result is computed with
the DC power flow and checked against the network limits:

  - every node voltage within [voltage_min_v, voltage_max_v]
  - every cable current within line_current_max_a, when that limit is set
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

    @classmethod
    def from_config(cls, cfg: NetworkConfig) -> "Limits":
        return cls(cfg.voltage_min_v.value, cfg.voltage_max_v.value, cfg.line_current_max_a.value)


def violations(flow: FlowResult, limits: Limits) -> list:
    """Names of the limits this power-flow result breaks (empty = safe)."""
    if not flow.converged:
        return ["no_convergence"]
    found = []
    if flow.v_node.min() < limits.v_min:
        found.append("undervoltage")
    if flow.v_node.max() > limits.v_max:
        found.append("overvoltage")
    if limits.i_max is not None and np.abs(flow.i_edge).max() > limits.i_max:
        found.append("overcurrent")
    return found
