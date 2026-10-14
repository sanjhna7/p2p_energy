"""Community environment: the closed loop of slide 15.

One step = one 30-minute interval for every house at once:

  1. local state     real PV and load (Ausgrid), battery SoC
  2. action          each agent requests a battery power (kW, > 0 discharge)
  3. battery         the request is limited by the battery physics
  4. safety          DC power flow of the resulting injections; if a limit
                     is broken, the action is blocked and all batteries hold
  5. execution       the approved action is applied
  6. market          residuals cleared P2P at the MMR, the rest with the grid
  7. next state      new SoC, voltages, currents

Reward per agent is minus its electricity cost for the interval (AUD).
Every step is logged so metrics.py can compute the evaluation measures.
"""

from dataclasses import dataclass, field

import numpy as np

from p2p import battery, market, powerflow, safety
from p2p.config import Config
from p2p.scenario import Scenario


@dataclass
class StepLog:
    """Per-step arrays, (T, N) for houses and (T,) for the community."""
    rows: dict = field(default_factory=lambda: {k: [] for k in KEYS})

    def add(self, **values):
        for k in KEYS:
            self.rows[k].append(values[k])

    def as_arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.rows.items()}


KEYS = [
    "pv_kw", "load_kw", "battery_kw", "soc", "residual_kw",
    "p2p_sold_kw", "p2p_bought_kw", "grid_export_kw", "grid_import_kw", "cost_aud",
    "v_house", "v_min", "v_max", "i_max_abs", "loss_kw", "grid_kw",
    "blocked", "inherent_violation", "converged",
]


class CommunityEnv:
    def __init__(self, sc: Scenario, cfg: Config, p2p: bool = True):
        self.sc, self.cfg, self.p2p = sc, cfg, p2p
        self.spec = battery.BatterySpec.from_config(cfg.battery)
        self.net = powerflow.DCNetwork.from_scenario(sc, cfg.network.bus_voltage_v.value)
        self.limits = safety.Limits.from_config(cfg.network)
        self.n_houses = sc.load_kw.shape[1]
        self.reset()

    def reset(self):
        self.t = 0
        self.soc = np.full(self.n_houses, self.cfg.battery.initial_soc.value)
        self.log = StepLog()
        return self.observe()

    @property
    def done(self) -> bool:
        return self.t >= len(self.sc.timestamps)

    def observe(self) -> dict:
        """What every agent can see at the start of the current interval."""
        t = min(self.t, len(self.sc.timestamps) - 1)
        return {"t": t, "pv_kw": self.sc.pv_kw[t], "load_kw": self.sc.load_kw[t], "soc": self.soc.copy()}

    def _apply(self, request_kw):
        step = battery.step(self.spec, self.soc, request_kw, self.sc.dt_hours)
        residual = self.sc.net_power_kw[self.t] + step.power_kw
        return step, residual, powerflow.solve(self.net, residual)

    def step(self, request_kw):
        if self.done:
            raise RuntimeError("episode is over; call reset()")

        step, residual, flow = self._apply(request_kw)
        blocked = inherent = False
        if safety.violations(flow, self.limits):
            blocked = True
            step, residual, flow = self._apply(np.zeros(self.n_houses))
            inherent = bool(safety.violations(flow, self.limits))

        self.soc = step.soc
        clearing = market.clear(residual, self.cfg.tariff, self.sc.dt_hours, p2p=self.p2p)
        self.log.add(
            pv_kw=self.sc.pv_kw[self.t], load_kw=self.sc.load_kw[self.t],
            battery_kw=step.power_kw, soc=step.soc, residual_kw=residual,
            p2p_sold_kw=clearing.p2p_sold_kw, p2p_bought_kw=clearing.p2p_bought_kw,
            grid_export_kw=clearing.grid_export_kw, grid_import_kw=clearing.grid_import_kw,
            cost_aud=clearing.cost_aud,
            v_house=flow.v_house(self.net), v_min=flow.v_node.min(), v_max=flow.v_node.max(),
            i_max_abs=np.abs(flow.i_edge).max(), loss_kw=flow.loss_kw, grid_kw=flow.grid_kw,
            blocked=blocked, inherent_violation=inherent, converged=flow.converged,
        )
        self.t += 1
        return self.observe(), -clearing.cost_aud, self.done, {"blocked": blocked, "inherent": inherent}
