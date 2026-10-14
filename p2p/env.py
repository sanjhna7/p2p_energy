"""Community environment: the closed loop of slide 15.

One step = one 30-minute interval for every house at once:

  1. local state     real PV and load (Ausgrid), battery SoC, last voltages
  2. action          each agent requests a battery power (kW, > 0 discharge)
  3. battery         the request is limited by the battery physics
  4. safety          DC power flow of the resulting injections; if a limit
                     is broken, the action is blocked and all batteries hold
  5. execution       the approved action is applied
  6. market          residuals cleared P2P at the MMR, the rest with the grid
  7. next state      new SoC, voltages, currents

Reward per agent is minus its cost minus penalties for what it proposed
(see p2p/reward.py). Every step is logged for metrics.py.
"""

import numpy as np

from p2p import battery, market, powerflow, reward, safety
from p2p.config import Config
from p2p.scenario import Scenario

KEYS = [
    "pv_kw", "load_kw", "requested_kw", "battery_kw", "soc", "residual_kw",
    "p2p_sold_kw", "p2p_bought_kw", "grid_export_kw", "grid_import_kw", "cost_aud",
    "penalty_aud", "reward",
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

    def reset(self, t0: int = 0, horizon: int = None, soc=None):
        """Start an episode at step t0 for `horizon` steps (default: to the end)."""
        total = len(self.sc.timestamps)
        self.t0 = t0
        self.t = t0
        self.t_end = total if horizon is None else min(total, t0 + horizon)
        init = self.cfg.battery.initial_soc.value if soc is None else soc
        self.soc = np.broadcast_to(np.asarray(init, dtype=float), (self.n_houses,)).copy()
        self.v_node = np.full(len(self.net.parent), self.net.v_slack)
        self.log = {k: [] for k in KEYS}
        return self.observe()

    @property
    def done(self) -> bool:
        return self.t >= self.t_end

    def observe(self) -> dict:
        """What the agents can see at the start of the current interval."""
        t = min(self.t, len(self.sc.timestamps) - 1)
        return {
            "t": t,
            "hour": self.sc.timestamps[t].hour + self.sc.timestamps[t].minute / 60,
            "pv_kw": self.sc.pv_kw[t],
            "load_kw": self.sc.load_kw[t],
            "soc": self.soc.copy(),
            "v_node": self.v_node.copy(),   # voltages after the previous interval
        }

    def _apply(self, request_kw):
        step = battery.step(self.spec, self.soc, request_kw, self.sc.dt_hours)
        residual = self.sc.net_power_kw[self.t] + step.power_kw
        return step, residual, powerflow.solve(self.net, residual)

    def step(self, request_kw):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        request_kw = np.asarray(request_kw, dtype=float)

        step, residual, proposed = self._apply(request_kw)
        blocked = bool(safety.violations(proposed, self.limits))
        executed, flow, inherent = step, proposed, False
        if blocked:
            executed, residual, flow = self._apply(np.zeros(self.n_houses))
            inherent = bool(safety.violations(flow, self.limits))

        pen = reward.penalties(self.cfg.reward, self.net, proposed, self.limits,
                               request_kw, step.power_kw, blocked, self.sc.dt_hours)
        clearing = market.clear(residual, self.cfg.tariff, self.sc.dt_hours, p2p=self.p2p)
        r = -clearing.cost_aud - pen.total_aud

        self.soc = executed.soc
        if flow.converged:
            self.v_node = flow.v_node
        self._record(
            pv_kw=self.sc.pv_kw[self.t], load_kw=self.sc.load_kw[self.t],
            requested_kw=request_kw, battery_kw=executed.power_kw, soc=executed.soc, residual_kw=residual,
            p2p_sold_kw=clearing.p2p_sold_kw, p2p_bought_kw=clearing.p2p_bought_kw,
            grid_export_kw=clearing.grid_export_kw, grid_import_kw=clearing.grid_import_kw,
            cost_aud=clearing.cost_aud, penalty_aud=pen.total_aud, reward=r,
            v_house=flow.v_house(self.net), v_min=flow.v_node.min(), v_max=flow.v_node.max(),
            i_max_abs=np.abs(flow.i_edge).max(), loss_kw=flow.loss_kw, grid_kw=flow.grid_kw,
            blocked=blocked, inherent_violation=inherent, converged=flow.converged,
        )
        self.t += 1
        return self.observe(), r, self.done, {"blocked": blocked, "inherent": inherent, "penalty": pen}

    def _record(self, **values):
        for k in KEYS:
            self.log[k].append(values[k])

    def log_arrays(self) -> dict:
        """Per-step arrays: (T, N) for houses, (T,) for the community."""
        return {k: np.asarray(v) for k, v in self.log.items()}
