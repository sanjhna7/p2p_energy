import unittest
from dataclasses import replace

import numpy as np

from p2p import baselines, metrics
from p2p.config import Param, load_config
from p2p.scenario import load_scenario

CFG = load_config()
HAVE_DATA = (CFG.processed_dir / "houses_timeseries.csv.gz").exists()


def _cfg(**network):
    s = replace(CFG.scenario, n_houses=Param(5, "SCENARIO"), days=Param(2, "SCENARIO"))
    n = CFG.network
    for k, v in network.items():
        n = replace(n, **{k: Param(v, "PENDING", note="test")})
    return replace(CFG, scenario=s, network=n)


@unittest.skipUnless(HAVE_DATA, "run `python -m data_pipeline.build` first")
class EnvTest(unittest.TestCase):
    def test_every_house_balances_every_step(self):
        cfg = _cfg()
        log = baselines.run("rule_p2p", load_scenario(cfg), cfg)
        lhs = log["pv_kw"] - log["load_kw"] + log["battery_kw"]
        rhs = log["p2p_sold_kw"] - log["p2p_bought_kw"] + log["grid_export_kw"] - log["grid_import_kw"]
        np.testing.assert_allclose(lhs, rhs, atol=1e-9)

    def test_ordering_of_baselines(self):
        cfg = _cfg()
        sc = load_scenario(cfg)
        logs = {p: baselines.run(p, sc, cfg) for p in baselines.POLICIES}
        cost = {p: logs[p]["cost_aud"].sum() for p in logs}
        self.assertLessEqual(cost["rule_p2p"], cost["battery_only"] + 1e-9)
        self.assertLessEqual(cost["battery_only"], cost["grid_only"] + 1e-9)
        s = metrics.summarise(logs["rule_p2p"], sc.dt_hours, baseline=logs["grid_only"])
        self.assertEqual(s["houses_worse_off"], 0)

    def test_safety_blocks_actions_when_limits_are_tight(self):
        # A +-1 mV band: only steps with no cable flow at all can pass.
        cfg = _cfg(voltage_min_v=379.999, voltage_max_v=380.001)
        log = baselines.run("rule_p2p", load_scenario(cfg), cfg)
        blocked = log["blocked"]
        self.assertTrue(blocked.any())
        self.assertTrue(np.all(log["battery_kw"][blocked] == 0))
        ok = ~blocked
        self.assertTrue(np.all(log["v_min"][ok] >= 379.999) and np.all(log["v_max"][ok] <= 380.001))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_DATA, "run `python -m data_pipeline.build` first")
class RewardTest(unittest.TestCase):
    def test_reward_is_minus_cost_minus_penalty(self):
        cfg = _cfg()
        log = baselines.run("rule_p2p", load_scenario(cfg), cfg)
        np.testing.assert_allclose(log["reward"], -log["cost_aud"] - log["penalty_aud"])

    def test_idle_safe_policy_has_no_penalty(self):
        cfg = _cfg()
        log = baselines.run("grid_only", load_scenario(cfg), cfg)
        self.assertEqual(float(np.abs(log["penalty_aud"]).sum()), 0.0)

    def test_blocked_proposals_are_penalised(self):
        cfg = _cfg(voltage_min_v=379.999, voltage_max_v=380.001)
        log = baselines.run("rule_p2p", load_scenario(cfg), cfg)
        blocked = log["blocked"]
        w = cfg.reward.blocked_aud.value
        self.assertTrue(np.all(log["penalty_aud"][blocked] >= w))
        # Penalties reflect the proposal, so they exceed the blocked fee alone.
        self.assertTrue(np.any(log["penalty_aud"][blocked] > w))
