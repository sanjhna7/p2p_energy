import unittest
from dataclasses import replace

import numpy as np
import torch

from p2p.config import Param, load_config
from p2p.env import CommunityEnv
from p2p.gat import N_FEATURES, ActorCritic, GraphContext
from p2p.marl import GATPolicy, gae, training_scenario
from p2p.scenario import load_scenario

CFG = load_config()
HAVE_DATA = (CFG.processed_dir / "houses_timeseries.csv.gz").exists()


def _cfg(n_houses=5, days=1):
    return replace(CFG, scenario=replace(CFG.scenario, n_houses=Param(n_houses, "SCENARIO"),
                                         days=Param(days, "SCENARIO")))


class GAETest(unittest.TestCase):
    def test_single_step_advantage(self):
        r = np.array([[1.0, 2.0]], dtype=np.float32)
        v = np.array([[0.5, 0.5]], dtype=np.float32)
        adv, ret = gae(r, v, gamma=0.9, lam=0.95)
        # Cut-off episode: bootstrap with the last value, delta = r + 0.9 v - v.
        np.testing.assert_allclose(adv, r + 0.9 * v - v, rtol=1e-6)
        np.testing.assert_allclose(ret, adv + v, rtol=1e-6)


@unittest.skipUnless(HAVE_DATA, "run `python -m data_pipeline.build` first")
class GATTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = _cfg()
        self.sc = load_scenario(self.cfg)
        self.ctx = GraphContext(self.sc, self.cfg)
        self.env = CommunityEnv(self.sc, self.cfg)

    def test_features_shape_and_house_rows(self):
        x = self.ctx.features(self.env.observe())
        self.assertEqual(x.shape, (len(self.sc.nodes), N_FEATURES))
        self.assertTrue(np.all(x[self.ctx.house_nodes, 0] == 1))
        np.testing.assert_allclose(x[self.ctx.house_nodes, 3] * self.ctx.p_scale, self.sc.pv_kw[0], rtol=1e-6)

    def test_batched_graph_equals_single_graphs(self):
        model = ActorCritic()
        x1 = torch.as_tensor(self.ctx.features(self.env.observe()))
        self.env.step(np.zeros(5))
        x2 = torch.as_tensor(self.ctx.features(self.env.observe()))
        m = self.ctx.n_nodes
        single = [model(x, self.ctx.edge_index, 1, m, self.ctx.house_nodes)[0] for x in (x1, x2)]
        batched = model(torch.cat([x1, x2]), self.ctx.batch_edges(2), 2, m, self.ctx.house_nodes)[0]
        torch.testing.assert_close(batched, torch.cat(single))

    def test_policy_requests_stay_within_battery_rating(self):
        policy = GATPolicy(ActorCritic(), self.ctx)
        obs = self.env.reset()
        while not self.env.done:
            req = policy(obs)
            self.assertTrue(np.all(np.abs(req) <= self.ctx.p_scale + 1e-9))
            obs, _, _, _ = self.env.step(req)

    def test_training_window_does_not_overlap_evaluation(self):
        sc = training_scenario(CFG)
        self.assertLess(sc.timestamps[-1], load_scenario(CFG).timestamps[0])
        bad = replace(CFG, training=replace(CFG.training, start=Param("2011-01-05", "SCENARIO")))
        with self.assertRaises(ValueError):
            training_scenario(bad)


if __name__ == "__main__":
    unittest.main()
