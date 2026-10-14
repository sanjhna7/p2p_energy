import unittest
from dataclasses import replace

import numpy as np

from p2p.config import Param, load_config
from p2p.scenario import load_scenario

CFG = load_config()
HAVE_DATA = (CFG.processed_dir / "houses_timeseries.csv.gz").exists()


def _with(n_houses=None, days=None):
    s = CFG.scenario
    if n_houses is not None:
        s = replace(s, n_houses=Param(n_houses, "SCENARIO"))
    if days is not None:
        s = replace(s, days=Param(days, "SCENARIO"))
    return replace(CFG, scenario=s)


@unittest.skipUnless(HAVE_DATA, "run `python -m data_pipeline.build` first")
class ScenarioTest(unittest.TestCase):
    def test_shapes_and_window(self):
        sc = load_scenario(_with(n_houses=10, days=2))
        self.assertEqual(sc.load_kw.shape, (96, 10))
        self.assertEqual(sc.pv_kw.shape, (96, 10))
        self.assertEqual(sc.dt_hours, 0.5)
        np.testing.assert_allclose(sc.net_power_kw, sc.pv_kw - sc.load_kw)

    def test_subgraph_is_a_tree_rooted_at_the_transformer(self):
        for n in (1, 5, 10, 55):
            sc = load_scenario(_with(n_houses=n, days=1))
            self.assertEqual(len(sc.edges), len(sc.nodes) - 1)
            self.assertEqual(sc.nodes.loc[0, "node_type"], "substation_lv")
            self.assertEqual(len(set(sc.house_local_ids)), n)

    def test_whole_feeder(self):
        sc = load_scenario(_with(n_houses=55, days=1))
        self.assertEqual(len(sc.nodes), 110)
        self.assertEqual(len(sc.edges), 109)

    def test_too_many_houses_is_rejected(self):
        with self.assertRaises(ValueError):
            load_scenario(_with(n_houses=56, days=1))


if __name__ == "__main__":
    unittest.main()
