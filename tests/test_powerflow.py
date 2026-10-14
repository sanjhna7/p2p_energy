import math
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from p2p.powerflow import DCNetwork, solve


def _line(r1_per_edge):
    """Transformer - n1 - n2 - ... with one house at the far end."""
    m = len(r1_per_edge) + 1
    nodes = pd.DataFrame({"local_id": range(m), "node_type": ["substation_lv"] + ["junction"] * (m - 2) + ["house"]})
    edges = pd.DataFrame({"from_local": range(m - 1), "to_local": range(1, m), "r1_ohm": r1_per_edge})
    sc = SimpleNamespace(nodes=nodes, edges=edges, house_local_ids=np.array([m - 1]))
    return DCNetwork.from_scenario(sc, v_slack=380.0)


class PowerFlowTest(unittest.TestCase):
    def test_single_load_matches_closed_form(self):
        # A load P at the end of loop resistance R: V^2 - V0 V + R P = 0.
        net = _line([0.05])
        p_w, r, v0 = 5000.0, 0.1, 380.0
        flow = solve(net, [-p_w / 1000])
        expected = (v0 + math.sqrt(v0**2 - 4 * r * p_w)) / 2
        self.assertTrue(flow.converged)
        self.assertAlmostEqual(flow.v_node[-1], expected, places=5)
        self.assertGreater(flow.i_edge[0], 0)   # load current flows away from the transformer

    def test_export_raises_voltage_and_reverses_current(self):
        flow = solve(_line([0.05, 0.05]), [3.0])
        self.assertGreater(flow.v_node[-1], 380.0)
        self.assertTrue(np.all(flow.i_edge < 0))

    def test_power_balance(self):
        flow = solve(_line([0.02, 0.03, 0.04]), [-4.0])
        self.assertAlmostEqual(flow.grid_kw, 4.0 + flow.loss_kw, places=9)
        self.assertAlmostEqual(flow.grid_kw * 1000, 380.0 * flow.i_edge[0], places=3)

    def test_collapse_is_reported_not_returned_as_numbers(self):
        flow = solve(_line([0.5]), [-500.0])
        self.assertFalse(flow.converged)
        self.assertTrue(np.isnan(flow.v_node).all())


if __name__ == "__main__":
    unittest.main()
