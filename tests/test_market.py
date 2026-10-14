import unittest

import numpy as np

from p2p.config import load_config
from p2p.market import clear

TARIFF = load_config().tariff


class MarketTest(unittest.TestCase):
    def test_mmr_is_midpoint(self):
        self.assertAlmostEqual(TARIFF.mmr_c_per_kwh, (33.1372 + 4.95) / 2)

    def test_matching_and_energy_balance(self):
        residual = np.array([2.0, 1.0, -0.5, -3.5])
        c = clear(residual, TARIFF, 0.5)
        self.assertAlmostEqual(c.p2p_sold_kw.sum(), 3.0)
        self.assertAlmostEqual(c.p2p_bought_kw.sum(), 3.0)
        np.testing.assert_allclose(c.p2p_sold_kw - c.p2p_bought_kw + c.grid_export_kw - c.grid_import_kw, residual)

    def test_everyone_does_at_least_as_well_as_with_the_grid(self):
        residual = np.array([2.0, -0.5, -3.5, 0.7])
        p2p, grid = clear(residual, TARIFF, 0.5), clear(residual, TARIFF, 0.5, p2p=False)
        self.assertTrue(np.all(p2p.cost_aud <= grid.cost_aud + 1e-12))

    def test_grid_only_prices(self):
        c = clear(np.array([1.0, -1.0]), TARIFF, 0.5, p2p=False)
        np.testing.assert_allclose(c.cost_aud, [-0.5 * 4.95 / 100, 0.5 * 33.1372 / 100])


if __name__ == "__main__":
    unittest.main()
