import unittest

import numpy as np

from p2p.battery import BatterySpec, step

SPEC = BatterySpec(capacity_kwh=4.0, max_power_kw=3.3, round_trip_efficiency=0.95)
DT = 0.5


class BatteryTest(unittest.TestCase):
    def test_power_is_clipped_to_rating(self):
        self.assertAlmostEqual(float(step(SPEC, 0.5, 10.0, DT).power_kw), 3.3)
        self.assertAlmostEqual(float(step(SPEC, 0.5, -10.0, DT).power_kw), -3.3)

    def test_cannot_discharge_empty_or_charge_full(self):
        self.assertEqual(float(step(SPEC, 0.0, 2.0, DT).power_kw), 0.0)
        self.assertEqual(float(step(SPEC, 1.0, -2.0, DT).power_kw), 0.0)

    def test_full_cycle_returns_round_trip_efficiency(self):
        soc, energy_in = 0.0, 0.0
        while soc < 1.0 - 1e-12:
            s = step(SPEC, soc, -1.0, DT)
            energy_in += -float(s.power_kw) * DT
            soc = float(s.soc)
        energy_out = 0.0
        while soc > 1e-12:
            s = step(SPEC, soc, 1.0, DT)
            energy_out += float(s.power_kw) * DT
            soc = float(s.soc)
        self.assertAlmostEqual(energy_out / energy_in, 0.95, places=9)
        self.assertAlmostEqual(energy_in, 4.0 / np.sqrt(0.95), places=9)

    def test_vectorised_matches_scalar(self):
        socs, reqs = np.array([0.0, 0.3, 1.0]), np.array([-2.0, 1.5, 3.0])
        vec = step(SPEC, socs, reqs, DT)
        for i in range(3):
            one = step(SPEC, socs[i], reqs[i], DT)
            self.assertAlmostEqual(float(vec.power_kw[i]), float(one.power_kw))
            self.assertAlmostEqual(float(vec.soc[i]), float(one.soc))


if __name__ == "__main__":
    unittest.main()
