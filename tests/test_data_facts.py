"""The pipeline must reproduce facts published by the data owners.

Ausgrid: "Solar home electricity data - notes (August 2014)", summary
statistics table, 2010-11 column, 300-customer sample.
IEEE European LV Test Feeder (2015): 906 buses, 905 lines, 55 loads.
"""

import unittest

from data_pipeline import ausgrid, feeder
from data_pipeline.build import DEFAULT_AUSGRID, DEFAULT_FEEDER

HAVE_RAW = DEFAULT_AUSGRID.exists() and DEFAULT_FEEDER.exists()


@unittest.skipUnless(HAVE_RAW, "raw data missing; see data/README.md")
class PublishedFactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.year = ausgrid.read_wide(DEFAULT_AUSGRID)
        cls.report = ausgrid.quality_report(cls.year)

    def test_ausgrid_2010_11_published_means(self):
        r = self.report
        self.assertEqual(len(r), 300)
        self.assertEqual(round(r["annual_consumption_kwh"].mean()), 6980)
        self.assertEqual(round(r["annual_generation_kwh"].mean()), 2119)
        self.assertEqual(round(r["pv_capacity_kwp"].mean(), 2), 1.68)
        self.assertEqual(round(r["specific_yield_kwh_per_kwp"].mean()), 1253)

    def test_ausgrid_clean_customer_count(self):
        self.assertEqual(int(self.report["included"].sum()), 269)

    def test_power_is_twice_half_hour_energy(self):
        long = ausgrid.to_long(self.year, customers=[1])
        kwh = self.year.wide.query("Customer == 1 and `Consumption Category` == 'GG'")[self.year.time_columns]
        self.assertAlmostEqual(long["pv_power_kw"].sum(), 2 * kwh.to_numpy().sum(), places=6)

    def test_ieee_european_lv_feeder(self):
        lines, loads, coords, _, root = feeder.read_feeder(DEFAULT_FEEDER)
        self.assertEqual(len(lines), 905)
        self.assertEqual(len(coords), 906)
        self.assertEqual(len(loads), 55)
        nodes, edges = feeder.reduce_feeder(feeder.prune_dead_ends(lines, loads, root), loads, coords, root)
        self.assertEqual((len(nodes), len(edges)), (110, 109))
        self.assertEqual(int((nodes["node_type"] == "house").sum()), 55)


if __name__ == "__main__":
    unittest.main()
