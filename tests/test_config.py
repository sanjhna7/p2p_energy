import tempfile
import unittest
from pathlib import Path

import yaml

from p2p.config import DEFAULT_CONFIG_PATH, load_config


def _write(raw):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(raw, f)
    f.close()
    return Path(f.name)


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())

    def test_default_loads_and_every_spec_is_cited(self):
        cfg = load_config()
        for name, _, source, citation in cfg.provenance():
            if source == "SPEC":
                self.assertTrue(citation, name)

    def test_spec_without_ref_is_rejected(self):
        del self.raw["battery"]["usable_energy_kwh"]["ref"]
        with self.assertRaises(ValueError):
            load_config(_write(self.raw))

    def test_unknown_source_is_rejected(self):
        self.raw["battery"]["usable_energy_kwh"]["source"] = "GUESS"
        with self.assertRaises(ValueError):
            load_config(_write(self.raw))

    def test_bare_number_is_rejected(self):
        self.raw["battery"]["usable_energy_kwh"] = 4.0
        with self.assertRaises(ValueError):
            load_config(_write(self.raw))

    def test_unknown_key_is_rejected(self):
        self.raw["battery"]["voltage_droop"] = {"value": 0.5, "source": "SCENARIO"}
        with self.assertRaises(ValueError):
            load_config(_write(self.raw))


if __name__ == "__main__":
    unittest.main()
