# P2P Energy Distribution using GNN and MARL — baseline code

Real household data on a real low-voltage network, for a topology-aware
GAT-MARL framework for peer-to-peer energy sharing.

- **Household load and rooftop PV:** real half-hourly measurements from
  the Ausgrid Solar Home Electricity Data (2010–11).
- **Network:** the IEEE European LV Test Feeder, 55 customer connections.
- **Battery:** BYD Battery-Box Premium LVS 4.0, from the manufacturer datasheet.
- **Prices:** Ausgrid flat-rate retail cap (AER DMO 2026–27) and IPART's
  2026–27 feed-in benchmark; P2P trades at the mid-market rate.
- **DC network:** the feeder's topology and cable resistance as a two-wire
  DC network. Bus voltage and limits are PENDING the EEE specification.

Every number has a stated source. See `docs/dataset_decision.md` for what is
real and what is our adaptation, and `docs/REFERENCES.md` for citations.

## Quick start

```bash
pip install -r requirements.txt
# download raw data: see data/README.md, section 1
python -m data_pipeline.build                 # raw -> data/processed/ (about 15 s)
python -m unittest discover -s tests -t .     # 28 tests, incl. published-fact checks
python -m p2p.run                             # baselines -> results/ (about 5 s)
```

```python
from p2p.config import load_config
from p2p.scenario import load_scenario

cfg = load_config()              # config/p2p.yaml
sc = load_scenario(cfg)          # 10 houses on one feeder branch, 1-14 Jan 2011
sc.load_kw, sc.pv_kw             # (T, n_houses), real kW
sc.edge_index                    # feeder subgraph for the GAT
```

## Layout

```
data_pipeline/          OFFLINE, run once: raw -> data/processed/
  ausgrid.py            Ausgrid CSV -> cleaned per-house kW series
  feeder.py             IEEE European LV feeder -> 110-node graph
  build.py              joins them; writes manifest with hashes
p2p/                    ONLINE: reads only data/processed/
  config.py             typed config; every value carries REAL/SPEC/SCENARIO/PENDING
  scenario.py           picks houses by feeder branch + date window
  battery.py            battery energy model (physics only, no policy)
  market.py             P2P matching at the MMR, remainder with the grid
  powerflow.py          DC power flow (backward/forward sweep) on the feeder
  safety.py             voltage / current / convergence checks
  env.py                closed loop: battery -> safety -> market, logged
  baselines.py          grid_only, battery_only, rule_p2p
  metrics.py            slide-27 measures from a log
  run.py                runs the baselines, writes results/
config/p2p.yaml         all numbers, with sources and references
tests/                  unit tests + checks against published source facts
docs/
  dataset_decision.md   why these sources; every adaptation of ours
  REFERENCES.md         verified citations
data/                   raw/ and processed/ (not committed; see data/README.md)
```

Data flows one way: `data/raw` → `data_pipeline` → `data/processed` → `p2p`.

## Config provenance

Each value in `config/p2p.yaml` is written as

```yaml
usable_energy_kwh:
  value: 4.0
  source: SPEC          # REAL | SPEC | SCENARIO | PENDING
  ref: byd_lvs_2024     # required for SPEC
```

The loader rejects bare numbers, unknown sources, and SPEC values without
a reference.

## Sign conventions

| Quantity | Convention |
| --- | --- |
| `net_power_kw` | `pv − load`; > 0 surplus, < 0 deficit |
| battery `power_kw` | > 0 discharging, < 0 charging |
| residual / injection | > 0 exported into the network, < 0 drawn |
| edge current | > 0 away from the transformer |
| grid power | > 0 the grid supplies the community |

Power in kW, energy in kWh, voltage in V, current in A.
