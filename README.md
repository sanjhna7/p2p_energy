# P2P Energy — provisional MARL development dataset

Builds a **temporary but physically consistent** dataset for developing
and testing the MARL algorithm, while the Simulink/hardware model is
still being finished.

The temporal behaviour is **real** — household PV generation and
consumption from the Ausgrid Solar Home Electricity Dataset (2010–2011).
The electrical variables that dataset never contained — voltages,
currents, battery state of charge, bus-tie power — come from a simple
nanogrid model and are labelled `provisional_simulation` throughout.
**They will be replaced by Simulink/hardware outputs.** There is no
random number generator anywhere in `nanogrid/`.

Full detail, including the per-column provenance table and the research
statement that must accompany any publication:
**[`docs/PROVISIONAL_DATASET.md`](docs/PROVISIONAL_DATASET.md)**.

## Quick start

```bash
pip install -r requirements.txt
python build_marl_dataset.py          # generate + validate + plot
```

Other entry points:

```bash
python -m nanogrid.build_dataset      # generate only
python -m nanogrid.validate           # validate an existing dataset
python -m nanogrid.plots              # re-render the plots
#  ... --config path/to/other.yaml    # any of them, alternative config
```

## Layout

```
build_marl_dataset.py        Entry point: generate → validate → plot
config/
  nanogrid_config.yaml       EVERY assumption. No magic numbers in Python.
nanogrid/                    The experiment
  config.py                  Typed config loader; rejects unknown keys
  ausgrid_source.py          Real Ausgrid data ONLY — no modelling here
  battery_model.py           Shared battery charge/discharge equations
  models.py                  PV, load, battery, local bus, tie, community
  build_dataset.py           Time loop; carries SoC across timesteps
  observation.py             The single source of truth for obs shape
  topology.py                Community graph → topology.json
  provenance.py              ausgrid_derived vs provisional_simulation
  validate.py                25 consistency checks
  plots.py                   9 visual checks
data/
  raw/                       Inputs: Ausgrid CSV + p2p_energy.db
  processed/                 Generated dataset, metadata, topology, plots/
docs/
  PROVISIONAL_DATASET.md     Model, conventions, provenance, hand-off
removedfornnow/              Earlier SQLite pipeline — see its README
```

## Architecture modelled

```
                 PV
                  |
             Local DC Bus
            /             \
       Battery          DC Load
            \             /
         Bus Tie Converter
                  |
         Community DC Bus
```

Each house is one MARL agent. The community DC bus is shared — one
voltage per timestep, observed identically by every house.

## Outputs

| File | Contents |
| --- | --- |
| `data/processed/marl_simulation_dataset.csv` | Complete per-house simulation state |
| `data/processed/agent_observations.csv` | The MARL observation vectors |
| `data/processed/community_state.csv` | Shared community bus, network loss, grid slack |
| `data/processed/topology.json` | Community graph (stored apart from the time series) |
| `data/processed/dataset_metadata.json` | Provenance, shapes, and the exact config used |
| `data/processed/plots/` | 9 visual-validation figures |

## Observation

```
[pv_power, load_power, battery_soc,
 local_bus_voltage, tie_power, community_bus_voltage]
```

To change it, edit `observation.features` in
`config/nanogrid_config.yaml` — and nothing else.

## Sign conventions

| Quantity | Convention |
| --- | --- |
| `net_power` | `pv_power - load_power` |
| `battery_power` | **> 0 discharging**, < 0 charging |
| `tie_power` | **> 0 house → community**, < 0 community → house |
| `tie_current` | magnitude only; direction lives in `tie_power` |

Power in kW, energy in kWh, voltage in V, current in A.

## Hand-off to the hardware team

The swap-in point is `nanogrid/models.py::step_house`, and only that:

```
(ausgrid_pv_kw, ausgrid_load_kw, ausgrid_capacity_kwp, prev_soc, cfg, dt)
    -> HouseState
```

Replace its body with the Simulink outputs, flip the affected labels in
`nanogrid/provenance.py`, and rerun. The observation spec, topology and
all 25 validation checks keep working unchanged.
