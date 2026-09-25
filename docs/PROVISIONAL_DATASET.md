# Provisional MARL development dataset

## Research statement

> The preliminary MARL dataset uses real temporal household PV generation
> and consumption profiles derived from the Ausgrid Solar Home Electricity
> Dataset. Electrical variables unavailable in the source dataset,
> including battery state-of-charge, DC-bus voltage, converter/tie power
> and current, are generated using a simplified physically consistent
> nanogrid model for algorithm-development purposes. **These provisional
> variables will subsequently be replaced by measurements from the
> completed Simulink/hardware model.**

This distinction must survive into any paper written on this work. It is
enforced in three places: the `provenance` field of every column in
`data/processed/dataset_metadata.json`, the table below, and
`nanogrid/provenance.py`.

## Why this exists

The hardware team's Simulink model of the DC nanogrid is not finished.
Rather than wait, this package produces a stand-in dataset that keeps the
one thing that cannot be invented — the **real temporal behaviour of
household PV generation and demand** — and approximates only the
electrical variables the source dataset never contained.

```
REAL AUSGRID PV / LOAD BEHAVIOUR
             |
simple physically consistent energy model
             |
temporary full electrical state
             |
      MARL development
```

Nothing here is randomly generated. There is no random number generator
anywhere in `nanogrid/`. Every column is a deterministic function of the
real Ausgrid series and the configuration.

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

Each house is one MARL agent. The community DC bus is shared: it has one
voltage per timestep, observed identically by every connected house.

## Source data and unit conversion

Source: `data/raw/Solar home 2010-2011.csv` (Ausgrid Solar Home Electricity Data).

| Ausgrid channel | Used as |
| --- | --- |
| `GG` | gross PV generation |
| `GC` + `CL` | household consumption (general + controlled load) |
| `Generator Capacity` | installed PV rating, used to normalise the profile |

The CSV stores **energy**: each of the 48 daily columns is the kWh in the
half hour *ending* at that column's time. Everything downstream works in
average **power**:

```
P(kW) = E(kWh) / interval_hours = E(kWh) / 0.5 = 2 x E(kWh)
```

The project's SQLite database already holds `meter_readings` in average
kW (`removedfornnow/ingest.py` applied exactly this conversion), so the database path
applies no further scaling; the CSV fallback applies it explicitly. Both
paths have been verified to produce bit-identical output.

## PV scaling

The real generation profile is normalised by that customer's own
installed capacity and re-scaled to the provisional hardware array:

```
pv_pu    = ausgrid_pv_kw / ausgrid_generator_capacity_kwp
pv_power = pv_pu * pv.capacity_kw
```

This is a re-scaling, not a re-invention: the per-unit series is the
household's real, weather-driven, seasonal generation shape and only its
magnitude changes. **No generic clear-sky or solar-geometry equation is
used anywhere.** Validation check 13 asserts the correlation with the
source profile is 1.0 and that the scale factor is exactly
`pv.capacity_kw / generator_capacity`.

## Sign conventions

Used everywhere, without exception:

| Quantity | Convention |
| --- | --- |
| `net_power` | `pv_power - load_power` |
| `battery_power` | **> 0 discharging**, < 0 charging |
| `tie_power` | **> 0 house → community** (export), < 0 community → house (import) |
| `tie_current` | **magnitude only**; the direction is carried by `tie_power`'s sign |

Units: power in kW, energy in kWh, voltage in V, current in A — so every
current carries a factor of 1000, `I(A) = 1000 * P(kW) / V(V)`.

## The provisional model

Each timestep, per house, **in this order** — the order is the energy
flow, and it is why the columns are coupled rather than independent:

1. `pv_power`, `load_power` from the real Ausgrid series (above).
2. `net_power = pv_power - load_power`.
3. **Battery** reacts to `net_power` under greedy self-consumption:
   surplus charges it first, deficit discharges it first, both bounded by
   `max_charge_kw` / `max_discharge_kw` and by the `[min_soc, max_soc]`
   energy window.

   ```
   charging:    SOC(t+1) = SOC(t) + (P_charge * eff * dt) / capacity
   discharging: SOC(t+1) = SOC(t) - (P_discharge * dt) / (capacity * eff)
   ```

4. **Bus tie** carries whatever the battery did not absorb, limited by
   `tie.max_power_kw`.
5. **Local bus voltage** droops on the imported power:
   `V_local = V_nominal - virtual_droop * P_imported`, clipped to
   `[voltage_min, voltage_max]`.
6. **Community bus voltage**, once per timestep for the whole community:
   `V_comm = V_nominal - community_droop * (total_import - total_export)`,
   clipped to its own limits.
7. Currents follow from the powers by `P = V x I`.

So `PV = 4 kW, Load = 2 kW` gives `net = +2 kW`, of which the battery
takes what it can and the tie exports the rest. It can never produce a
battery discharging while the tie imports.

### Slack terms

Two columns exist purely so energy conservation is never violated
silently, and are zero unless the bus-tie rating binds:

- `curtailed_power` — surplus the converter could not export.
- `unserved_power` — deficit the converter could not import.

At community level the same honesty applies. Real households rarely
balance each other exactly, so `data/processed/community_state.csv`
records an explicit external-grid `grid_slack_kw`:

```
delivered = total_export * (1 - network.loss_fraction)
slack     = total_import - delivered      # > 0 community imports from the grid
```

## Provenance of every column

`ausgrid_derived` = real temporal behaviour from Ausgrid, with only a
documented scaling/mapping. `provisional_simulation` = produced by the
model in `nanogrid/models.py`, **to be replaced by Simulink/hardware
output**.

| Column | Provenance |
| --- | --- |
| `timestamp` | `ausgrid_derived` |
| `ausgrid_customer_id` | `ausgrid_derived` |
| `pv_power` | `ausgrid_derived` (+ hardware scaling) |
| `load_power` | `ausgrid_derived` (+ hardware mapping) |
| `net_power`, `surplus_power`, `deficit_power` | `ausgrid_derived` |
| `house_id` | `provisional_simulation` |
| `pv_voltage`, `pv_current` | `provisional_simulation` |
| `load_voltage`, `load_current` | `provisional_simulation` |
| `battery_voltage`, `battery_current`, `battery_power`, `battery_soc` | `provisional_simulation` |
| `battery_charge_power`, `battery_discharge_power` | `provisional_simulation` |
| `local_bus_voltage`, `local_bus_current` | `provisional_simulation` |
| `community_bus_voltage` | `provisional_simulation` |
| `tie_power`, `tie_current`, `energy_exchange` | `provisional_simulation` |
| `curtailed_power`, `unserved_power` | `provisional_simulation` |

The community topology in `data/processed/topology.json` is also
`provisional_simulation`: Ausgrid contains no network connectivity, and
customer ids carry **no** spatial meaning — customer 1 and customer 2 are
not neighbours. The layout is our own assumption, stored separately from
the time series so it can be swapped without regenerating any data.

## MARL observation

The dataset holds the complete state; the environment shows an agent only

```
[pv_power, load_power, battery_soc,
 local_bus_voltage, tie_power, community_bus_voltage]
```

To change the observation space, edit `observation.features` in
`config/nanogrid_config.yaml` — nothing else. `nanogrid/observation.py`
is the single place that knows the observation shape, and
`observations_at(dataset, timestamp, features)` returns the
`(n_houses, n_features)` joint observation a MARL `step()` would emit.

## Running it

```bash
python build_marl_dataset.py            # generate + validate + plot
python -m nanogrid.build_dataset        # generate only
python -m nanogrid.validate             # validate an existing dataset
python -m nanogrid.plots                # re-render the plots
```

Point any of them at an alternative config with `--config path.yaml`.

## Replacing this with the real hardware model

When the Simulink model is ready, the swap-in point is
`nanogrid/models.py::step_house`, and only that. Its contract is:

```
(ausgrid_pv_kw, ausgrid_load_kw, ausgrid_capacity_kwp, prev_soc, cfg, dt)
    -> HouseState
```

Replace its body with the simulation outputs, flip the affected entries
in `nanogrid/provenance.py` from `provisional_simulation` to whatever the
hardware model is called, and rerun. The MARL environment, the
observation spec, the topology and every validation check keep working
unchanged.

## Relationship to the earlier SQLite pipeline

This package **reads** `meter_readings` from `data/raw/p2p_energy.db` and
writes nothing back to it.

The earlier SQLite pipeline — `main.py`, `ingest.py`, `database.py`,
`simulation.py`, `state.py`, `grid_model.py` — is not used by this
experiment and now lives in `removedfornnow/`, intact and still runnable;
see `removedfornnow/README.md`. `ingest.py` in particular is what built
`data/raw/p2p_energy.db` in the first place, and is how you would rebuild
it.

The one piece of that code reused directly is
`battery_model.step_battery`, which moved into this package as
`nanogrid/battery_model.py` so the repository holds a single set of
charge/discharge equations.
