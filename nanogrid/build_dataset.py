"""
Provisional MARL dataset generator.

    REAL AUSGRID PV / LOAD BEHAVIOUR
                 |
    simple physically consistent energy model
                 |
    temporary full electrical state
                 |
          MARL development

Run:  python -m nanogrid.build_dataset

Writes (paths configurable):
    data/processed/marl_simulation_dataset.csv   full simulation state
    data/processed/agent_observations.csv        MARL observation vectors
    data/processed/community_state.csv           shared community bus state
    data/processed/topology.json                 community graph
    data/processed/dataset_metadata.json         provenance + config used

The time loop steps forward in real Ausgrid order and carries battery
state of charge from one step to the next, so the dataset is a single
coherent trajectory - not a table of independently sampled rows.
"""

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from . import models, observation, provenance, topology
from .config import DEFAULT_CONFIG_PATH, load_config
from .ausgrid_source import load_profiles

# Column order of the generated dataset. Identity first, then the
# measured-in-hardware state, then the derived quantities.
DATASET_COLUMNS = [
    "timestamp",
    "house_id",
    "ausgrid_customer_id",

    "pv_voltage", "pv_current", "pv_power",
    "load_voltage", "load_current", "load_power",
    "battery_voltage", "battery_current", "battery_power", "battery_soc",
    "local_bus_voltage", "local_bus_current",
    "community_bus_voltage",
    "tie_current", "tie_power",

    "net_power", "surplus_power", "deficit_power",
    "battery_charge_power", "battery_discharge_power", "energy_exchange",
    "curtailed_power", "unserved_power",
]

COMMUNITY_COLUMNS = [
    "timestamp",
    "community_bus_voltage",
    "total_export_kw",
    "total_import_kw",
    "network_loss_kw",
    "grid_slack_kw",
    "aggregate_imbalance_kw",
]


def _assign_houses(profiles, cfg):
    """Map each selected real customer onto a house/agent id.

    house_id is this simulation's agent index; ausgrid_customer_id keeps
    every house traceable back to the real profile behind it.
    """
    customers = profiles.customer_ids
    return [
        {
            "house_id": f"house_{i}",
            "node": i,
            "ausgrid_customer_id": c,
            "ausgrid_capacity_kwp": profiles.capacity_kwp[c],
        }
        for i, c in enumerate(customers)
    ]


def generate(cfg):
    """Build the provisional dataset. Returns (dataset, community, houses)."""

    profiles = load_profiles(cfg)
    houses = _assign_houses(profiles, cfg)
    dt_hours = cfg.source.interval_hours

    # Wide lookup: index = timestamp, columns = customer. Keeps the real
    # temporal order intact and makes each step a cheap row read.
    pv_wide = profiles.frame.pivot(
        index="timestamp", columns="customer_id", values="ausgrid_pv_kw"
    ).sort_index()
    load_wide = profiles.frame.pivot(
        index="timestamp", columns="customer_id", values="ausgrid_load_kw"
    ).sort_index()

    timestamps = pv_wide.index

    # Battery state carried across timesteps - this is what couples the
    # trajectory together.
    soc = {h["house_id"]: cfg.battery.initial_soc for h in houses}

    rows = []
    community_rows = []

    for ts in timestamps:
        pv_now = pv_wide.loc[ts]
        load_now = load_wide.loc[ts]

        step_states = []
        for house in houses:
            customer = house["ausgrid_customer_id"]

            state = models.step_house(
                ausgrid_pv_kw=float(pv_now[customer]),
                ausgrid_load_kw=float(load_now[customer]),
                ausgrid_capacity_kwp=house["ausgrid_capacity_kwp"],
                prev_soc=soc[house["house_id"]],
                cfg=cfg,
                dt_hours=dt_hours,
            )

            soc[house["house_id"]] = state.battery_soc
            step_states.append((house, state))

        # The community bus is shared: computed once per timestep from
        # every house's tie power, then written to all of them.
        comm = models.community_step(
            [s.tie_power for _, s in step_states],
            cfg.bus,
            cfg.network.loss_fraction,
        )

        for house, state in step_states:
            row = state.as_dict()
            row.update(
                timestamp=ts,
                house_id=house["house_id"],
                ausgrid_customer_id=house["ausgrid_customer_id"],
                community_bus_voltage=comm.voltage,
            )
            rows.append(row)

        community_rows.append(
            {
                "timestamp": ts,
                "community_bus_voltage": comm.voltage,
                "total_export_kw": comm.total_export_kw,
                "total_import_kw": comm.total_import_kw,
                "network_loss_kw": comm.network_loss_kw,
                "grid_slack_kw": comm.slack_kw,
                "aggregate_imbalance_kw": comm.aggregate_imbalance_kw,
            }
        )

    dataset = pd.DataFrame(rows)[DATASET_COLUMNS]
    community = pd.DataFrame(community_rows)[COMMUNITY_COLUMNS]

    return dataset, community, houses, profiles


def _metadata(cfg, dataset, houses, profiles, features):
    per_agent, joint = observation.observation_shape(features, len(houses))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "research_statement": provenance.RESEARCH_STATEMENT,
        "source_dataset": "Ausgrid Solar Home Electricity Data 2010-2011",
        "source_loaded_from": profiles.origin,
        "unit_conversion": (
            "Ausgrid stores kWh per half-hour interval; "
            "P(kW) = E(kWh) / 0.5 = 2 x E(kWh)."
        ),
        "sign_conventions": {
            "battery_power": "> 0 discharging, < 0 charging",
            "tie_power": "> 0 house -> community, < 0 community -> house",
            "tie_current": "magnitude only; direction is carried by tie_power",
            "net_power": "pv_power - load_power",
        },
        "rows": int(len(dataset)),
        "timesteps": int(dataset["timestamp"].nunique()),
        "interval_hours": cfg.source.interval_hours,
        "n_houses": len(houses),
        "houses": [
            {k: h[k] for k in ("house_id", "node", "ausgrid_customer_id",
                               "ausgrid_capacity_kwp")}
            for h in houses
        ],
        "observation": {
            "features": list(features),
            "per_agent_shape": [per_agent],
            "joint_shape": [len(houses), per_agent],
            "flattened_joint_dim": joint,
        },
        "state": {
            "columns": DATASET_COLUMNS,
            "per_house_state_dim": len(DATASET_COLUMNS) - 3,
            "joint_state_shape": [len(houses), len(DATASET_COLUMNS) - 3],
        },
        "provenance_summary": provenance.summary(DATASET_COLUMNS),
        "provenance": provenance.provenance_for(DATASET_COLUMNS),
        "config_used": cfg.raw,
    }


def build(config_path=DEFAULT_CONFIG_PATH, verbose=True):
    cfg = load_config(config_path)

    if verbose:
        print("=" * 62)
        print("Provisional MARL dataset generator (Ausgrid-based)")
        print("=" * 62)
        print(f"Config: {cfg.path}")

    dataset, community, houses, profiles = generate(cfg)

    features = observation.validate_features(
        cfg.observation.features, dataset.columns
    )
    observations = observation.observation_frame(dataset, features)

    graph = topology.build_topology(
        [h["house_id"] for h in houses],
        [h["ausgrid_customer_id"] for h in houses],
        cfg.network.topology,
    )

    paths = {
        "dataset": cfg.resolve(cfg.output.dataset_path),
        "observations": cfg.resolve(cfg.output.observations_path),
        "community": cfg.resolve(cfg.output.community_path),
        "metadata": cfg.resolve(cfg.output.metadata_path),
        "topology": cfg.resolve(cfg.network.topology_path),
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(paths["dataset"], index=False)
    observations.to_csv(paths["observations"], index=False)
    community.to_csv(paths["community"], index=False)
    topology.save_topology(graph, paths["topology"])

    meta = _metadata(cfg, dataset, houses, profiles, features)
    with open(paths["metadata"], "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    if verbose:
        print(f"Source          : {profiles.origin}")
        print(f"Houses          : {len(houses)} "
              f"(Ausgrid customers {[h['ausgrid_customer_id'] for h in houses]})")
        print(f"Timesteps       : {dataset['timestamp'].nunique()}")
        print(f"Rows            : {len(dataset)}")
        print(f"Window          : {dataset['timestamp'].min()} -> "
              f"{dataset['timestamp'].max()}")
        print(f"Observation dim : {len(features)} {features}")
        print("\nWritten:")
        for name, p in paths.items():
            print(f"  {name:13s} {p}")

    return dataset, observations, community, meta, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
