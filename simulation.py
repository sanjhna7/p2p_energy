"""
Simulation: dataset generation, validation, and visualization.

This module orchestrates the full MARL dataset pipeline:

1. **Generation** — steps through real Ausgrid profiles in temporal
   order, running the DC nanogrid physics model at each timestep to
   produce a coherent trajectory (not independently sampled rows).

2. **Validation** — 25 physics/data consistency checks that prove the
   dataset conserves energy, obeys rate limits, and preserves the real
   Ausgrid temporal profiles.

3. **Visualization** — 9 matplotlib plots for visual validation.

Run:  python main.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt

import data
import environment as env
from config import DEFAULT_CONFIG_PATH, load_config

# =====================================================================
# 1. Dataset generation
# =====================================================================

# Column order of the generated dataset.
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
    """Map each selected real customer onto a house/agent id."""
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
    """Build the provisional dataset. Returns (dataset, community, houses, profiles)."""

    profiles = data.load_profiles(cfg)
    houses = _assign_houses(profiles, cfg)
    dt_hours = cfg.source.interval_hours

    pv_wide = profiles.frame.pivot(
        index="timestamp", columns="customer_id", values="ausgrid_pv_kw"
    ).sort_index()
    load_wide = profiles.frame.pivot(
        index="timestamp", columns="customer_id", values="ausgrid_load_kw"
    ).sort_index()

    timestamps = pv_wide.index

    # Battery state carried across timesteps.
    soc = {h["house_id"]: cfg.battery.initial_soc for h in houses}

    rows = []
    community_rows = []

    for ts in timestamps:
        pv_now = pv_wide.loc[ts]
        load_now = load_wide.loc[ts]

        step_states = []
        for house in houses:
            customer = house["ausgrid_customer_id"]

            state = env.step_house(
                ausgrid_pv_kw=float(pv_now[customer]),
                ausgrid_load_kw=float(load_now[customer]),
                ausgrid_capacity_kwp=house["ausgrid_capacity_kwp"],
                prev_soc=soc[house["house_id"]],
                cfg=cfg,
                dt_hours=dt_hours,
            )

            soc[house["house_id"]] = state.battery_soc
            step_states.append((house, state))

        comm = env.community_step(
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
    per_agent, joint = env.observation_shape(features, len(houses))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "research_statement": data.RESEARCH_STATEMENT,
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
        "provenance_summary": data.provenance_summary(DATASET_COLUMNS),
        "provenance": data.provenance_for(DATASET_COLUMNS),
        "config_used": cfg.raw,
    }


def build_dataset(config_path=DEFAULT_CONFIG_PATH, verbose=True):
    """Full pipeline: generate, write, return."""
    cfg = load_config(config_path)

    if verbose:
        print("=" * 62)
        print("Provisional MARL dataset generator (Ausgrid-based)")
        print("=" * 62)
        print(f"Config: {cfg.path}")

    dataset, community, houses, profiles = generate(cfg)

    features = env.validate_features(
        cfg.observation.features, dataset.columns
    )
    observations = env.observation_frame(dataset, features)

    graph = env.build_topology(
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
    env.save_topology(graph, paths["topology"])

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


# =====================================================================
# 2. Validation
# =====================================================================

class Report:
    """Collects check results and renders them as a table."""

    def __init__(self):
        self.results = []

    def check(self, name, passed, detail=""):
        self.results.append((name, bool(passed), detail))
        return passed

    @property
    def failed(self):
        return [r for r in self.results if not r[1]]

    def render(self):
        width = max(len(n) for n, _, _ in self.results)
        lines = []
        for name, passed, detail in self.results:
            mark = "PASS" if passed else "FAIL"
            line = f"  [{mark}] {name.ljust(width)}"
            if detail:
                line += f"  {detail}"
            lines.append(line)
        return "\n".join(lines)


def _finite(frame, columns):
    values = frame[columns].to_numpy(dtype=float)
    return int(np.isnan(values).sum()), int(np.isinf(values).sum())


def _max_abs(series):
    return float(np.abs(np.asarray(series, dtype=float)).max()) if len(series) else 0.0


def validate(dataset, community, cfg, profiles=None, report=None):
    report = report or Report()
    tol = cfg.validation
    numeric = dataset.select_dtypes("number").columns.tolist()

    # 1-2. No NaN / inf anywhere.
    n_nan, n_inf = _finite(dataset, numeric)
    report.check("1. no NaN values", n_nan == 0, f"{n_nan} found")
    report.check("2. no infinite values", n_inf == 0, f"{n_inf} found")

    # 3-4. PV and load are non-negative by physical definition.
    n_neg_pv = int((dataset["pv_power"] < 0).sum())
    n_neg_load = int((dataset["load_power"] < 0).sum())
    report.check("3. pv_power >= 0", n_neg_pv == 0, f"{n_neg_pv} negative")
    report.check("4. load_power >= 0", n_neg_load == 0, f"{n_neg_load} negative")

    # 5. SoC stays inside [0, 1] and inside the configured window.
    soc = dataset["battery_soc"]
    in_unit = bool(((soc >= 0) & (soc <= 1)).all())
    report.check(
        "5. 0 <= battery_soc <= 1", in_unit,
        f"range [{soc.min():.4f}, {soc.max():.4f}]",
    )
    b = cfg.battery
    in_window = bool(
        ((soc >= b.min_soc - 1e-9) & (soc <= b.max_soc + 1e-9)).all()
    )
    report.check(
        "5b. soc within [min_soc, max_soc]", in_window,
        f"window [{b.min_soc}, {b.max_soc}]",
    )

    # 6. Battery power never exceeds its rate limits.
    over_charge = float((dataset["battery_charge_power"] - b.max_charge_kw).max())
    over_discharge = float(
        (dataset["battery_discharge_power"] - b.max_discharge_kw).max()
    )
    report.check(
        "6. battery within charge/discharge limits",
        over_charge <= tol.power_tolerance_kw
        and over_discharge <= tol.power_tolerance_kw,
        f"max overshoot charge {over_charge:.2e} kW, "
        f"discharge {over_discharge:.2e} kW",
    )
    both = int(
        ((dataset["battery_charge_power"] > 0)
         & (dataset["battery_discharge_power"] > 0)).sum()
    )
    report.check("6b. battery never charges and discharges at once", both == 0,
                 f"{both} rows")

    # 7. Bus-tie converter rating.
    over_tie = float((dataset["tie_power"].abs() - cfg.tie.max_power_kw).max())
    report.check(
        "7. tie power within +/- max_power_kw",
        over_tie <= tol.power_tolerance_kw,
        f"max |tie_power| = {dataset['tie_power'].abs().max():.4f} kW, "
        f"limit {cfg.tie.max_power_kw} kW",
    )

    # 8. Voltage limits.
    v_local = dataset["local_bus_voltage"]
    within = bool(
        ((v_local >= cfg.bus.voltage_min - 1e-9)
         & (v_local <= cfg.bus.voltage_max + 1e-9)).all()
    )
    at_rail = int(
        ((np.isclose(v_local, cfg.bus.voltage_min))
         | (np.isclose(v_local, cfg.bus.voltage_max))).sum()
    )
    report.check(
        "8. local bus voltage within limits", within,
        f"range [{v_local.min():.3f}, {v_local.max():.3f}] V, "
        f"{at_rail} row(s) clipped at a limit "
        f"({100 * at_rail / max(len(v_local), 1):.2f}%)",
    )
    v_comm = community["community_bus_voltage"]
    comm_within = bool(
        ((v_comm >= cfg.bus.community_voltage_min - 1e-9)
         & (v_comm <= cfg.bus.community_voltage_max + 1e-9)).all()
    )
    comm_at_rail = int(
        ((np.isclose(v_comm, cfg.bus.community_voltage_min))
         | (np.isclose(v_comm, cfg.bus.community_voltage_max))).sum()
    )
    report.check(
        "8b. community bus voltage within limits", comm_within,
        f"range [{v_comm.min():.3f}, {v_comm.max():.3f}] V, "
        f"{comm_at_rail} row(s) clipped",
    )

    # 9. P = V x I for every element that carries both.
    ohm_pairs = [
        ("pv", "pv_power", "pv_voltage", "pv_current", False),
        ("load", "load_power", "load_voltage", "load_current", False),
        ("battery", "battery_power", "battery_voltage", "battery_current", False),
        ("tie", "tie_power", "local_bus_voltage", "tie_current", True),
    ]
    for name, p_col, v_col, i_col, magnitude in ohm_pairs:
        expected = 1000.0 * dataset[p_col] / dataset[v_col]
        if magnitude:
            expected = expected.abs()
        error = _max_abs(expected - dataset[i_col])
        report.check(
            f"9. P = V x I ({name})", error <= tol.ohm_tolerance_a,
            f"max error {error:.2e} A",
        )

    # 10. Household energy balance.
    imported = dataset["tie_power"].clip(upper=0).abs()
    exported = dataset["tie_power"].clip(lower=0)
    supply = (
        dataset["pv_power"]
        + dataset["battery_discharge_power"]
        + imported
        + dataset["unserved_power"]
    )
    demand = (
        dataset["load_power"]
        + dataset["battery_charge_power"]
        + exported
        + dataset["curtailed_power"]
    )
    house_error = _max_abs(supply - demand)
    report.check(
        "10. household energy balance", house_error <= tol.energy_balance_tolerance_kw,
        f"max residual {house_error:.2e} kW",
    )

    # 10b. SoC transitions must match battery powers.
    soc_error = 0.0
    dt = cfg.source.interval_hours
    for _, group in dataset.groupby("house_id"):
        group = group.sort_values("timestamp")
        delta_kwh = group["battery_soc"].diff() * b.capacity_kwh
        expected = (
            group["battery_charge_power"] * b.charge_efficiency * dt
            - group["battery_discharge_power"] / b.discharge_efficiency * dt
        )
        soc_error = max(soc_error, _max_abs((delta_kwh - expected).dropna()))
    report.check(
        "10b. SoC transitions match battery power",
        soc_error <= tol.energy_balance_tolerance_kw,
        f"max residual {soc_error:.2e} kWh",
    )

    # 11. Community balance.
    per_step = dataset.groupby("timestamp")["tie_power"]
    exports = per_step.apply(lambda s: s.clip(lower=0).sum())
    imports = per_step.apply(lambda s: s.clip(upper=0).abs().sum())
    merged = community.set_index("timestamp")
    delivered = (
        exports.reindex(merged.index) - merged["network_loss_kw"]
    )
    comm_error = _max_abs(
        delivered + merged["grid_slack_kw"] - imports.reindex(merged.index)
    )
    report.check(
        "11. community energy balance",
        comm_error <= tol.energy_balance_tolerance_kw,
        f"max residual {comm_error:.2e} kW "
        f"(loss fraction {cfg.network.loss_fraction})",
    )
    export_error = _max_abs(exports.reindex(merged.index) - merged["total_export_kw"])
    report.check(
        "11b. community totals match per-house rows",
        export_error <= tol.energy_balance_tolerance_kw,
        f"max residual {export_error:.2e} kW",
    )

    # 12. Timestamps ordered, uniform spacing.
    ordered = bool(
        dataset.groupby("house_id")["timestamp"]
        .apply(lambda s: s.is_monotonic_increasing)
        .all()
    )
    report.check("12. timestamps ordered per house", ordered)
    steps = dataset[dataset["house_id"] == dataset["house_id"].iloc[0]]["timestamp"]
    gaps = pd.Series(steps).diff().dropna().unique()
    expected_gap = pd.Timedelta(hours=cfg.source.interval_hours)
    report.check(
        "12b. uniform half-hourly spacing",
        len(gaps) == 1 and gaps[0] == expected_gap,
        f"{len(gaps)} distinct interval(s)",
    )

    # 13. Profile fidelity (if source profiles are available).
    if profiles is not None:
        _check_profiles(dataset, profiles, cfg, report)

    # Informational: how often the converter rating actually bound.
    spill = int(
        ((dataset["curtailed_power"] > 0) | (dataset["unserved_power"] > 0)).sum()
    )
    report.check(
        "14. tie rating rarely binds (informational)", True,
        f"{spill} row(s) curtailed/unserved "
        f"({100 * spill / max(len(dataset), 1):.2f}%)",
    )

    return report


def _check_profiles(dataset, profiles, cfg, report):
    source = profiles.frame
    worst_pv, worst_load = 1.0, 1.0
    worst_scale = 0.0

    for customer, group in dataset.groupby("ausgrid_customer_id"):
        src = source[source["customer_id"] == customer].set_index("timestamp")
        gen = group.set_index("timestamp").reindex(src.index)

        for col, src_col, tracker in (
            ("pv_power", "ausgrid_pv_kw", "pv"),
            ("load_power", "ausgrid_load_kw", "load"),
        ):
            a = src[src_col].to_numpy(dtype=float)
            b_arr = gen[col].to_numpy(dtype=float)
            if a.std() == 0 or b_arr.std() == 0:
                corr = 1.0
            else:
                corr = float(np.corrcoef(a, b_arr)[0, 1])
            if tracker == "pv":
                worst_pv = min(worst_pv, corr)
            else:
                worst_load = min(worst_load, corr)

        capacity = profiles.capacity_kwp[customer]
        expected = cfg.pv.capacity_kw / capacity
        nonzero = src["ausgrid_pv_kw"].to_numpy(dtype=float) > 0
        if nonzero.any():
            ratio = (
                gen["pv_power"].to_numpy(dtype=float)[nonzero]
                / src["ausgrid_pv_kw"].to_numpy(dtype=float)[nonzero]
            )
            worst_scale = max(worst_scale, float(np.abs(ratio - expected).max()))

    minimum = cfg.validation.profile_correlation_min
    report.check(
        "13. PV profile matches real Ausgrid shape", worst_pv >= minimum,
        f"min correlation {worst_pv:.6f}",
    )
    report.check(
        "13b. load profile matches real Ausgrid shape", worst_load >= minimum,
        f"min correlation {worst_load:.6f}",
    )
    report.check(
        "13c. PV rescaling is exactly capacity_kw / generator_capacity",
        worst_scale <= 1e-9,
        f"max deviation {worst_scale:.2e}",
    )


# =====================================================================
# 3. Visualization
# =====================================================================

PROVISIONAL_NOTE = "provisional simulation - not hardware measurements"


def _pick_day(dataset):
    """A representative day: the one with the most PV generation."""
    days = dataset.assign(day=dataset["timestamp"].dt.date)
    return days.groupby("day")["pv_power"].sum().idxmax()


def _day_slice(frame, day):
    return frame[frame["timestamp"].dt.date == day].sort_values("timestamp")


def _finish(fig, ax_or_axes, path):
    axes = ax_or_axes if isinstance(ax_or_axes, (list, tuple)) else [ax_or_axes]
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def generate_plots(dataset, community, cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    houses = sorted(dataset["house_id"].unique())
    focus = houses[0]
    day = _pick_day(dataset[dataset["house_id"] == focus])

    house_day = _day_slice(dataset[dataset["house_id"] == focus], day)
    comm_day = _day_slice(community, day)
    written = []

    single_series = [
        ("01_pv_power", "pv_power", "PV power (kW)",
         f"PV power - {focus}, {day}", "tab:orange"),
        ("02_load_power", "load_power", "Load power (kW)",
         f"DC load power - {focus}, {day}", "tab:red"),
        ("03_battery_soc", "battery_soc", "State of charge (fraction)",
         f"Battery SoC - {focus}, {day}", "tab:green"),
        ("04_tie_power", "tie_power", "Tie power (kW)",
         f"Bus-tie power - {focus}, {day}\n(+ export to community, "
         "- import from community)", "tab:blue"),
        ("05_local_bus_voltage", "local_bus_voltage", "Local bus voltage (V)",
         f"Local DC bus voltage - {focus}, {day}", "tab:purple"),
    ]

    for name, column, ylabel, title, colour in single_series:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(house_day["timestamp"], house_day[column], color=colour, lw=1.6)
        ax.set_title(f"{title}\n[{PROVISIONAL_NOTE}]"
                     if column not in ("pv_power", "load_power")
                     else f"{title}\n[real Ausgrid temporal profile]")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("time")

        if column == "battery_soc":
            ax.axhline(cfg.battery.min_soc, ls="--", c="grey", lw=1,
                       label=f"min_soc {cfg.battery.min_soc}")
            ax.axhline(cfg.battery.max_soc, ls="--", c="grey", lw=1,
                       label=f"max_soc {cfg.battery.max_soc}")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc="best", fontsize=8)
        if column == "tie_power":
            ax.axhline(0, c="black", lw=0.8)
        if column == "local_bus_voltage":
            ax.axhline(cfg.bus.voltage_min, ls="--", c="grey", lw=1)
            ax.axhline(cfg.bus.voltage_max, ls="--", c="grey", lw=1)

        written.append(_finish(fig, ax, out_dir / f"{name}.png"))

    # 6. Community bus voltage.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(comm_day["timestamp"], comm_day["community_bus_voltage"],
            color="tab:brown", lw=1.6)
    ax.axhline(cfg.bus.community_voltage_min, ls="--", c="grey", lw=1)
    ax.axhline(cfg.bus.community_voltage_max, ls="--", c="grey", lw=1)
    ax.set_title(f"Community DC bus voltage (shared by all houses), {day}\n"
                 f"[{PROVISIONAL_NOTE}]")
    ax.set_ylabel("Community bus voltage (V)")
    ax.set_xlabel("time")
    written.append(_finish(fig, ax, out_dir / "06_community_bus_voltage.png"))

    # 7. PV against load.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(house_day["timestamp"], house_day["pv_power"],
            label="PV power", color="tab:orange", lw=1.6)
    ax.plot(house_day["timestamp"], house_day["load_power"],
            label="Load power", color="tab:red", lw=1.6)
    ax.fill_between(house_day["timestamp"], house_day["pv_power"],
                    house_day["load_power"],
                    where=house_day["pv_power"] >= house_day["load_power"],
                    color="tab:green", alpha=0.25, label="surplus")
    ax.fill_between(house_day["timestamp"], house_day["pv_power"],
                    house_day["load_power"],
                    where=house_day["pv_power"] < house_day["load_power"],
                    color="tab:red", alpha=0.15, label="deficit")
    ax.set_title(f"PV vs load - {focus}, {day}\n[real Ausgrid temporal profiles]")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("time")
    ax.legend(fontsize=8)
    written.append(_finish(fig, ax, out_dir / "07_pv_vs_load.png"))

    # 8. Energy flow.
    written.append(_energy_flow(house_day, focus, day, out_dir))

    # 9. Multiple houses together.
    written.append(_multi_house(dataset, day, houses, out_dir))

    return written


def _energy_flow(house_day, focus, day, out_dir):
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    t = house_day["timestamp"]
    ax.plot(t, house_day["pv_power"], label="PV (supply)",
            color="tab:orange", lw=1.6)
    ax.plot(t, -house_day["load_power"], label="Load (demand)",
            color="tab:red", lw=1.6)
    ax.plot(t, house_day["battery_power"], label="Battery (+ discharge / - charge)",
            color="tab:green", lw=1.4)
    ax.plot(t, house_day["tie_power"], label="Tie (+ export / - import)",
            color="tab:blue", lw=1.4)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Energy flow - {focus}, {day}\n"
                 "supply above zero, demand below; "
                 f"[{PROVISIONAL_NOTE}]")
    ax.legend(fontsize=8, ncol=2)

    imported = house_day["tie_power"].clip(upper=0).abs()
    exported = house_day["tie_power"].clip(lower=0)
    residual = (
        house_day["pv_power"] + house_day["battery_discharge_power"]
        + imported + house_day["unserved_power"]
        - house_day["load_power"] - house_day["battery_charge_power"]
        - exported - house_day["curtailed_power"]
    )
    ax2.plot(t, residual, color="black", lw=1.2)
    ax2.set_ylabel("Balance\nresidual (kW)")
    ax2.set_xlabel("time")
    ax2.set_title("Household energy balance residual (should be zero)",
                  fontsize=9)
    ax2.set_ylim(-1e-6, 1e-6)

    return _finish(fig, [ax, ax2], out_dir / "08_energy_flow.png")


def _multi_house(dataset, day, houses, out_dir):
    """Confirm the houses are backed by DIFFERENT real customers."""

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    panels = [
        ("pv_power", "PV power (kW)"),
        ("load_power", "Load power (kW)"),
        ("battery_soc", "Battery SoC"),
    ]

    for ax, (column, ylabel) in zip(axes, panels):
        for house in houses:
            day_frame = _day_slice(dataset[dataset["house_id"] == house], day)
            customer = day_frame["ausgrid_customer_id"].iloc[0]
            ax.plot(day_frame["timestamp"], day_frame[column], lw=1.4,
                    label=f"{house} (Ausgrid customer {customer})")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0].set_title(f"All houses, {day} - distinct real Ausgrid customers")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time")

    return _finish(fig, list(axes), out_dir / "09_all_houses.png")


def run_plots(config_path=DEFAULT_CONFIG_PATH, verbose=True):
    """Load the generated dataset and produce all plots."""
    cfg = load_config(config_path)

    dataset = pd.read_csv(
        cfg.resolve(cfg.output.dataset_path), parse_dates=["timestamp"]
    )
    community = pd.read_csv(
        cfg.resolve(cfg.output.community_path), parse_dates=["timestamp"]
    )

    out_dir = cfg.resolve(cfg.output.plots_dir)
    written = generate_plots(dataset, community, cfg, out_dir)

    if verbose:
        print(f"Wrote {len(written)} plot(s) to {out_dir}:")
        for p in written:
            print(f"  {p.name}")

    return written
