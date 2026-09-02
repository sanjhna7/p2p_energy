"""
Visual validation of the provisional dataset.

The numeric checks in validate.py prove the dataset is internally
consistent. These plots are for the other half of the question: does it
still LOOK like real household data, and do the houses actually differ
from one another (if they do not, there is nothing for a MARL agent to
trade).

Run:  python -m nanogrid.plots
"""

import argparse

import matplotlib
matplotlib.use("Agg")            # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_config

PROVISIONAL_NOTE = "provisional simulation - not hardware measurements"


def _pick_day(dataset):
    """A representative day: the one with the most PV generation, so the
    plots show the interesting case (surplus, charging, export) rather
    than an overcast day where nothing happens."""
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

    # 6. Community bus voltage - one shared series for the whole community.
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

    # 7. PV against load - the surplus/deficit an agent has to manage.
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

    # 8. Energy flow: every power in one stacked picture, so a coherent
    # flow (PV surplus -> battery -> export) is visible at a glance.
    written.append(_energy_flow(house_day, focus, day, out_dir))

    # 9. Several houses together - they must NOT look alike.
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

    # Residual of the household balance, which should sit on zero.
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


def run(config_path=DEFAULT_CONFIG_PATH, verbose=True):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
