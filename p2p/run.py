"""Run the baselines on the configured scenario and write the results.

    python -m p2p.run
    python -m p2p.run --config path/to/other.yaml --out results/other

Writes to --out (default results/<n>houses_<start>_<days>d/):
  summary.csv        one row per policy, the slide-27 measures
  provenance.csv     every config value with its source and citation
  <policy>_steps.csv community totals per interval
  plots/             visual checks
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p2p import baselines, metrics
from p2p.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from p2p.scenario import load_scenario

COMMUNITY_SERIES = ["load_kw", "pv_kw", "grid_import_kw", "grid_export_kw", "p2p_sold_kw", "battery_kw", "cost_aud"]


def default_out(cfg) -> Path:
    s = cfg.scenario
    return PROJECT_ROOT / "results" / f"{s.n_houses.value}houses_{s.start.value}_{s.days.value}d"


def steps_frame(log, sc) -> pd.DataFrame:
    frame = pd.DataFrame({k: log[k].sum(axis=1) for k in COMMUNITY_SERIES}, index=sc.timestamps)
    frame["mean_soc"] = log["soc"].mean(axis=1)
    for k in ("v_min", "v_max", "i_max_abs", "loss_kw", "grid_kw", "blocked", "inherent_violation", "converged"):
        frame[k] = log[k]
    frame.index.name = "timestamp"
    return frame


def plot(logs, sc, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t = sc.timestamps

    fig, ax = plt.subplots(figsize=(11, 4))
    log = logs["rule_p2p"]
    ax.plot(t, log["load_kw"].sum(1), label="load")
    ax.plot(t, log["pv_kw"].sum(1), label="PV")
    ax.plot(t, log["p2p_sold_kw"].sum(1), label="P2P traded")
    ax.plot(t, log["grid_import_kw"].sum(1), label="grid import")
    ax.set_ylabel("kW (community total)")
    ax.set_title(f"rule_p2p, {sc.load_kw.shape[1]} houses")
    ax.legend(ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "energy_flows.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, log in logs.items():
        ax.plot(t, np.cumsum(log["cost_aud"].sum(1)), label=name)
    ax.set_ylabel("cumulative community cost (AUD)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_cost.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, log in logs.items():
        ax.fill_between(t, log["v_min"], log["v_max"], alpha=0.3, label=name)
    ax.set_ylabel("node voltage range (V)")
    ax.set_title("DC network voltages (bus voltage and limits PENDING EEE spec)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "voltages.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sc = load_scenario(cfg)
    out = args.out or default_out(cfg)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{sc.load_kw.shape[1]} houses ({', '.join(sc.houses['house_id'])}), "
          f"{len(sc.timestamps)} steps from {sc.timestamps[0]}")

    logs = {name: baselines.run(name, sc, cfg) for name in baselines.POLICIES}
    rows = {name: metrics.summarise(log, sc.dt_hours, baseline=logs["grid_only"]) for name, log in logs.items()}
    summary = pd.DataFrame(rows).T
    summary.index.name = "policy"
    summary.to_csv(out / "summary.csv", float_format="%.4f")
    pd.DataFrame(cfg.provenance(), columns=["name", "value", "source", "citation"]).to_csv(
        out / "provenance.csv", index=False)
    for name, log in logs.items():
        steps_frame(log, sc).to_csv(out / f"{name}_steps.csv", float_format="%.5f")
    if not args.no_plots:
        plot(logs, sc, out / "plots")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(summary.T)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
