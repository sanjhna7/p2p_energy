"""Build the MVP dataset: real Ausgrid household load/PV mapped onto the
55 residential connection points of the IEEE European LV Test Feeder.

    python -m data_pipeline.build
    python -m data_pipeline.build --ausgrid-csv "data/raw/ausgrid_solar_home/Solar home 2011-2012.csv"

Outputs (in --out):
    houses_timeseries.csv.gz   per house, per 30-min interval
    house_mapping.csv          house_id <-> feeder load <-> Ausgrid customer
    nodes.csv / edges.csv      reduced feeder graph used by the GAT
    edges_full.csv             all 905 original line segments
    ausgrid_quality_report.csv every customer, checks and exclusion reasons
    manifest.json              sources, file hashes, parameters, counts

The pairing of a Sydney household with a European benchmark feeder position is OUR
adaptation. It is not in either source and must be described as such.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import ausgrid, feeder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUSGRID = ROOT / "data/raw/ausgrid_solar_home/Solar home 2010-2011.csv"
DEFAULT_FEEDER = ROOT / "data/raw/ieee_european_lv/European_LV_CSV"
DEFAULT_OUT = ROOT / "data/processed/mvp_ausgrid_eulv"

UNAVAILABLE_COLUMNS = {
    "battery_power_kw": "Not in Ausgrid Solar Home; to be produced by the battery model in the environment.",
    "battery_soc": "Not in Ausgrid Solar Home; to be produced by the battery model in the environment.",
    "grid_power_kw": "Not measurable from gross-metered data once a battery/P2P exchange exists; "
                     "computed by the environment from the energy balance.",
}


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ausgrid-csv", type=Path, default=DEFAULT_AUSGRID)
    parser.add_argument("--feeder-dir", type=Path, default=DEFAULT_FEEDER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling customers onto feeder loads")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Reading feeder: {args.feeder_dir}")
    lines, loads, coords, _, root_bus = feeder.read_feeder(args.feeder_dir)
    active_lines = feeder.prune_dead_ends(lines, loads, root_bus)
    nodes, edges = feeder.reduce_feeder(active_lines, loads, coords, root_bus)
    houses = nodes[nodes["node_type"] == "house"].copy()
    houses["load_number"] = houses["load_name"].str.replace("LOAD", "").astype(int)
    houses = houses.sort_values("load_number")
    print(f"  {len(lines)} segments ({len(lines) - len(active_lines)} dead-end stubs pruned) -> "
          f"{len(nodes)} nodes / {len(edges)} edges, {len(houses)} houses")

    print(f"Reading Ausgrid: {args.ausgrid_csv}")
    year = ausgrid.read_wide(args.ausgrid_csv)
    rules = ausgrid.CleaningRules()
    report = ausgrid.quality_report(year, rules)
    eligible = report.index[report["included"]].to_numpy()
    print(f"  {len(report)} customers, {len(eligible)} pass the cleaning rules")
    if len(eligible) < len(houses):
        raise SystemExit(f"Only {len(eligible)} clean customers for {len(houses)} feeder loads")

    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(eligible, size=len(houses), replace=False))

    mapping = pd.DataFrame(
        {
            "house_id": [f"H{n:02d}" for n in houses["load_number"]],
            "node_id": houses["node_id"].to_numpy(),
            "feeder_load_name": houses["load_name"].to_numpy(),
            "feeder_source_bus": houses["source_bus"].to_numpy(),
            "feeder_phase": houses["load_phase"].to_numpy(),
            "ausgrid_customer_id": chosen.astype(int),
        }
    )
    mapping = mapping.merge(
        report[["pv_capacity_kwp", "postcode", "has_controlled_load", "annual_consumption_kwh",
                "annual_generation_kwh"]],
        left_on="ausgrid_customer_id", right_index=True, how="left",
    )

    series = ausgrid.to_long(year, customers=chosen)
    series = series.merge(mapping[["ausgrid_customer_id", "house_id", "node_id"]], on="ausgrid_customer_id")
    series["net_load_kw_derived"] = series["load_power_kw"] - series["pv_power_kw"]
    for col in UNAVAILABLE_COLUMNS:
        series[col] = np.nan
    series = series[
        ["timestamp", "interval_end", "house_id", "node_id", "ausgrid_customer_id",
         "load_power_kw", "load_gc_kw", "load_cl_kw", "pv_power_kw", "net_load_kw_derived",
         "battery_power_kw", "battery_soc", "grid_power_kw"]
    ].sort_values(["timestamp", "house_id"])

    expected_rows = len(chosen) * year.wide["date"].nunique() * 48
    if len(series) != expected_rows or series[["load_power_kw", "pv_power_kw"]].isna().any().any():
        raise SystemExit(f"Time series incomplete: {len(series)} rows, expected {expected_rows}")

    series.to_csv(args.out / "houses_timeseries.csv.gz", index=False, float_format="%.4f")
    mapping.to_csv(args.out / "house_mapping.csv", index=False)
    nodes.to_csv(args.out / "nodes.csv", index=False)
    edges.to_csv(args.out / "edges.csv", index=False)
    feeder.full_edges(lines).to_csv(args.out / "edges_full.csv", index=False)
    report.to_csv(args.out / "ausgrid_quality_report.csv")

    feeder_files = ["Lines.csv", "LineCodes.csv", "Loads.csv", "Buscoords.csv", "Transformer.csv", "Source.csv"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "ausgrid_solar_home": {
                "type": "REAL measured (billing meters, AC side, gross-metered PV)",
                "file": rel(args.ausgrid_csv),
                "sha256": sha256(args.ausgrid_csv),
                "licence": "CC BY 3.0 AU (data.gov.au record)",
                "citation": "E. L. Ratnam, S. R. Weller, C. M. Kellett, A. T. Murray, 'Residential load and rooftop PV "
                            "generation: an Australian distribution network dataset', Int. J. Sustainable Energy "
                            "36(8):787-806, 2017, doi:10.1080/14786451.2015.1100196",
            },
            "ieee_european_lv_test_feeder": {
                "type": "BENCHMARK network model (AC, 416 V, three-phase)",
                "dir": rel(args.feeder_dir),
                "sha256": {f: sha256(args.feeder_dir / f) for f in feeder_files},
                "citation": "IEEE PES Distribution Test Feeders WG, European LV Test Feeder (2015); K. P. Schneider "
                            "et al., 'Analytic Considerations and Design Basis for the IEEE Distribution Test "
                            "Feeders', IEEE Trans. Power Syst. 33(3), 2018, doi:10.1109/TPWRS.2017.2760011",
            },
        },
        "our_adaptations": [
            "kWh per 30 min converted to average kW (x2)",
            "timestamp = interval start, local wall-clock (EST/EDT) as recorded; DST not re-aligned",
            "load_power_kw = GC + CL; missing CL rows treated as zero controlled load",
            "customers excluded by the cleaning rules below (ours, not Ratnam et al.'s)",
            f"{len(chosen)} clean customers sampled with seed {args.seed} and assigned in ascending ID order "
            "to feeder loads LOAD1..LOAD55",
            f"feeder: {len(lines) - len(active_lines)} dead-end segments without a load pruned "
            f"({lines['length_m'].sum() - active_lines['length_m'].sum():.1f} m); edges_full.csv keeps all 905",
            "feeder reduced by merging series segments between transformer, house and junction buses; "
            "R/X summed as published (AC positive/zero sequence); no DC conversion applied here",
        ],
        "cleaning_rules": asdict(rules),
        "unavailable_columns": UNAVAILABLE_COLUMNS,
        "counts": {
            "ausgrid_customers": int(len(report)),
            "ausgrid_customers_clean": int(len(eligible)),
            "houses_mapped": int(len(chosen)),
            "intervals_per_house": int(len(series) // len(chosen)),
            "time_start": str(series["timestamp"].min()),
            "time_end": str(series["interval_end"].max()),
            "feeder_segments": int(len(lines)),
            "feeder_segments_after_pruning": int(len(active_lines)),
            "graph_nodes": int(len(nodes)),
            "graph_edges": int(len(edges)),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {rel(args.out)}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
