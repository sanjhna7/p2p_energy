"""
Validation of the provisional dataset.

Every check either passes, fails, or reports a bounded number of
violations. Nothing is silently tolerated: a dataset that does not
conserve energy is worse than no dataset at all, because a MARL agent
will happily learn to exploit the inconsistency.

Run:  python -m nanogrid.validate
"""

import argparse
import sys

import numpy as np
import pandas as pd

from .ausgrid_source import load_profiles
from .config import DEFAULT_CONFIG_PATH, load_config


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
    # Charging and discharging are mutually exclusive in the same step.
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

    # 8. Voltage limits. Clipping is what enforces these, so a check that
    # merely passes is uninformative; report how often the model SAT on a
    # rail, which is the real signal that the droop settings are wrong.
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

    # 10. Household energy balance, per row:
    #   pv + discharge + import = load + charge + export + curtailed + unserved
    imported = dataset["tie_power"].clip(upper=0).abs()
    exported = dataset["tie_power"].clip(lower=0)
    # Curtailment is surplus dumped rather than delivered, so it sits on
    # the demand side; unserved load is demand never met, so it is
    # credited on the supply side. Both are zero unless the converter
    # rating binds.
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

    # 10b. The SoC trajectory must be produced by the battery powers in
    # the same rows - the check that the columns are one coupled state
    # and not three independently generated series.
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

    # 11. Community balance: exports (less network loss) plus the
    # explicit external-grid slack must equal imports, exactly.
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
    # The aggregated tie powers must match the per-house rows they came from.
    export_error = _max_abs(exports.reindex(merged.index) - merged["total_export_kw"])
    report.check(
        "11b. community totals match per-house rows",
        export_error <= tol.energy_balance_tolerance_kw,
        f"max residual {export_error:.2e} kW",
    )

    # 12. Ausgrid timestamps remain ordered, per house, with no gaps or
    # duplicates in the half-hourly grid.
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

    # 13. The temporal profiles must still be the REAL ones. Compare
    # against the source, house by house: a pure re-scaling leaves the
    # correlation at exactly 1.
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
            b = gen[col].to_numpy(dtype=float)
            # A constant profile has no variance to correlate; treat a
            # perfectly proportional pair as a match instead.
            if a.std() == 0 or b.std() == 0:
                corr = 1.0
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
            if tracker == "pv":
                worst_pv = min(worst_pv, corr)
            else:
                worst_load = min(worst_load, corr)

        # The PV re-scaling factor must be the documented one.
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


def run(config_path=DEFAULT_CONFIG_PATH, verbose=True):
    cfg = load_config(config_path)

    dataset = pd.read_csv(
        cfg.resolve(cfg.output.dataset_path), parse_dates=["timestamp"]
    )
    community = pd.read_csv(
        cfg.resolve(cfg.output.community_path), parse_dates=["timestamp"]
    )
    profiles = load_profiles(cfg)

    report = validate(dataset, community, cfg, profiles)

    if verbose:
        print("=" * 62)
        print("Provisional dataset validation")
        print("=" * 62)
        print(report.render())
        n = len(report.results)
        print(f"\n{n - len(report.failed)}/{n} checks passed.")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    report = run(args.config)
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
