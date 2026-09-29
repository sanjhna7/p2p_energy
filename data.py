"""
Data loading, provenance tracking, and synthetic data generation.

This module consolidates three responsibilities:

1. **Real Ausgrid data loading** — loads real PV generation and household
   consumption profiles from either the project SQLite database or the
   raw CSV, with unit conversion (kWh → kW).

2. **Provenance tracking** — labels every column in the dataset as either
   "ausgrid_derived" (real) or "provisional_simulation" (modelled), so
   the distinction survives into any paper written on the data.

3. **Synthetic DC microgrid data generation** — produces a self-contained
   synthetic dataset with consistent physics (Ohm's law, DC power,
   battery SoC) for testing and development.
"""

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from config import NanogridConfig

# =====================================================================
# 1. Real Ausgrid data loading
# =====================================================================

# Columns of the raw CSV that are not half-hourly readings.
_CSV_ID_COLS = ["Customer", "Consumption Category", "date"]
_CSV_META_COLS = ["Generator Capacity", "Postcode"]


@dataclass
class AusgridProfiles:
    """Real Ausgrid profiles for the selected customers.

    frame: long format, one row per (customer_id, timestamp), columns
           `customer_id`, `timestamp`, `ausgrid_pv_kw`, `ausgrid_load_kw`.
    capacity_kwp: installed PV capacity per customer, straight from the
           dataset's "Generator Capacity" field.
    """

    frame: pd.DataFrame
    capacity_kwp: dict
    origin: str          # "sqlite" or "csv" - recorded in the metadata
    # Selection order, so house_0 is the first customer chosen (or the
    # first one listed in config) rather than the lexicographically
    # smallest id that happens to appear first in the frame.
    selected: List[str]

    @property
    def customer_ids(self) -> List[str]:
        return list(self.selected)

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.frame["timestamp"].unique()))


def load_profiles(cfg: NanogridConfig) -> AusgridProfiles:
    """Load real profiles for `n_houses` distinct Ausgrid customers.

    Prefers the project SQLite database (already ingested, already in
    kW, and far cheaper than re-melting the 63 MB CSV); falls back to
    parsing the CSV when the DB is absent or empty.
    """

    db_path = cfg.resolve(cfg.source.db_path)

    if db_path.exists():
        try:
            return _load_from_sqlite(cfg, db_path)
        except (_NoData, Exception):
            # _NoData: DB exists but has no usable readings.
            # Other exceptions: DB exists but schema is missing/incompatible
            # (e.g. no 'houses' table if ingest was never run).
            # Either way, fall through to CSV.
            pass

    return _load_from_csv(cfg)


class _NoData(Exception):
    """The database exists but holds no usable Ausgrid readings."""


def _select_customers(available: Sequence[str], cfg: NanogridConfig) -> List[str]:
    """Pick distinct real customers, in a stable order.

    Distinct on purpose: duplicating one profile across every house
    would make the houses' surplus/deficit perfectly correlated and
    leave nothing for a MARL agent to trade.
    """

    requested = cfg.source.customer_ids

    if requested:
        wanted = [str(c) for c in requested]
        missing = [c for c in wanted if c not in set(available)]
        if missing:
            raise ValueError(f"customer_ids not present in the dataset: {missing}")
        return wanted

    ordered = sorted(available, key=lambda c: (len(c), c))
    n = cfg.source.n_houses

    if len(ordered) < n:
        raise ValueError(
            f"config asks for {n} houses but only {len(ordered)} customers "
            "with PV capacity and a complete time series are available"
        )

    return ordered[:n]


def _load_from_sqlite(cfg: NanogridConfig, db_path) -> AusgridProfiles:
    conn = sqlite3.connect(db_path)
    try:
        # Only customers with a real PV rating and the full time series
        # are eligible; a partial profile would put gaps in the state.
        eligible = pd.read_sql(
            """
            SELECT h.house_id, h.panel_capacity_kw, COUNT(m.id) AS n_readings
            FROM houses h
            JOIN meter_readings m ON m.house_id = h.house_id
            WHERE h.panel_capacity_kw > 0
            GROUP BY h.house_id, h.panel_capacity_kw
            """,
            conn,
        )

        if eligible.empty:
            raise _NoData

        full = eligible["n_readings"].max()
        eligible = eligible[eligible["n_readings"] == full]

        eligible["house_id"] = eligible["house_id"].astype(str)
        chosen = _select_customers(eligible["house_id"].tolist(), cfg)
        capacity = dict(
            zip(eligible["house_id"], eligible["panel_capacity_kw"].astype(float))
        )

        placeholders = ",".join("?" * len(chosen))
        where, params = _timestamp_filter(cfg)

        frame = pd.read_sql(
            f"""
            SELECT timestamp,
                   house_id  AS customer_id,
                   solar_kw  AS ausgrid_pv_kw,
                   load_kw   AS ausgrid_load_kw
            FROM meter_readings
            WHERE house_id IN ({placeholders}) {where}
            ORDER BY timestamp, house_id
            """,
            conn,
            params=chosen + params,
        )
    finally:
        conn.close()

    if frame.empty:
        raise _NoData

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["customer_id"] = frame["customer_id"].astype(str)

    # meter_readings is already average kW (ingest.py divided by the
    # interval on the way in). No second conversion here - that would
    # double-count.
    return _finalise(
        frame,
        {c: capacity[c] for c in chosen},
        origin="sqlite",
        cfg=cfg,
        chosen=chosen,
    )


def _timestamp_filter(cfg: NanogridConfig):
    clauses, params = [], []
    if cfg.source.start_timestamp:
        clauses.append("AND timestamp >= ?")
        params.append(cfg.source.start_timestamp)
    if cfg.source.end_timestamp:
        clauses.append("AND timestamp <= ?")
        params.append(cfg.source.end_timestamp)
    return " ".join(clauses), params


def _load_from_csv(cfg: NanogridConfig) -> AusgridProfiles:
    """Parse the raw Ausgrid CSV.

    Mirrors the original ingest.py's reshaping (wide half-hourly
    columns -> long -> pivot on Consumption Category), then applies
    the kWh -> kW conversion explicitly.
    """

    csv_path = cfg.resolve(cfg.source.csv_path)
    # Row 0 is the dataset's disclaimer banner, not a header.
    df = pd.read_csv(csv_path, skiprows=1)
    df["Customer"] = df["Customer"].astype(str)

    capacity_all = df.groupby("Customer")["Generator Capacity"].max()
    capacity_all = capacity_all[capacity_all > 0]

    chosen = _select_customers(capacity_all.index.tolist(), cfg)
    df = df[df["Customer"].isin(chosen)]

    time_cols = [c for c in df.columns if c not in _CSV_ID_COLS + _CSV_META_COLS]

    long_df = df.melt(
        id_vars=_CSV_ID_COLS,
        value_vars=time_cols,
        var_name="time_of_day",
        value_name="value",
    )

    # "0:00" is midnight at the END of the day, i.e. hour 24.
    offsets = {}
    for t in long_df["time_of_day"].unique():
        h, m = map(int, t.split(":"))
        if h == 0 and m == 0:
            h = 24
        offsets[t] = pd.Timedelta(hours=h, minutes=m)

    long_df["timestamp"] = pd.to_datetime(
        long_df["date"], format="%d-%b-%y"
    ) + long_df["time_of_day"].map(offsets)

    pivot = long_df.pivot_table(
        index=["Customer", "timestamp"],
        columns="Consumption Category",
        values="value",
        aggfunc="first",
    ).reset_index()

    for category in ("GG", "GC", "CL"):
        if category not in pivot:
            pivot[category] = 0.0

    interval = cfg.source.interval_hours
    frame = pd.DataFrame(
        {
            "timestamp": pivot["timestamp"],
            "customer_id": pivot["Customer"].astype(str),
            # E(kWh per interval) -> average P(kW).
            "ausgrid_pv_kw": pivot["GG"].fillna(0) / interval,
            "ausgrid_load_kw": (
                pivot["GC"].fillna(0) + pivot["CL"].fillna(0)
            ) / interval,
        }
    )

    if cfg.source.start_timestamp:
        frame = frame[frame["timestamp"] >= pd.Timestamp(cfg.source.start_timestamp)]
    if cfg.source.end_timestamp:
        frame = frame[frame["timestamp"] <= pd.Timestamp(cfg.source.end_timestamp)]

    return _finalise(
        frame,
        {c: float(capacity_all[c]) for c in chosen},
        origin="csv",
        cfg=cfg,
        chosen=chosen,
    )


def _finalise(frame, capacity_kwp, origin, cfg, chosen) -> AusgridProfiles:
    """Trim to the configured window and keep only timestamps that every
    selected customer shares, so each step has a complete community."""

    frame = frame.sort_values(["timestamp", "customer_id"]).reset_index(drop=True)

    n_customers = frame["customer_id"].nunique()
    counts = frame.groupby("timestamp")["customer_id"].nunique()
    complete = counts[counts == n_customers].index
    frame = frame[frame["timestamp"].isin(complete)]

    if cfg.source.max_days:
        cutoff = frame["timestamp"].min() + pd.Timedelta(days=cfg.source.max_days)
        frame = frame[frame["timestamp"] < cutoff]

    if frame.empty:
        raise ValueError(
            "no Ausgrid readings left after applying the configured window"
        )

    # The real temporal order is the whole point of using this dataset -
    # assert it rather than trusting the sort.
    per_customer = frame.groupby("customer_id")["timestamp"]
    if not per_customer.apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("Ausgrid timestamps are not ordered per customer")

    return AusgridProfiles(
        frame=frame.reset_index(drop=True),
        capacity_kwp=capacity_kwp,
        origin=origin,
        selected=list(chosen),
    )


# =====================================================================
# 2. Provenance tracking
# =====================================================================

AUSGRID = "ausgrid_derived"
PROVISIONAL = "provisional_simulation"

RESEARCH_STATEMENT = (
    "The preliminary MARL dataset uses real temporal household PV "
    "generation and consumption profiles derived from the Ausgrid Solar "
    "Home Electricity Dataset. Electrical variables unavailable in the "
    "source dataset, including battery state-of-charge, DC-bus voltage, "
    "converter/tie power and current, are generated using a simplified "
    "physically consistent nanogrid model for algorithm-development "
    "purposes. These provisional variables will subsequently be replaced "
    "by measurements from the completed Simulink/hardware model."
)

PROVENANCE = {
    "timestamp": (AUSGRID, "Original Ausgrid half-hourly interval timestamp."),
    "house_id": (PROVISIONAL, "Index of the house/agent in this simulation."),
    "ausgrid_customer_id": (
        AUSGRID,
        "Real Ausgrid customer whose profile backs this house.",
    ),

    "pv_power": (
        AUSGRID,
        "Ausgrid GG (gross generation), kWh/interval -> kW, normalised by "
        "the customer's Generator Capacity and re-scaled to pv.capacity_kw.",
    ),
    "pv_voltage": (PROVISIONAL, "Assumed nominal PV operating voltage."),
    "pv_current": (PROVISIONAL, "pv_power / pv_voltage."),

    "load_power": (
        AUSGRID,
        "Ausgrid GC + CL, kWh/interval -> kW, scaled by load.dc_load_fraction.",
    ),
    "load_voltage": (PROVISIONAL, "Assumed nominal DC load voltage."),
    "load_current": (PROVISIONAL, "load_power / load_voltage."),

    "battery_power": (
        PROVISIONAL,
        "Greedy self-consumption battery. > 0 discharging, < 0 charging.",
    ),
    "battery_soc": (PROVISIONAL, "Modelled state of charge, fraction of capacity."),
    "battery_voltage": (PROVISIONAL, "Assumed nominal battery terminal voltage."),
    "battery_current": (PROVISIONAL, "battery_power / battery_voltage."),
    "battery_charge_power": (PROVISIONAL, "Charging magnitude, kW."),
    "battery_discharge_power": (PROVISIONAL, "Discharging magnitude, kW."),

    "local_bus_voltage": (
        PROVISIONAL,
        "Droop approximation: V_nominal - virtual_droop x imported power.",
    ),
    "local_bus_current": (
        PROVISIONAL,
        "Total power injected into the local DC bus / local_bus_voltage.",
    ),

    "community_bus_voltage": (
        PROVISIONAL,
        "Shared community bus droop on the aggregate community imbalance. "
        "Identical for every house at a given timestep.",
    ),

    "tie_power": (
        PROVISIONAL,
        "Bidirectional bus-tie power. > 0 house -> community, "
        "< 0 community -> house. Limited by tie.max_power_kw.",
    ),
    "tie_current": (
        PROVISIONAL,
        "|tie_power| / local_bus_voltage. Magnitude; direction is in tie_power.",
    ),

    "net_power": (AUSGRID, "pv_power - load_power; both Ausgrid-derived."),
    "surplus_power": (AUSGRID, "max(net_power, 0)."),
    "deficit_power": (AUSGRID, "max(-net_power, 0)."),
    "energy_exchange": (PROVISIONAL, "tie_power x interval hours, in kWh."),

    "curtailed_power": (
        PROVISIONAL,
        "Surplus above the bus-tie rating; recorded so the household "
        "energy balance closes exactly.",
    ),
    "unserved_power": (
        PROVISIONAL,
        "Deficit above the bus-tie rating; recorded for the same reason.",
    ),
}


def provenance_for(columns):
    """Provenance entries for the columns actually written, in order."""
    return {
        c: {"provenance": PROVENANCE[c][0], "description": PROVENANCE[c][1]}
        for c in columns
        if c in PROVENANCE
    }


def provenance_summary(columns):
    """Count of columns per provenance label."""
    counts = {AUSGRID: 0, PROVISIONAL: 0}
    for c in columns:
        if c in PROVENANCE:
            counts[PROVENANCE[c][0]] += 1
    return counts


# =====================================================================
# 3. Synthetic DC microgrid data generation
# =====================================================================

# ---------------------------------------------------------------- CONFIG

RANDOM_SEED = 42

NUM_HOUSES = 20                      # houses on the DC microgrid
INTERVAL_MINUTES = 30                # sampling interval (matches Ausgrid's 30-min cadence)
START_DATE = "2024-01-01 00:00:00"
TARGET_ROWS = 10_000                 # NUM_TIMESTEPS is derived to hit this exactly
NUM_TIMESTEPS = TARGET_ROWS // NUM_HOUSES

DC_NOMINAL_VOLTAGE = 48.0            # V - common low-voltage DC microgrid bus standard
LINE_RESISTANCE_OHM = (0.03, 0.08)   # per-house feeder resistance range (Ohm), randomised per house
GRID_RESISTANCE_OHM = 0.015          # resistance from the shared bus to the upstream grid connection

LOAD_BASE_KW = (0.15, 0.6)           # per-house baseline load range
LOAD_PEAK_MULTIPLIER = 2.2           # morning/evening peak multiplier over baseline
LOAD_NOISE_STD_KW = 0.05

BATTERY_CAPACITY_KWH = (4.0, 12.0)
BATTERY_MAX_POWER_KW = (1.0, 3.0)
BATTERY_INITIAL_SOC_PCT = (20.0, 80.0)
BATTERY_EFFICIENCY = 0.95

SOLAR_CAPACITY_KW = (0.5, 4.0)       # installed capacity for houses that have panels
SOLAR_PENETRATION = 0.6              # fraction of houses with solar installed
SOLAR_NOISE_STD = 0.08               # fractional cloud-cover noise

SYNTHETIC_OUTPUT_FILE = "synthetic_dc_microgrid_data.csv"

# ------------------------------------------------------------------------

_rng = np.random.default_rng(RANDOM_SEED)


def _diurnal_solar_fraction(hour: float) -> float:
    """Bell-shaped generation fraction, 0 outside 6:00-18:00, peaking at noon."""
    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, np.sin(np.pi * (hour - 6) / 12)) ** 1.4


def _diurnal_load_fraction(hour: float) -> float:
    """Two peaks (morning ~8:00, evening ~19:00), lower overnight baseline."""
    morning = np.exp(-((hour - 8) ** 2) / (2 * 2.0 ** 2))
    evening = np.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2))
    base = 0.35
    return base + 0.65 * max(morning, evening)


def _build_house_params(num_houses: int) -> pd.DataFrame:
    has_solar = _rng.random(num_houses) < SOLAR_PENETRATION
    solar_capacity = np.where(
        has_solar,
        _rng.uniform(*SOLAR_CAPACITY_KW, num_houses),
        0.0,
    )
    return pd.DataFrame({
        "house_id": [f"H{100 + i}" for i in range(num_houses)],
        "load_base_kw": _rng.uniform(*LOAD_BASE_KW, num_houses),
        "solar_capacity_kw": solar_capacity,
        "battery_capacity_kwh": _rng.uniform(*BATTERY_CAPACITY_KWH, num_houses),
        "battery_max_power_kw": _rng.uniform(*BATTERY_MAX_POWER_KW, num_houses),
        "line_resistance_ohm": _rng.uniform(*LINE_RESISTANCE_OHM, num_houses),
        "soc_pct": _rng.uniform(*BATTERY_INITIAL_SOC_PCT, num_houses),
    })


def generate_synthetic() -> pd.DataFrame:
    """Generate a self-contained synthetic DC microgrid dataset.

    Produces a DataFrame with columns: timestamp, house_id,
    household_load_kw, solar_kw, battery_power_kw, battery_soc_pct,
    line_current_a, line_voltage_v, grid_current_a, grid_voltage_v.

    Physics used (all DC):
        - Ohm's law:        V = V_nominal + I * R
        - DC power law:     P (kW) = V (V) * I (A) / 1000
        - Battery SoC:      integrated from charge/discharge power
    """
    houses = _build_house_params(NUM_HOUSES)
    timestamps = pd.date_range(START_DATE, periods=NUM_TIMESTEPS, freq=f"{INTERVAL_MINUTES}min")
    dt_hours = INTERVAL_MINUTES / 60.0

    all_rows = []

    for ts in timestamps:
        hour = ts.hour + ts.minute / 60.0
        solar_frac = _diurnal_solar_fraction(hour)
        load_frac = _diurnal_load_fraction(hour)

        step_currents = []
        step_rows = []

        for idx, h in houses.iterrows():
            # --- Household load (kW) ---
            load_kw = max(
                0.0,
                h.load_base_kw * load_frac * LOAD_PEAK_MULTIPLIER
                + _rng.normal(0, LOAD_NOISE_STD_KW),
            )

            # --- Local solar generation (kW), 0 if no panels installed ---
            if h.solar_capacity_kw > 0:
                solar_kw = max(
                    0.0,
                    h.solar_capacity_kw * solar_frac * (1 + _rng.normal(0, SOLAR_NOISE_STD)),
                )
            else:
                solar_kw = 0.0

            net_kw = solar_kw - load_kw   # +ve = surplus, -ve = deficit

            # --- Battery: greedy self-consumption policy ---
            soc_kwh = h.soc_pct / 100.0 * h.battery_capacity_kwh

            if net_kw >= 0:
                charge_kw = min(net_kw, h.battery_max_power_kw)
                headroom_kwh = h.battery_capacity_kwh - soc_kwh
                max_chargeable_kw = headroom_kwh / dt_hours / BATTERY_EFFICIENCY
                charge_kw = max(0.0, min(charge_kw, max_chargeable_kw))
                soc_kwh += charge_kw * BATTERY_EFFICIENCY * dt_hours
                battery_power_kw = charge_kw          # +ve = charging
                residual_kw = net_kw - charge_kw
            else:
                deficit_kw = -net_kw
                discharge_kw = min(deficit_kw, h.battery_max_power_kw)
                max_dischargeable_kw = (soc_kwh * BATTERY_EFFICIENCY) / dt_hours
                discharge_kw = max(0.0, min(discharge_kw, max_dischargeable_kw))
                soc_kwh -= (discharge_kw / BATTERY_EFFICIENCY) * dt_hours
                battery_power_kw = -discharge_kw      # -ve = discharging
                residual_kw = net_kw + discharge_kw

            soc_pct = 100.0 * soc_kwh / h.battery_capacity_kwh
            houses.at[idx, "soc_pct"] = soc_pct   # carry forward to next timestep

            # --- DC electrical quantities ---
            line_current_a = (residual_kw * 1000.0) / DC_NOMINAL_VOLTAGE
            line_voltage_v = DC_NOMINAL_VOLTAGE + line_current_a * h.line_resistance_ohm

            step_currents.append(line_current_a)
            step_rows.append({
                "timestamp": ts,
                "house_id": h.house_id,
                "household_load_kw": round(load_kw, 4),
                "solar_kw": round(solar_kw, 4),
                "battery_power_kw": round(battery_power_kw, 4),
                "battery_soc_pct": round(soc_pct, 2),
                "line_current_a": round(line_current_a, 3),
                "line_voltage_v": round(line_voltage_v, 3),
            })

        # --- Grid / PCC voltage: shared bus, driven by the SUM of all houses' currents ---
        total_current_a = sum(step_currents)
        grid_voltage_v = DC_NOMINAL_VOLTAGE + total_current_a * GRID_RESISTANCE_OHM

        for r in step_rows:
            r["grid_current_a"] = round(total_current_a, 3)
            r["grid_voltage_v"] = round(grid_voltage_v, 3)

        all_rows.extend(step_rows)

    return pd.DataFrame(all_rows)


def run_synthetic():
    """Generate and save the synthetic dataset, printing a summary."""
    df = generate_synthetic()
    df.to_csv(SYNTHETIC_OUTPUT_FILE, index=False)

    print(f"Generated {len(df)} rows across {NUM_HOUSES} houses "
          f"and {NUM_TIMESTEPS} timesteps ({INTERVAL_MINUTES}-min interval).")
    print(f"Saved to {SYNTHETIC_OUTPUT_FILE}")
    print()
    print(df.head(10).to_string(index=False))
    print()
    print("Summary stats:")
    print(df[[
        "household_load_kw", "battery_power_kw", "battery_soc_pct",
        "line_voltage_v", "grid_voltage_v",
    ]].describe().round(3))
