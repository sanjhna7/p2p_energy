"""
Real Ausgrid data loading.

This is the ONLY place the provisional dataset touches real data, and
it is deliberately the only place that does no modelling at all: it
returns the untouched Ausgrid temporal profiles (as average kW) plus
each customer's installed Generator Capacity, and nothing else.

Source mapping, per "Ausgrid solar home electricity data notes":
    GG        -> gross PV generation
    GC + CL   -> household consumption (general + controlled load)

UNIT CONVERSION
---------------
The CSV records ENERGY: each of the 48 daily columns is the kWh in the
half hour ENDING at that column's time. Everything downstream works in
average POWER, so:

    P(kW) = E(kWh) / interval_hours = E(kWh) / 0.5 = 2 x E(kWh)

The project's SQLite database already stores meter_readings in average
kW (removedfornnow/ingest.py applied exactly this conversion), so the
DB path needs no
further scaling. The CSV fallback applies it explicitly.
"""

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from .config import NanogridConfig

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

    Mirrors removedfornnow/ingest.py's reshaping (wide half-hourly columns -> long ->
    pivot on Consumption Category), then applies the kWh -> kW
    conversion explicitly.
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
