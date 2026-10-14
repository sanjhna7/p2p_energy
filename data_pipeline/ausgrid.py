"""Ausgrid Solar Home Electricity Data -> long, per-house power time series.

Source facts (from "Ausgrid solar home electricity data notes (Aug 2014).pdf"):
  - 300 gross-metered solar homes, Ausgrid network area (Sydney region, NSW)
  - GC = general consumption, CL = controlled load (off-peak tariff),
    GG = gross PV generation, metered separately from household load
  - each value is kWh in the half hour ENDING at the column time
    ("0:30" = 0:00-0:30, "0:00" = 23:30-24:00)
  - times are EST, and EDT during daylight saving (48 columns every day)
  - "Row Quality" NA = some values estimated/substituted (2011-12, 2012-13 only)

What this module does NOT have, because the dataset does not contain it:
  battery power, battery SoC, voltage, current, grid import/export
  as metered with a battery, network location. Those columns are emitted
  as empty (NA) so nobody mistakes a modelled value for a measured one.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INTERVAL_H = 0.5
KWH_TO_KW = 1.0 / INTERVAL_H  # average power over a 30-min interval

# Intervals ending 0:30 .. 4:00 cover 00:00-04:00, when PV output must be ~0.
NIGHT_COLUMNS = ["0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00"]


@dataclass
class CleaningRules:
    """Customer exclusion rules. These are OUR rules, documented in
    docs/dataset_decision.md; they are not Ratnam et al.'s cleaning."""

    exclude_row_quality_na: bool = True
    # A whole day with zero gross generation is taken as a meter/inverter outage.
    max_zero_generation_days: int = 0
    # Generation above this (kWh per half hour) between 00:00 and 04:00 is
    # physically impossible for rooftop PV and indicates a metering anomaly.
    night_generation_threshold_kwh: float = 0.01
    max_night_generation_days: int = 0


@dataclass
class AusgridYear:
    wide: pd.DataFrame
    time_columns: list
    has_row_quality: bool
    source_file: str
    quality: pd.DataFrame = field(default=None)


def read_wide(csv_path):
    """Read one yearly Ausgrid CSV (first line is a title row)."""
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    has_row_quality = "Row Quality" in df.columns
    time_columns = [c for c in df.columns if ":" in c]
    if len(time_columns) != 48:
        raise ValueError(f"{csv_path}: expected 48 half-hour columns, found {len(time_columns)}")

    # 2010-11 uses "1-Jul-10", later years use "1/07/2011"; both are day-first.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["Consumption Category"] = df["Consumption Category"].str.strip()
    return AusgridYear(df, time_columns, has_row_quality, str(csv_path))


def quality_report(year, rules=None):
    """One row per customer with the checks behind each exclusion decision."""
    rules = rules or CleaningRules()
    df = year.wide
    values = df[year.time_columns]

    gg = df[df["Consumption Category"] == "GG"]
    gg_daily = values.loc[gg.index].sum(axis=1)
    night_max = values.loc[gg.index, NIGHT_COLUMNS].max(axis=1)

    per_customer = pd.DataFrame(
        {
            "customer": gg["Customer"].values,
            "zero_gen_day": (gg_daily == 0).values,
            "night_gen_day": (night_max > rules.night_generation_threshold_kwh).values,
            "gen_kwh": gg_daily.values,
        }
    ).groupby("customer").agg(
        zero_generation_days=("zero_gen_day", "sum"),
        night_generation_days=("night_gen_day", "sum"),
        annual_generation_kwh=("gen_kwh", "sum"),
    )

    meta = df.groupby("Customer").agg(
        pv_capacity_kwp=("Generator Capacity", "max"),
        postcode=("Postcode", "first"),
    )
    categories = df.groupby("Customer")["Consumption Category"].agg(lambda s: set(s))
    meta["has_controlled_load"] = categories.apply(lambda s: "CL" in s)
    meta["has_general_consumption"] = categories.apply(lambda s: "GC" in s)
    meta["has_generation"] = categories.apply(lambda s: "GG" in s)

    consumption = df[df["Consumption Category"].isin(["GC", "CL"])]
    meta["annual_consumption_kwh"] = (
        values.loc[consumption.index].sum(axis=1).groupby(consumption["Customer"]).sum()
    )

    days = df.groupby(["Customer", "Consumption Category"])["date"].nunique().unstack()
    meta["gc_days"] = days.get("GC")
    meta["gg_days"] = days.get("GG")

    if year.has_row_quality:
        na_rows = df["Row Quality"].astype("string").str.strip().eq("NA")
        meta["row_quality_na_rows"] = na_rows.groupby(df["Customer"]).sum()
    else:
        meta["row_quality_na_rows"] = 0

    report = meta.join(per_customer, how="left")
    report["zero_generation_days"] = report["zero_generation_days"].fillna(0).astype(int)
    report["night_generation_days"] = report["night_generation_days"].fillna(0).astype(int)
    report["specific_yield_kwh_per_kwp"] = (
        report["annual_generation_kwh"] / report["pv_capacity_kwp"]
    )

    expected_days = df["date"].nunique()
    reasons = pd.Series("", index=report.index, dtype="string")

    def flag(mask, text):
        nonlocal reasons
        reasons = reasons.where(~mask, reasons + text + ";")

    flag(~report["has_general_consumption"] | ~report["has_generation"], "missing_GC_or_GG")
    flag((report["gc_days"] < expected_days) | (report["gg_days"] < expected_days), "incomplete_days")
    if rules.exclude_row_quality_na:
        flag(report["row_quality_na_rows"] > 0, "row_quality_NA")
    flag(report["zero_generation_days"] > rules.max_zero_generation_days, "zero_generation_days")
    flag(report["night_generation_days"] > rules.max_night_generation_days, "night_generation")

    report["exclusion_reasons"] = reasons.str.rstrip(";")
    report["included"] = report["exclusion_reasons"] == ""
    report.index.name = "customer"
    year.quality = report
    return report


def to_long(year, customers=None):
    """Wide Ausgrid rows -> one row per (customer, half-hour interval).

    Energy (kWh per 30 min) is converted to average power (kW) by x2.
    Timestamps are naive local wall-clock time as recorded (EST/EDT); the
    `timestamp` column is the START of the interval, `interval_end` the end.
    """
    df = year.wide
    if customers is not None:
        df = df[df["Customer"].isin(customers)]

    long = df.melt(
        id_vars=["Customer", "Consumption Category", "date"],
        value_vars=year.time_columns,
        var_name="time_of_day",
        value_name="kwh",
    )

    end_offsets = {}
    for label in year.time_columns:
        h, m = map(int, label.split(":"))
        if h == 0 and m == 0:
            h = 24
        end_offsets[label] = pd.Timedelta(hours=h, minutes=m)

    long["interval_end"] = long["date"] + long["time_of_day"].map(end_offsets)
    long["timestamp"] = long["interval_end"] - pd.Timedelta(hours=INTERVAL_H)

    wide_by_cat = long.pivot_table(
        index=["Customer", "timestamp", "interval_end"],
        columns="Consumption Category",
        values="kwh",
        aggfunc="first",
    ).reset_index()

    gc = wide_by_cat["GC"] if "GC" in wide_by_cat else np.nan
    gg = wide_by_cat["GG"] if "GG" in wide_by_cat else np.nan
    # A customer without CL rows is not on a controlled-load tariff in that
    # year, so controlled-load consumption is zero rather than unknown.
    cl = wide_by_cat["CL"].fillna(0.0) if "CL" in wide_by_cat else 0.0

    out = pd.DataFrame(
        {
            "timestamp": wide_by_cat["timestamp"],
            "interval_end": wide_by_cat["interval_end"],
            "ausgrid_customer_id": wide_by_cat["Customer"].astype(int),
            "load_gc_kw": gc * KWH_TO_KW,
            "load_cl_kw": cl * KWH_TO_KW,
            "pv_power_kw": gg * KWH_TO_KW,
        }
    )
    out["load_power_kw"] = out["load_gc_kw"] + out["load_cl_kw"]
    return out.sort_values(["ausgrid_customer_id", "timestamp"]).reset_index(drop=True)
