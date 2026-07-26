import pandas as pd
from database import get_connection, METER_INTERVAL_HOURS

CSV_FILE = "Solar home 2010-2011.csv"


def ingest_data():

    print("Reading CSV...")

    df = pd.read_csv(CSV_FILE, skiprows=1)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    id_cols = [
        "Customer",
        "Consumption Category",
        "date"
    ]

    time_cols = [
        c for c in df.columns
        if c not in id_cols + ["Generator Capacity", "Postcode"]
    ]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=time_cols,
        var_name="time_of_day",
        value_name="value"
    )

    long_df["date_dt"] = pd.to_datetime(
        long_df["date"],
        format="%d-%b-%y"
    )

    offsets = {}

    for t in long_df["time_of_day"].unique():

        h, m = map(int, t.split(":"))

        if h == 0 and m == 0:
            h = 24

        offsets[t] = pd.Timedelta(hours=h, minutes=m)

    long_df["timestamp"] = (
        long_df["date_dt"]
        + long_df["time_of_day"].map(offsets)
    )

    pivot = long_df.pivot_table(
        index=["Customer", "timestamp"],
        columns="Consumption Category",
        values="value",
        aggfunc="first"
    ).reset_index()

    pivot["load_kw"] = (
        pivot.get("GC", 0).fillna(0)
        + pivot.get("CL", 0).fillna(0)
    )

    pivot = pivot.rename(
        columns={
            "Customer": "house_id",
            "GG": "solar_kw"
        }
    )

    pivot["house_id"] = pivot["house_id"].astype(str)

    # UNITS: the Ausgrid CSV records ENERGY, not power - every interval
    # value is the kWh consumed/generated during that half hour (see
    # "Ausgrid solar home electricity data notes (Aug 2014).pdf",
    # columns 6-53). The rest of the system works in average POWER, so
    # convert once, here:
    #
    #     average_kW = kWh_in_interval / interval_hours
    #
    # i.e. a 0.5 h interval means multiplying by 2. Doing it at ingest
    # is what keeps meter_readings.solar_kw / load_kw honest to their
    # names: everything downstream (battery_model, grid_model) then
    # reads real kW and applies its own dt_hours for energy, with no
    # second interval factor sneaking in.
    pivot["solar_kw"] = pivot["solar_kw"] / METER_INTERVAL_HOURS
    pivot["load_kw"] = pivot["load_kw"] / METER_INTERVAL_HOURS

    # Installed PV capacity per house, taken from the CSV instead of
    # being hardcoded. Each customer carries one capacity value.
    # Already in kW (the CSV records kWp, peak power), so no conversion.
    capacity_by_house = df.groupby("Customer")["Generator Capacity"].max()
    capacity_by_house.index = capacity_by_house.index.astype(str)

    conn = get_connection()

    for hid in pivot["house_id"].unique():

        capacity = capacity_by_house.get(hid)

        if capacity is None or pd.isna(capacity):
            capacity = 0.0

        capacity = float(capacity)

        has_solar = 1 if capacity > 0 else 0

        # NOTE: the Ausgrid CSV has no battery data at all. These are
        # simulation-baseline assumptions (only solar houses get a
        # battery, sized off panel capacity), not measured values.
        # Tune/replace once real battery specs are available.
        has_battery = has_solar
        battery_capacity_kwh = round(capacity * 2, 2) if has_battery else 0.0
        battery_max_charge_kw = round(capacity, 2) if has_battery else 0.0
        battery_max_discharge_kw = round(capacity, 2) if has_battery else 0.0

        conn.execute(
            """
            INSERT INTO houses
            (house_id, bus_id, has_solar, panel_capacity_kw,
             has_battery, battery_capacity_kwh,
             battery_max_charge_kw, battery_max_discharge_kw)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(house_id) DO UPDATE SET
                has_solar = excluded.has_solar,
                panel_capacity_kw = excluded.panel_capacity_kw,
                has_battery = excluded.has_battery,
                battery_capacity_kwh = excluded.battery_capacity_kwh,
                battery_max_charge_kw = excluded.battery_max_charge_kw,
                battery_max_discharge_kw = excluded.battery_max_discharge_kw
            """,
            (
                hid,
                f"bus_{hid}",
                has_solar,
                capacity,
                has_battery,
                battery_capacity_kwh,
                battery_max_charge_kw,
                battery_max_discharge_kw,
            )
        )

    # Same text format the timestamps were previously stored in,
    # so existing rows are recognised as duplicates.
    pivot["timestamp"] = pivot["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    rows = pivot[
        [
            "timestamp",
            "house_id",
            "solar_kw",
            "load_kw"
        ]
    ].itertuples(index=False, name=None)

    before = conn.total_changes

    conn.executemany(
        """
        INSERT OR IGNORE INTO meter_readings
        (timestamp, house_id, solar_kw, load_kw)
        VALUES (?,?,?,?)
        """,
        rows
    )

    inserted = conn.total_changes - before

    conn.commit()
    conn.close()

    print("--------------------------------")
    print("CSV Imported Successfully")
    print(f"Total Meter Readings : {len(pivot)}")
    print(f"New Readings Inserted : {inserted}")
    print(f"Total Houses : {pivot['house_id'].nunique()}")


if __name__ == "__main__":
    ingest_data()