import pandas as pd
from database import get_connection

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

    conn = get_connection()

    for hid in pivot["house_id"].unique():

        conn.execute(
            "INSERT OR IGNORE INTO houses VALUES (?,?,?,?)",
            (
                hid,
                f"bus_{hid}",
                1,
                0
            )
        )

    pivot[
        [
            "timestamp",
            "house_id",
            "solar_kw",
            "load_kw"
        ]
    ].to_sql(
        "meter_readings",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()
    conn.close()

    print("--------------------------------")
    print("CSV Imported Successfully")
    print(f"Total Meter Readings : {len(pivot)}")
    print(f"Total Houses : {pivot['house_id'].nunique()}")


if __name__ == "__main__":
    ingest_data()