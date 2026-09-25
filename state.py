import sqlite3
from database import get_connection


def get_state(timestamp):
    """
    Single read API for the data layer. Assembles, per house:
    solar/load (from meter_readings), battery SoC (from battery_state,
    if already computed for this timestamp), and net/status.

    `net_kw` and `status` are POST-battery: they describe the power the
    house exchanges with the grid after charging/discharging, which is
    the true environment an agent observes. The pre-battery meter
    difference is still available as `gross_net_kw`.

    Bus-level grid state (voltage, transformer loading, losses) is
    attached separately via `bus_id` lookups against grid_state.

    Callers (simulation.py, RL agent, later GNN) should only ever go
    through this function - never query meter_readings/battery_state/
    grid_state directly.
    """

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            h.house_id,
            h.bus_id,
            h.has_battery,
            h.battery_capacity_kwh,
            COALESCE(m.solar_kw,0) AS solar_kw,
            COALESCE(m.load_kw,0) AS load_kw,
            b.soc_kwh,
            b.soc_pct,
            b.charge_kw,
            b.residual_kw

        FROM houses h

        LEFT JOIN meter_readings m
            ON h.house_id = m.house_id
            AND m.timestamp = ?

        LEFT JOIN battery_state b
            ON h.house_id = b.house_id
            AND b.timestamp = ?
    """, (timestamp, timestamp)).fetchall()

    bus_rows = conn.execute("""
        SELECT bus_id, voltage_pu, transformer_loading_pct, losses_kw
        FROM grid_state
        WHERE timestamp = ?
    """, (timestamp,)).fetchall()

    conn.close()

    bus_lookup = {r["bus_id"]: dict(r) for r in bus_rows}

    houses = []

    for row in rows:

        house = dict(row)

        # Raw meter difference, before the battery does anything.
        gross_net_kw = house["solar_kw"] - house["load_kw"]
        house["gross_net_kw"] = round(gross_net_kw, 3)

        # What the house actually exchanges with the grid once the
        # battery has charged/discharged - this is the observable an
        # agent must act on, so it is what `net_kw` and `status` report.
        # A missing residual means nothing buffered the flow (no battery
        # installed, or this timestamp has not been simulated yet), and
        # in both cases the residual is simply the gross figure.
        residual_kw = house.pop("residual_kw")

        if residual_kw is None:
            residual_kw = gross_net_kw

        house["net_kw"] = round(residual_kw, 3)

        if house["net_kw"] > 0:
            house["status"] = "surplus"
        elif house["net_kw"] < 0:
            house["status"] = "deficit"
        else:
            house["status"] = "balanced"

        bus = bus_lookup.get(house["bus_id"])
        house["voltage_pu"] = bus["voltage_pu"] if bus else None
        house["transformer_loading_pct"] = bus["transformer_loading_pct"] if bus else None
        house["losses_kw"] = bus["losses_kw"] if bus else None

        houses.append(house)

    return {
        "timestamp": timestamp,
        "houses": houses
    }


if __name__ == "__main__":

    state = get_state("2010-07-15 13:00:00")

    print("Timestamp :", state["timestamp"])
    print("Total Houses :", len(state["houses"]))

    print("\nFirst 5 Houses:\n")

    for house in state["houses"][:5]:
        print(house)