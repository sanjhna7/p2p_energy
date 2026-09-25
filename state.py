import sqlite3
from database import get_connection
from grid_model import NOMINAL_VOLTAGE_PU


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

    Every numeric field returned here is guaranteed non-NULL, even when
    this timestamp hasn't been simulated yet (e.g. a `pre_state` snapshot
    taken before simulate_step runs) - callers that turn this into a
    feature vector/tensor should never have to special-case None:
      - soc_kwh / soc_pct / charge_kw default to 0.0 (empty, idle
        battery) whether that's because the house has no battery at all
        or because this timestamp just hasn't been simulated yet - same
        convention _previous_battery_soc already uses for "no history".
      - voltage_pu / transformer_loading_pct / losses_kw default to
        nominal/no-flow (as if net_injection_kw were 0) when this bus
        hasn't been simulated for this timestamp yet.

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
            COALESCE(b.soc_kwh,0) AS soc_kwh,
            COALESCE(b.soc_pct,0) AS soc_pct,
            COALESCE(b.charge_kw,0) AS charge_kw,
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
        if bus:
            house["voltage_pu"] = bus["voltage_pu"]
            house["transformer_loading_pct"] = bus["transformer_loading_pct"]
            house["losses_kw"] = bus["losses_kw"]
        else:
            # Not simulated yet for this timestamp - assume nominal,
            # no-flow conditions (equivalent to net_injection_kw == 0)
            # rather than leaving these None for downstream consumers.
            house["voltage_pu"] = NOMINAL_VOLTAGE_PU
            house["transformer_loading_pct"] = 0.0
            house["losses_kw"] = 0.0

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