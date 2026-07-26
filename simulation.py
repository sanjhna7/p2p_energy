"""
Simulation step: for a given timestamp, compute battery SoC and bus
voltage for every house/bus, persist them, then hand the assembled
state to the (still placeholder) RL agent.

This file is deliberately thin. All actual physics live in
battery_model.py / grid_model.py so they can be tested and swapped
independently (see the module docstrings there for why).
"""

from collections import defaultdict
import uuid

from database import get_connection
from state import get_state
import battery_model
import grid_model


def _previous_battery_soc(conn, house_id, timestamp):
    """Look up this house's SoC from the most recent prior timestamp.
    Falls back to 0 (empty battery) if there's no history yet."""

    row = conn.execute("""
        SELECT soc_kwh FROM battery_state
        WHERE house_id = ? AND timestamp < ?
        ORDER BY timestamp DESC LIMIT 1
    """, (house_id, timestamp)).fetchone()

    return row[0] if row else 0.0


def simulate_step(timestamp):

    conn = get_connection()

    houses = conn.execute("""
        SELECT house_id, bus_id, has_battery, battery_capacity_kwh,
               battery_max_charge_kw, battery_max_discharge_kw
        FROM houses
    """).fetchall()

    meter = dict(conn.execute("""
        SELECT house_id, solar_kw - load_kw AS net_kw
        FROM meter_readings WHERE timestamp = ?
    """, (timestamp,)).fetchall())

    residual_by_bus = defaultdict(float)
    load_by_bus = defaultdict(float)

    battery_rows = []

    for house_id, bus_id, has_battery, capacity_kwh, max_charge_kw, max_discharge_kw in houses:

        net_kw = meter.get(house_id, 0.0)

        if has_battery and capacity_kwh:
            prev_soc = _previous_battery_soc(conn, house_id, timestamp)

            result = battery_model.step_battery(
                prev_soc_kwh=prev_soc,
                capacity_kwh=capacity_kwh,
                net_kw=net_kw,
                max_charge_kw=max_charge_kw or capacity_kwh,
                max_discharge_kw=max_discharge_kw or capacity_kwh,
            )

            residual_kw = result.residual_kw

            # residual_kw is stored, not just handed to grid_model: it is
            # the power this house actually exchanges with the grid, so
            # it is what state.py must report to an agent.
            battery_rows.append((
                timestamp, house_id,
                result.soc_kwh, result.soc_pct, result.charge_kw, residual_kw,
            ))
        else:
            residual_kw = net_kw

        residual_by_bus[bus_id] += residual_kw
        load_by_bus[bus_id] += abs(net_kw)

    conn.executemany("""
        INSERT OR REPLACE INTO battery_state
        (timestamp, house_id, soc_kwh, soc_pct, charge_kw, residual_kw)
        VALUES (?,?,?,?,?,?)
    """, battery_rows)

    grid_rows = []
    for bus_id, residual_kw in residual_by_bus.items():
        bus_result = grid_model.compute_bus_state(residual_kw, load_by_bus[bus_id])
        grid_rows.append((
            timestamp, bus_id,
            bus_result.voltage_pu,
            bus_result.transformer_loading_pct,
            bus_result.losses_kw,
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO grid_state
        (timestamp, bus_id, voltage_pu, transformer_loading_pct, losses_kw)
        VALUES (?,?,?,?,?)
    """, grid_rows)

    conn.commit()
    conn.close()

    # Now the full state (solar/load/battery/voltage) can be read back
    # through the normal data-layer API for whatever consumes it next
    # (RL agent placeholder for now).
    state = get_state(timestamp)

    episode_id = str(uuid.uuid4())
    action = {"house_1": "export_surplus"}  # RL agent placeholder
    reward = 0.5                             # RL agent placeholder

    conn = get_connection()
    conn.execute("""
        INSERT INTO transitions
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        episode_id,
        0,
        str(state),
        str(action),
        reward,
        str(state),
    ))
    conn.commit()
    conn.close()

    print(f"Step complete for {timestamp}: "
          f"{len(battery_rows)} batteries, {len(grid_rows)} buses updated.")


if __name__ == "__main__":
    simulate_step("2010-07-15 13:00:00")
