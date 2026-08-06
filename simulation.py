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


def _previous_battery_socs(conn, timestamp):
    """Look up every house's SoC from its most recent prior timestamp in
    one query, instead of one query per house. Falls back to 0 (empty
    battery) for any house with no history yet - callers should use
    .get(house_id, 0.0) on the returned dict.

    Same result as calling the old per-house lookup in a loop, but one
    round-trip to SQLite instead of one per house per step - matters
    once this runs across many houses x many timestamps in a training
    loop."""

    rows = conn.execute("""
        SELECT bs.house_id, bs.soc_kwh
        FROM battery_state bs
        INNER JOIN (
            SELECT house_id, MAX(timestamp) AS max_ts
            FROM battery_state
            WHERE timestamp < ?
            GROUP BY house_id
        ) latest
        ON bs.house_id = latest.house_id AND bs.timestamp = latest.max_ts
    """, (timestamp,)).fetchall()

    return dict(rows)


def simulate_step(timestamp, episode_id=None, step=0):
    """
    Run one physics step and log it as an RL transition (s, a, r, s').

    episode_id / step let a caller stitch multiple calls into a single
    trajectory (e.g. a training loop iterating over timestamps for one
    episode). If episode_id is omitted, a fresh one is generated - fine
    for a standalone step, but a real multi-step episode should
    generate one episode_id up front and pass it in on every call,
    incrementing step each time. Returns episode_id so the caller can
    reuse it on the next call.
    """

    if episode_id is None:
        episode_id = str(uuid.uuid4())

    conn = get_connection()

    # PRE-action state: captured before this step's battery_state/
    # grid_state rows are written, so get_state() falls back to gross
    # meter figures (no battery result yet, no bus voltage yet) - this
    # is what was actually observed before acting, i.e. `state` in
    # (s, a, r, s'). NOTE: if this timestamp was already simulated in a
    # previous run, this will reflect that prior run's result rather
    # than a true "before" - fine for the normal forward-simulation
    # case, worth knowing if you ever re-run a timestamp.
    pre_state = get_state(timestamp)

    houses = conn.execute("""
        SELECT house_id, bus_id, has_battery, battery_capacity_kwh,
               battery_max_charge_kw, battery_max_discharge_kw
        FROM houses
    """).fetchall()

    meter = dict(conn.execute("""
        SELECT house_id, solar_kw - load_kw AS net_kw
        FROM meter_readings WHERE timestamp = ?
    """, (timestamp,)).fetchall())

    prev_socs = _previous_battery_socs(conn, timestamp)

    residual_by_bus = defaultdict(float)

    battery_rows = []

    for house_id, bus_id, has_battery, capacity_kwh, max_charge_kw, max_discharge_kw in houses:

        net_kw = meter.get(house_id, 0.0)

        if has_battery and capacity_kwh:
            prev_soc = prev_socs.get(house_id, 0.0)

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

    conn.executemany("""
        INSERT OR REPLACE INTO battery_state
        (timestamp, house_id, soc_kwh, soc_pct, charge_kw, residual_kw)
        VALUES (?,?,?,?,?,?)
    """, battery_rows)

    grid_rows = []
    for bus_id, residual_kw in residual_by_bus.items():
        bus_result = grid_model.compute_bus_state(residual_kw)
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

    # POST-action state: battery_state/grid_state rows now exist for
    # this timestamp, so get_state() returns the actual post-step
    # observation - this is `next_state` in (s, a, r, s').
    next_state = get_state(timestamp)

    action = {"house_1": "export_surplus"}  # RL agent placeholder
    reward = 0.5                             # RL agent placeholder

    conn.execute("""
        INSERT INTO transitions
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        episode_id,
        step,
        str(pre_state),
        str(action),
        reward,
        str(next_state),
    ))
    conn.commit()
    conn.close()

    print(f"Step complete for {timestamp}: "
          f"{len(battery_rows)} batteries, {len(grid_rows)} buses updated. "
          f"(episode={episode_id}, step={step})")

    return episode_id


if __name__ == "__main__":
    simulate_step("2010-07-15 13:00:00")