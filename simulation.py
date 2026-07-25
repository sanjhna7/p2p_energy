from database import get_connection
from state import get_state
import uuid


def simulate_step(timestamp):

    # Get current state
    state = get_state(timestamp)

    # Generate a unique episode ID
    episode_id = str(uuid.uuid4())

    # RL Agent picks an action (placeholder)
    action = {"house_1": "export_surplus"}

    # Power Flow Simulation (placeholder)
    new_voltage = 0.98
    losses = 1.2

    reward = 0.5

    conn = get_connection()

    # Store grid state
    conn.execute("""
        INSERT INTO grid_state
        (timestamp, bus_id, voltage_pu, losses_kw)
        VALUES (?, ?, ?, ?)
    """, (
        timestamp,
        "bus_1",
        new_voltage,
        losses
    ))

    # Store transition
    conn.execute("""
        INSERT INTO transitions
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        episode_id,
        0,
        str(state),
        str(action),
        reward,
        str(get_state("2010-07-15 13:30:00"))
    ))

    conn.commit()
    conn.close()

    print("Step complete.")


if __name__ == "__main__":
    simulate_step("2010-07-15 13:00:00")