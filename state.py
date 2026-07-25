import sqlite3
from database import get_connection


def get_state(timestamp):

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            h.house_id,
            h.bus_id,
            COALESCE(m.solar_kw,0) AS solar_kw,
            COALESCE(m.load_kw,0) AS load_kw

        FROM houses h

        LEFT JOIN meter_readings m

        ON h.house_id = m.house_id

        AND m.timestamp = ?
    """, (timestamp,)).fetchall()

    conn.close()

    houses = []

    for row in rows:

        house = dict(row)

        house["net_kw"] = round(
            house["solar_kw"] - house["load_kw"],
            3
        )

        if house["net_kw"] > 0:
            house["status"] = "surplus"
        elif house["net_kw"] < 0:
            house["status"] = "deficit"
        else:
            house["status"] = "balanced"

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