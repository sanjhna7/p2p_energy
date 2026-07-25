import sqlite3

DB_PATH = "p2p_energy.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS houses(
    house_id TEXT PRIMARY KEY,
    bus_id TEXT,
    has_solar INTEGER,
    panel_capacity_kw REAL
);

CREATE TABLE IF NOT EXISTS meter_readings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    house_id TEXT,
    solar_kw REAL,
    load_kw REAL
);

CREATE TABLE IF NOT EXISTS grid_state(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    bus_id TEXT,
    voltage_pu REAL,
    transformer_loading_pct REAL,
    losses_kw REAL
);

CREATE TABLE IF NOT EXISTS transitions(
    episode_id TEXT,
    step INTEGER,
    state TEXT,
    action TEXT,
    reward REAL,
    next_state TEXT,
    PRIMARY KEY(episode_id, step)
);
"""

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_database()
    print("Database created successfully!")