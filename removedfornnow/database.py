import sqlite3

DB_PATH = "data/raw/p2p_energy.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS houses(
    house_id TEXT PRIMARY KEY,
    bus_id TEXT,
    has_solar INTEGER,
    panel_capacity_kw REAL,
    has_battery INTEGER DEFAULT 0,
    battery_capacity_kwh REAL DEFAULT 0,
    battery_max_charge_kw REAL DEFAULT 0,
    battery_max_discharge_kw REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meter_readings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    house_id TEXT,
    solar_kw REAL,
    load_kw REAL
);

-- One reading per house per timestamp, so re-running the import
-- cannot duplicate rows.
CREATE UNIQUE INDEX IF NOT EXISTS idx_meter_readings_unique
    ON meter_readings(timestamp, house_id);

CREATE TABLE IF NOT EXISTS grid_state(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    bus_id TEXT,
    voltage_pu REAL,
    transformer_loading_pct REAL,
    losses_kw REAL
);

-- One computed grid reading per bus per timestamp.
CREATE UNIQUE INDEX IF NOT EXISTS idx_grid_state_unique
    ON grid_state(timestamp, bus_id);

-- Battery state of charge, recomputed every simulation step.
-- Kept as a time series (not just "current value") so RL replay /
-- auditing can look back at how SoC evolved.
CREATE TABLE IF NOT EXISTS battery_state(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    house_id TEXT,
    soc_kwh REAL,
    soc_pct REAL,
    charge_kw REAL,         -- positive = charging, negative = discharging
    residual_kw REAL        -- power left over for the grid after the
                            -- battery acted; this is what the bus sees
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_battery_state_unique
    ON battery_state(timestamp, house_id);

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

# Unique indexes declared in SCHEMA, and the columns they cover.
# A database created before an index existed may already hold duplicate
# rows; CREATE UNIQUE INDEX fails outright on those, so they have to be
# cleared out first.
UNIQUE_KEYS = {
    "idx_meter_readings_unique": ("meter_readings", ("timestamp", "house_id")),
    "idx_grid_state_unique": ("grid_state", ("timestamp", "bus_id")),
    "idx_battery_state_unique": ("battery_state", ("timestamp", "house_id")),
}

# Columns added after a table's first release. CREATE TABLE IF NOT
# EXISTS is a no-op on an existing table, so it will never add these -
# they need ALTER TABLE. Append here when a table grows again.
ADDED_COLUMNS = {
    "houses": (
        ("has_battery", "INTEGER DEFAULT 0"),
        ("battery_capacity_kwh", "REAL DEFAULT 0"),
        ("battery_max_charge_kw", "REAL DEFAULT 0"),
        ("battery_max_discharge_kw", "REAL DEFAULT 0"),
    ),
    "battery_state": (
        ("residual_kw", "REAL"),
    ),
}

# Length of one Ausgrid metering interval. Per "Ausgrid solar home
# electricity data notes (Aug 2014).pdf", each CSV value is the kWh
# consumed/generated "in the half hour ending at" that column's time.
METER_INTERVAL_HOURS = 0.5

# Bumped whenever stored *values* change meaning, or a new column needs
# populating from existing rows.
#   0 -> meter_readings held raw kWh-per-interval from the CSV
#   1 -> meter_readings hold average kW
#   2 -> battery_state.residual_kw populated
# Tracked with SQLite's built-in PRAGMA user_version, which costs no
# extra table.
SCHEMA_VERSION = 2


def get_connection():
    return sqlite3.connect(DB_PATH)


def _exists(conn, kind, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type=? AND name=?",
        (kind, name)
    ).fetchone() is not None


def _drop_duplicate_rows(conn):
    """Clear duplicates so the unique indexes in SCHEMA can be created.
    Keeps the earliest row (lowest rowid) of each group."""

    for index_name, (table, key_cols) in UNIQUE_KEYS.items():

        if not _exists(conn, "table", table):
            continue

        if _exists(conn, "index", index_name):
            # The index is already enforcing uniqueness, so duplicates
            # cannot exist. Skipping also avoids a full scan of a large
            # table on every startup.
            continue

        keys = ", ".join(key_cols)

        removed = conn.execute(f"""
            DELETE FROM {table}
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM {table} GROUP BY {keys}
            )
        """).rowcount

        if removed:
            print(f"  Removed {removed} duplicate row(s) from {table}.")


def _add_missing_columns(conn):
    """Bring existing tables up to the current schema."""

    for table, columns in ADDED_COLUMNS.items():

        if not _exists(conn, "table", table):
            continue

        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

        for column, declaration in columns:

            if column not in existing:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                )
                print(f"  Added column {table}.{column}.")


def _convert_readings_to_kw(conn):
    """Rescale meter_readings written before SCHEMA_VERSION 1.

    Those rows hold the CSV's raw kWh-per-interval values. ingest.py now
    converts to average kW on the way in, but it inserts with
    INSERT OR IGNORE - existing rows are skipped, so they would keep
    their old (halved) meaning forever unless corrected here.
    """

    converted = conn.execute(
        "UPDATE meter_readings SET solar_kw = solar_kw / ?, load_kw = load_kw / ?",
        (METER_INTERVAL_HOURS, METER_INTERVAL_HOURS)
    ).rowcount

    if converted:
        print(f"  Converted {converted} meter_readings row(s) "
              f"from kWh/interval to average kW.")


def _backfill_residual_kw(conn):
    """Populate residual_kw on battery_state rows written before the
    column existed.

    Those rows were produced by the same battery model, which leaves
    whatever it did not charge/discharge to the grid, so the residual is
    recoverable exactly:  residual = (solar - load) - charge_kw.
    Without this the rows would read as NULL and state.py would fall
    back to the pre-battery figure - the very thing this column fixes.
    """

    filled = conn.execute("""
        UPDATE battery_state
        SET residual_kw = (
            SELECT m.solar_kw - m.load_kw - battery_state.charge_kw
            FROM meter_readings m
            WHERE m.house_id = battery_state.house_id
              AND m.timestamp = battery_state.timestamp
        )
        WHERE residual_kw IS NULL
    """).rowcount

    if filled:
        print(f"  Backfilled residual_kw on {filled} battery_state row(s).")


def _apply_data_migrations(conn):
    """Run each data migration once, tracked by PRAGMA user_version."""

    version = conn.execute("PRAGMA user_version").fetchone()[0]

    if version >= SCHEMA_VERSION:
        return

    if version < 1:
        _convert_readings_to_kw(conn)

    if version < 2:
        _backfill_residual_kw(conn)

    # Recording the version is what makes these run exactly once.
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def initialize_database():
    conn = get_connection()

    # Order matters: duplicates must go before the unique indexes are
    # built, columns can only be added once their table exists, and the
    # migrations below write to those new columns.
    _drop_duplicate_rows(conn)
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    _apply_data_migrations(conn)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_database()
    print("Database created successfully!")