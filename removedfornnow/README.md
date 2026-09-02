# removedfornnow

Files parked here are **not used by the current experiment** (the
provisional MARL development dataset). Nothing was deleted — everything
is intact, with git history preserved, and any of it can be moved back
to the project root.

## What is here, and why it was parked

These files are the earlier **SQLite simulation pipeline**. It reads the
Ausgrid CSV into `p2p_energy.db`, then steps a battery and a per-unit
bus-voltage proxy over that database, writing results back into it.

| File | What it does | Why it is parked |
| --- | --- | --- |
| `main.py` | Driver: init DB → ingest CSV → read state → one sim step | The experiment's entry point is `build_marl_dataset.py` |
| `database.py` | SQLite schema, migrations, `get_connection()` | The dataset generator opens the DB read-only with plain `sqlite3` |
| `ingest.py` | Ausgrid CSV → `houses` + `meter_readings` (kWh → kW) | **This is what built `data/raw/p2p_energy.db`.** Only needed to rebuild it |
| `state.py` | Assembles per-house state dicts from the DB | Superseded by `nanogrid/build_dataset.py` |
| `simulation.py` | One timestep across all houses, writes `battery_state` / `grid_state` | Superseded by the `nanogrid` time loop |
| `grid_model.py` | Per-unit AC-style droop proxy (`voltage_pu`, transformer loading, losses) | The experiment needs **DC bus volts**, so `nanogrid/models.py` replaces it |
| `check_db.py` | Ad-hoc `SELECT`s against `houses` / `meter_readings` | Inspection script |
| `check_simulation.py` | Ad-hoc `SELECT`s against `grid_state` / `transitions` | Inspection script |

## What was NOT parked

`battery_model.py` moved to **`nanogrid/battery_model.py`**, not here. It
is the one piece of the old pipeline the experiment still uses:
`nanogrid/models.py` builds its provisional battery on top of
`step_battery()`, so the repository keeps a single set of charge and
discharge equations.

## Does the experiment still work without these?

Yes, completely. `nanogrid/ausgrid_source.py` reads `meter_readings`
straight from `data/raw/p2p_energy.db`, and if that database is missing
it falls back to parsing `data/raw/Solar home 2010-2011.csv` directly.
Both paths have been verified to produce bit-identical output.

## Running the parked pipeline

Paths inside these files are relative to the **project root**, so run
them from there and not from inside this folder:

```bash
python removedfornnow/main.py          # full old pipeline
python removedfornnow/ingest.py        # rebuild data/raw/p2p_energy.db
python removedfornnow/check_db.py      # inspect the database
```

Two paths were updated when the files moved, and nothing else changed:

- `database.py` → `DB_PATH = "data/raw/p2p_energy.db"`
- `ingest.py` → `CSV_FILE = "data/raw/Solar home 2010-2011.csv"`
- `simulation.py` → `from nanogrid import battery_model`

## Restoring a file

```bash
git mv removedfornnow/<file>.py .
```

Then undo the path edits above if you also move the raw data back.
