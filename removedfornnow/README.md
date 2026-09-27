# removedfornnow

This folder is intentionally kept as a small holding area for code that
is temporarily parked while the project is rebuilt around the provisional
MARL dataset.

At present, it contains only two files:

- `README.md` — notes about what is parked here and how to restore it
- `ingest.py` — legacy Ausgrid CSV ingestion script used to build the raw
  SQLite database

Nothing has been deleted from the repository. The rest of the older
SQLite pipeline has already been moved aside or superseded, and anything
here can be restored to the project root if needed.

## What is parked here

The remaining parked item is the older **SQLite ingestion pipeline** that
reads the Ausgrid CSV into `p2p_energy.db` and prepares the raw database
used by the earlier simulation stack.

| File | What it does | Why it is parked |
| --- | --- | --- |
| `README.md` | Explains the parked folder and restore steps | Keeps the archive's purpose clear without changing the active code path |
| `ingest.py` | Ausgrid CSV → `houses` + `meter_readings` (kWh → kW) | This is the only remaining legacy file in the folder; the current experiment no longer uses it directly |

## Does the experiment still work without these?

Yes, completely. `nanogrid/ausgrid_source.py` reads `meter_readings`
straight from `data/raw/p2p_energy.db`, and if that database is missing
it falls back to parsing `data/raw/Solar home 2010-2011.csv` directly.
Both paths have been verified to produce bit-identical output.

## Running the parked ingestion script

Paths inside `ingest.py` are relative to the **project root**, so run it
from there and not from inside this folder:

```bash
python removedfornnow/ingest.py
```

The script expects the data file at:

- `data/raw/Solar home 2010-2011.csv`

and writes the database to:

- `data/raw/p2p_energy.db`

## Restoring a file

```bash
git mv removedfornnow/<file>.py .
```

If the file is `ingest.py`, also restore the raw data path assumptions in
that script if you want to run it from the project root again.
