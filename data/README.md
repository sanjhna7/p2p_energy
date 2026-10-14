# data/

Raw sources and the processed MVP dataset for the GAT-MARL P2P project.
Why these sources were chosen: `docs/dataset_decision.md`. Full survey: `docs/dataset_research.md`.

Nothing under `raw/` or `processed/` is committed (see `data/.gitignore`). Everything is re-downloadable and reproducible.

## Layout

```
data/
├── raw/
│   ├── ausgrid_solar_home/        REAL data: Ausgrid Solar Home, 300 homes, 2010-2013
│   │   ├── Ausgrid_solar_home_data.zip
│   │   ├── Ausgrid solar home electricity data notes (Aug 2014).pdf   <- read this
│   │   ├── Solar home 2010-2011.csv
│   │   ├── Solar home 2011-2012.csv
│   │   ├── Solar home 2012-2013.csv
│   │   └── README.md               (archive note by P. Haessig)
│   ├── ieee_european_lv/          BENCHMARK network: IEEE European LV Test Feeder
│   │   ├── European_LV_Test_Feeder_v2.zip
│   │   ├── European_LV_CSV/        Lines, LineCodes, Loads, Buscoords, Transformer, Source, load shapes
│   │   ├── IEEE European LV Test Feeder - Draft.docx
│   │   └── Solutions/              OpenDSS + GridLAB-D reference results (AC)
│   └── p2p_energy.db               (older SQLite DB from ingest.py, untouched)
└── processed/
    └── mvp_ausgrid_eulv/           output of data_pipeline.build_mvp_dataset
```

## 1. Download

```bash
cd baseline_code/p2p_energy
mkdir -p data/raw/ausgrid_solar_home data/raw/ieee_european_lv

# Ausgrid Solar Home. The official Ausgrid page was withdrawn (HTTP 404 as of Sept 2026).
# Archive copy with the original notes PDF, made by Pierre Haessig (May 2026):
curl -L -o data/raw/ausgrid_solar_home/Ausgrid_solar_home_data.zip \
     https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip
(cd data/raw/ausgrid_solar_home && unzip -o Ausgrid_solar_home_data.zip)

# IEEE European LV Test Feeder (IEEE PES Test Feeder WG)
curl -L -o data/raw/ieee_european_lv/European_LV_Test_Feeder_v2.zip \
     https://cmte.ieee.org/pes-testfeeders/wp-content/uploads/sites/167/2017/08/European_LV_Test_Feeder_v2.zip
(cd data/raw/ieee_european_lv && unzip -o European_LV_Test_Feeder_v2.zip)
```

SHA-256 of the files we used (downloaded 2026-09-15):

| File | SHA-256 |
|---|---|
| `Ausgrid_solar_home_data.zip` | `5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82` |
| `Solar home 2010-2011.csv` (from the zip) | `2fc1f615e22fbd8ec350cdb19f7465d3a038453fc2ffd19473ef1bd38fd28289` |
| `European_LV_Test_Feeder_v2.zip` | `47fcb8188d958426cdeae225e3b82591c55a97e70e0e65dc6b7ce8e560446ddc` |

The `Solar home 2010-2011.csv` at the repo root is the same data. It differs from the archive copy only by line endings (LF vs CRLF): after stripping `\r` both hash to `94e8a6a2…f353`.

## 2. Build

```bash
.venv/bin/python -m data_pipeline.build_mvp_dataset
# other year:
.venv/bin/python -m data_pipeline.build_mvp_dataset \
    --ausgrid-csv "data/raw/ausgrid_solar_home/Solar home 2011-2012.csv" \
    --out data/processed/mvp_ausgrid_eulv_2011
```

Needs only `pandas` and `numpy`. Takes about 15 s.

## 3. Output files (`processed/mvp_ausgrid_eulv/`)

### `houses_timeseries.csv.gz`: 55 houses × 17,520 half-hours (1 Jul 2010 – 30 Jun 2011)

| Column | Unit | Origin | Notes |
|---|---|---|---|
| `timestamp` | local time | REAL | Start of the 30-min interval. Wall-clock EST/EDT as recorded by Ausgrid. |
| `interval_end` | local time | REAL | End of the interval (Ausgrid column label). |
| `house_id` | – | OUR ADAPTATION | `H01`..`H55`, equal to feeder `LOAD1`..`LOAD55`. |
| `node_id` | – | OUR ADAPTATION | Node in `nodes.csv`. |
| `ausgrid_customer_id` | – | REAL | Original Ausgrid customer (1–300). |
| `load_power_kw` | kW | REAL (unit-converted) | `load_gc_kw + load_cl_kw`. |
| `load_gc_kw` | kW | REAL (unit-converted) | General consumption. kWh/30 min × 2. |
| `load_cl_kw` | kW | REAL (unit-converted) | Controlled (off-peak) load. 0 when the customer has no CL tariff. |
| `pv_power_kw` | kW | REAL (unit-converted) | Gross PV generation, metered separately. AC side of the inverter. |
| `net_load_kw_derived` | kW | DERIVED | `load − pv`. Counterfactual net demand with no battery and no trading. Not a meter reading. |
| `battery_power_kw` | kW | **UNAVAILABLE** (empty) | Not in the dataset. Produced later by the battery model. |
| `battery_soc` | 0–1 | **UNAVAILABLE** (empty) | Not in the dataset. Produced later by the battery model. |
| `grid_power_kw` | kW | **UNAVAILABLE** (empty) | Depends on battery and P2P decisions. Computed by the environment. |

Values are **30-minute averages of AC-side energy**. They are not instantaneous, not DC, and not voltage or current.

### `nodes.csv`: reduced feeder graph (110 nodes)

`node_id, source_bus, node_type (substation_lv | house | junction), load_name, load_phase, x, y, depth, distance_from_transformer_m, path_r1_ohm_from_transformer, neighbor_source_buses, neighbor_ids`

- 1 transformer LV bus, 55 houses (all leaves), 54 junctions (degree 3).
- `neighbor_ids` is `;`-separated. Neighbours are *electrical* neighbours in the radial tree.

### `edges.csv`: 109 edges (radial tree, oriented away from the transformer)

`edge_id, from_node_id, to_node_id, from_source_bus, to_source_bus, length_m, r1_ohm, x1_ohm, r0_ohm, x0_ohm, n_segments, line_codes`

- R and X are **AC sequence impedances per phase conductor**, summed over the merged segments (`LineCodes.csv` Ω/km × length).
- Median R1/X1 ≈ 13, so the cables are strongly resistive.
- **No DC conversion is applied.** Loop resistance, voltage re-rating and so on belong to the environment and are documented there.

### `edges_full.csv`: all 905 original segments, for traceability.

### `house_mapping.csv`
`house_id, node_id, feeder_load_name, feeder_source_bus, feeder_phase, ausgrid_customer_id, pv_capacity_kwp, postcode, has_controlled_load, annual_consumption_kwh, annual_generation_kwh`

### `ausgrid_quality_report.csv`
All 300 customers, with the checks and `exclusion_reasons`. For 2010-11: 269 clean customers.
- 30 excluded for whole days of zero PV generation (733 days in total).
- 1 excluded for PV "generation" between 00:00 and 04:00 (218 days).

### `manifest.json`
Source files with SHA-256, licences, citations, every adaptation step, cleaning rules and counts.

## 4. Check against the source

The pipeline reproduces Ausgrid's own summary table for 2010-11 over all 300 customers:

| Statistic | Ausgrid notes | Pipeline |
|---|---|---|
| mean annual consumption | 6,980 kWh | 6,980 kWh |
| mean annual gross generation | 2,119 kWh | 2,119 kWh |
| mean system size | 1.68 kWp | 1.68 kWp |
| mean kWh/kWp | 1,253 | 1,253 |

## 5. Caveats

- **Known units issue in the older `ingest.py`:** it stores Ausgrid kWh-per-half-hour values in columns named `solar_kw` / `load_kw` without multiplying by 2. Every power in `p2p_energy.db` is therefore half the true average kW. The new pipeline converts correctly. `ingest.py` was left unchanged.
- Sydney households on a UK-derived feeder: the pairing is ours. See `docs/dataset_decision.md` §6.
- 2010-11 PV systems are small (mean 1.57 kWp for the 55 sampled homes). Surplus exists in about 20% of house-intervals.
- Dead-end cable stubs with no load (205 segments, 245.8 m) are pruned from `edges.csv`. `edges_full.csv` keeps them.
- DST days are left as recorded (Ausgrid gives 48 columns every day).
