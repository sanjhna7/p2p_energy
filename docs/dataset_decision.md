# Dataset decision

What the simulation is built on, why, and exactly what is ours rather
than the sources'. Reference numbers [R#] are in `REFERENCES.md`.

## The question the data has to answer

Each house is an agent deciding, every 30 minutes, how to use its net
energy: consume locally, store, share with neighbours, or trade with the
grid. That needs, per house and per interval, **real household
consumption and real rooftop PV generation**, and a **real low-voltage
network** connecting the houses.

No public dataset gives measured data for a multi-house residential DC
microgrid, so the two parts come from two sources and are joined by us.

## Household data: Ausgrid Solar Home [R1, R2]

- 300 homes in the Ausgrid network area (Sydney region, NSW), each with a
  gross-metered rooftop PV system, 1 Jul 2010 – 30 Jun 2013.
- Half-hourly kWh, with general consumption (GC), controlled load (CL) and
  gross PV generation (GG) metered separately, so load and generation are
  each known on their own rather than only as a net figure.
- Ausgrid states the 2010–11 year is a full set of actual meter readings.
- Ausgrid publishes summary statistics "as a basis for comparison and
  checking". The pipeline reproduces the 2010–11 means exactly:
  6,980 kWh consumption, 2,119 kWh generation, 1.68 kWp, 1,253 kWh/kWp
  (`tests/test_data_facts.py`).

What it does **not** contain: voltage, current, battery state, grid
import/export with a battery, or the location of homes on a network.
These are never filled in by the data layer.

The values are AC-side billing-meter energy. We use them as the energy a
household needs and produces in each interval. They are **not** DC
measurements, and any DC quantity derived from them is a model output.

## Network: IEEE European LV Test Feeder [R3, R4]

- A published benchmark: radial, 50 Hz, 416 V, fed by an 11/0.416 kV
  0.8 MVA transformer, with line codes giving sequence impedances.
- The published CSV files contain 906 buses, 905 line segments and
  55 single-phase residential loads. Fifty-five individual customer
  connections map one-to-one onto 55 agents, which aggregated MV feeders
  (where one bus is many houses) do not.
- Cables are strongly resistive (median R1/X1 ≈ 13), which suits a later
  DC abstraction, but it is an AC feeder and is kept as published here.

### Why this feeder

1. **One load is one house.** The widely used IEEE feeders (13, 34, 123
   bus) are North American medium-voltage feeders whose loads each stand
   for many customers. This feeder models 55 individual residential
   connections, which is what one-agent-per-house needs.
2. **It matches Australian low-voltage practice.** Australian supply is
   nominally 230/400 V [R7] at 50 Hz, the same as this 416 V, 50 Hz
   European-style feeder, and in both a single distribution transformer
   feeds many customers over a radial LV cable network. North American
   LV practice (120/240 V split-phase, 60 Hz) does not match the Ausgrid
   homes.
3. **Everything needed is published:** cable lengths and impedances,
   each customer's bus and phase, bus coordinates, and the transformer.
4. **Official reference results** (OpenDSS and GridLAB-D, in
   `Solutions/`) allow a power-flow model to be validated against the
   benchmark before it is adapted.
5. **Recognised and open.** It is documented in [R4] and free to use,
   so results are reproducible and comparable.

### Its limits

- It is not a Sydney network. No public Australian LV feeder with
  per-house connections is used as a standard benchmark; the pairing is
  stated as our adaptation.
- It is AC three-phase. We keep the topology and cable resistance and
  ignore phase allocation; any DC quantity is a model output.
- It is one feeder. Claims about unseen topologies need further networks
  (e.g. ENWL LV or SimBench, as listed in the Phase 1 dataset review).

## Our adaptations

None of these are in either source. Each must be stated wherever the
data is used.

1. **Unit conversion.** kWh per half hour × 2 = average kW over the interval.
2. **Cleaning rules** (`data_pipeline/ausgrid.py: CleaningRules`).
   A customer is excluded for: a missing GC or GG series, incomplete days,
   any whole day of zero PV generation, or any PV generation above
   0.01 kWh per half hour between 00:00 and 04:00. For 2010–11 this
   leaves 269 of 300 customers. These rules are ours, not Ausgrid's.
3. **Pairing.** 55 clean customers are sampled with a fixed seed and
   assigned to feeder loads LOAD1..LOAD55. A Sydney household on a
   European benchmark feeder position is our construction.
4. **Graph reduction.** Cable stubs with no load are pruned and chains of
   pass-through buses are merged, summing length, R and X (exact for
   series impedances). Result: 110 nodes (1 transformer, 55 houses,
   54 junctions) and 109 edges. `edges_full.csv` keeps all 905 segments.
5. **Scenario selection** (`p2p/scenario.py`). An experiment uses the
   n houses of the smallest feeder branch holding n, plus their path to
   the transformer, over a chosen date window.
6. **Equipment.** Batteries are not in the data. Their parameters come
   from a manufacturer datasheet [R5, R6], and the efficiency split
   between charge and discharge is ours (`p2p/battery.py`).

## Statement for reports

> Household load and rooftop PV profiles are real half-hourly measurements
> from the Ausgrid Solar Home Electricity Data [R1, R2]. Households are
> placed on the 55 customer connections of the IEEE European LV Test
> Feeder [R3, R4], chosen because it models individual residential
> connections and matches Australian 230/400 V, 50 Hz low-voltage
> practice [R7]; this pairing, the cleaning rules and the network
> reduction are our adaptations. Battery parameters are taken from the
> BYD Battery-Box Premium LVS datasheet [R5, R6]. No voltage, current or
> battery value is measured; all such quantities are model outputs.
