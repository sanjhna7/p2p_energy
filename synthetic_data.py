"""
Synthetic DC Microgrid Dataset Generator
=========================================

Generates a self-contained synthetic dataset for a DC-coupled P2P energy
trading microgrid. Unlike the Ausgrid-based pipeline (which has real
household load/solar but NO battery specs and NO network topology, so
grid_model.py has to fall back on placeholder constants), everything
here is generated from first principles so the electrical picture -
power, current, voltage - stays internally consistent from house to
house and timestep to timestep.

Columns produced:
    timestamp, house_id, household_load_kw, solar_kw,
    battery_power_kw, battery_soc_pct,
    line_current_a, line_voltage_v,
    grid_current_a, grid_voltage_v

Physics used (all DC):
    - Ohm's law:        V = V_nominal + I * R      (voltage rise/drop across a line)
    - DC power law:      P (kW) = V (V) * I (A) / 1000
    - Battery SoC:       integrated from charge/discharge power over dt,
                          same greedy self-consumption policy as battery_model.py

Two voltage points are modelled:
    - line_voltage_v  : the DC rail AT EACH HOUSE, after the IR drop/rise
                         across that house's own feeder resistance.
    - grid_voltage_v  : the shared DC bus (point of common coupling) that
                         all houses connect to - driven by the SUM of all
                         houses' currents at that timestep, through a
                         separate upstream resistance.

Run directly to write a CSV to disk:

    python generate_synthetic_dc_dataset.py

Tune the constants under CONFIG to change house count, duration,
voltage levels, or noise. TARGET_ROWS / NUM_HOUSES is chosen so the
default settings produce exactly 10,000 rows.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG

RANDOM_SEED = 42

NUM_HOUSES = 20                      # houses on the DC microgrid
INTERVAL_MINUTES = 30                # sampling interval (matches Ausgrid's 30-min cadence)
START_DATE = "2024-01-01 00:00:00"
TARGET_ROWS = 10_000                 # NUM_TIMESTEPS is derived to hit this exactly
NUM_TIMESTEPS = TARGET_ROWS // NUM_HOUSES

DC_NOMINAL_VOLTAGE = 48.0            # V - common low-voltage DC microgrid bus standard
LINE_RESISTANCE_OHM = (0.03, 0.08)   # per-house feeder resistance range (Ohm), randomised per house
GRID_RESISTANCE_OHM = 0.015          # resistance from the shared bus to the upstream grid connection

LOAD_BASE_KW = (0.15, 0.6)           # per-house baseline load range
LOAD_PEAK_MULTIPLIER = 2.2           # morning/evening peak multiplier over baseline
LOAD_NOISE_STD_KW = 0.05

BATTERY_CAPACITY_KWH = (4.0, 12.0)
BATTERY_MAX_POWER_KW = (1.0, 3.0)
BATTERY_INITIAL_SOC_PCT = (20.0, 80.0)
BATTERY_EFFICIENCY = 0.95

SOLAR_CAPACITY_KW = (0.5, 4.0)       # installed capacity for houses that have panels
SOLAR_PENETRATION = 0.6              # fraction of houses with solar installed
SOLAR_NOISE_STD = 0.08               # fractional cloud-cover noise

OUTPUT_FILE = "synthetic_dc_microgrid_data.csv"

# ------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)


def diurnal_solar_fraction(hour: float) -> float:
    """Bell-shaped generation fraction, 0 outside 6:00-18:00, peaking at noon."""
    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, np.sin(np.pi * (hour - 6) / 12)) ** 1.4


def diurnal_load_fraction(hour: float) -> float:
    """Two peaks (morning ~8:00, evening ~19:00), lower overnight baseline."""
    morning = np.exp(-((hour - 8) ** 2) / (2 * 2.0 ** 2))
    evening = np.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2))
    base = 0.35
    return base + 0.65 * max(morning, evening)


def build_house_params(num_houses: int) -> pd.DataFrame:
    has_solar = rng.random(num_houses) < SOLAR_PENETRATION
    solar_capacity = np.where(
        has_solar,
        rng.uniform(*SOLAR_CAPACITY_KW, num_houses),
        0.0,
    )
    return pd.DataFrame({
        "house_id": [f"H{100 + i}" for i in range(num_houses)],
        "load_base_kw": rng.uniform(*LOAD_BASE_KW, num_houses),
        "solar_capacity_kw": solar_capacity,
        "battery_capacity_kwh": rng.uniform(*BATTERY_CAPACITY_KWH, num_houses),
        "battery_max_power_kw": rng.uniform(*BATTERY_MAX_POWER_KW, num_houses),
        "line_resistance_ohm": rng.uniform(*LINE_RESISTANCE_OHM, num_houses),
        "soc_pct": rng.uniform(*BATTERY_INITIAL_SOC_PCT, num_houses),
    })


def generate_dataset() -> pd.DataFrame:
    houses = build_house_params(NUM_HOUSES)
    timestamps = pd.date_range(START_DATE, periods=NUM_TIMESTEPS, freq=f"{INTERVAL_MINUTES}min")
    dt_hours = INTERVAL_MINUTES / 60.0

    all_rows = []

    for ts in timestamps:
        hour = ts.hour + ts.minute / 60.0
        solar_frac = diurnal_solar_fraction(hour)
        load_frac = diurnal_load_fraction(hour)

        step_currents = []
        step_rows = []

        for idx, h in houses.iterrows():
            # --- Household load (kW) ---
            load_kw = max(
                0.0,
                h.load_base_kw * load_frac * LOAD_PEAK_MULTIPLIER
                + rng.normal(0, LOAD_NOISE_STD_KW),
            )

            # --- Local solar generation (kW), 0 if no panels installed ---
            if h.solar_capacity_kw > 0:
                solar_kw = max(
                    0.0,
                    h.solar_capacity_kw * solar_frac * (1 + rng.normal(0, SOLAR_NOISE_STD)),
                )
            else:
                solar_kw = 0.0

            net_kw = solar_kw - load_kw   # +ve = surplus, -ve = deficit

            # --- Battery: same greedy self-consumption policy as battery_model.py ---
            soc_kwh = h.soc_pct / 100.0 * h.battery_capacity_kwh

            if net_kw >= 0:
                charge_kw = min(net_kw, h.battery_max_power_kw)
                headroom_kwh = h.battery_capacity_kwh - soc_kwh
                max_chargeable_kw = headroom_kwh / dt_hours / BATTERY_EFFICIENCY
                charge_kw = max(0.0, min(charge_kw, max_chargeable_kw))
                soc_kwh += charge_kw * BATTERY_EFFICIENCY * dt_hours
                battery_power_kw = charge_kw          # +ve = charging
                residual_kw = net_kw - charge_kw
            else:
                deficit_kw = -net_kw
                discharge_kw = min(deficit_kw, h.battery_max_power_kw)
                max_dischargeable_kw = (soc_kwh * BATTERY_EFFICIENCY) / dt_hours
                discharge_kw = max(0.0, min(discharge_kw, max_dischargeable_kw))
                soc_kwh -= (discharge_kw / BATTERY_EFFICIENCY) * dt_hours
                battery_power_kw = -discharge_kw      # -ve = discharging
                residual_kw = net_kw + discharge_kw

            soc_pct = 100.0 * soc_kwh / h.battery_capacity_kwh
            houses.at[idx, "soc_pct"] = soc_pct   # carry forward to next timestep

            # --- DC electrical quantities ---
            # residual_kw > 0 -> house exports current onto the shared bus
            #                    (its own rail sits ABOVE nominal to push current out)
            # residual_kw < 0 -> house draws current from the bus
            #                    (its own rail sits BELOW nominal)
            line_current_a = (residual_kw * 1000.0) / DC_NOMINAL_VOLTAGE
            line_voltage_v = DC_NOMINAL_VOLTAGE + line_current_a * h.line_resistance_ohm

            step_currents.append(line_current_a)
            step_rows.append({
                "timestamp": ts,
                "house_id": h.house_id,
                "household_load_kw": round(load_kw, 4),
                "solar_kw": round(solar_kw, 4),
                "battery_power_kw": round(battery_power_kw, 4),
                "battery_soc_pct": round(soc_pct, 2),
                "line_current_a": round(line_current_a, 3),
                "line_voltage_v": round(line_voltage_v, 3),
            })

        # --- Grid / PCC voltage: shared bus, driven by the SUM of all houses' currents ---
        total_current_a = sum(step_currents)
        grid_voltage_v = DC_NOMINAL_VOLTAGE + total_current_a * GRID_RESISTANCE_OHM

        for r in step_rows:
            r["grid_current_a"] = round(total_current_a, 3)
            r["grid_voltage_v"] = round(grid_voltage_v, 3)

        all_rows.extend(step_rows)

    return pd.DataFrame(all_rows)


def main():
    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(df)} rows across {NUM_HOUSES} houses "
          f"and {NUM_TIMESTEPS} timesteps ({INTERVAL_MINUTES}-min interval).")
    print(f"Saved to {OUTPUT_FILE}")
    print()
    print(df.head(10).to_string(index=False))
    print()
    print("Summary stats:")
    print(df[[
        "household_load_kw", "battery_power_kw", "battery_soc_pct",
        "line_voltage_v", "grid_voltage_v",
    ]].describe().round(3))


if __name__ == "__main__":
    main()