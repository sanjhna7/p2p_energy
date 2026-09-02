"""
Baseline battery model.

This is a deterministic, rule-based physics model — NOT a learned model.
It exists so the data layer can produce a complete state (solar, load,
battery SoC, grid voltage) at every timestep without depending on the
RL agent / GNN work, which comes later.

Control policy (baseline, simple greedy self-consumption):
    - net_kw > 0 (solar surplus)  -> charge the battery with the surplus,
      capped by charge rate limit and remaining capacity.
    - net_kw < 0 (load deficit)   -> discharge the battery to cover the
      deficit, capped by discharge rate limit and remaining energy.
    - Whatever surplus/deficit the battery *cannot* absorb is left over
      as grid import/export (returned as `residual_kw`) — this is the
      number that should eventually feed the power-flow / GNN layer.

This function is intentionally pure (no DB, no globals) so it can be
unit tested in isolation and swapped for a smarter policy (e.g. an RL
action) later without touching the rest of the data layer.
"""

from dataclasses import dataclass

DEFAULT_DT_HOURS = 0.5          # matches the 30-min Ausgrid interval
DEFAULT_CHARGE_EFFICIENCY = 0.95
DEFAULT_DISCHARGE_EFFICIENCY = 0.95


@dataclass
class BatteryResult:
    soc_kwh: float
    soc_pct: float
    charge_kw: float     # +ve = charging, -ve = discharging, 0 = idle
    residual_kw: float   # leftover surplus/deficit after battery action


def step_battery(
    prev_soc_kwh: float,
    capacity_kwh: float,
    net_kw: float,
    max_charge_kw: float,
    max_discharge_kw: float,
    dt_hours: float = DEFAULT_DT_HOURS,
    charge_eff: float = DEFAULT_CHARGE_EFFICIENCY,
    discharge_eff: float = DEFAULT_DISCHARGE_EFFICIENCY,
) -> BatteryResult:
    """
    Advance one house's battery by a single timestep.

    prev_soc_kwh    : SoC carried over from the previous step
    capacity_kwh    : usable battery capacity for this house
    net_kw          : solar_kw - load_kw for this step (from meter_readings)
    max_charge_kw   : house battery charge rate limit
    max_discharge_kw: house battery discharge rate limit
    """

    if capacity_kwh <= 0:
        # No battery installed at this house -> pass everything straight
        # through as residual (this house has no storage to buffer with).
        return BatteryResult(0.0, 0.0, 0.0, net_kw)

    if net_kw >= 0:
        # Surplus: try to charge.
        charge_kw = min(net_kw, max_charge_kw)
        headroom_kwh = capacity_kwh - prev_soc_kwh
        max_chargeable_kw = headroom_kwh / dt_hours / charge_eff if dt_hours > 0 else 0
        charge_kw = max(0.0, min(charge_kw, max_chargeable_kw))

        new_soc_kwh = prev_soc_kwh + charge_kw * charge_eff * dt_hours
        residual_kw = net_kw - charge_kw

        return BatteryResult(
            soc_kwh=round(new_soc_kwh, 4),
            soc_pct=round(100 * new_soc_kwh / capacity_kwh, 2),
            charge_kw=round(charge_kw, 4),
            residual_kw=round(residual_kw, 4),
        )

    else:
        # Deficit: try to discharge.
        deficit_kw = -net_kw
        discharge_kw = min(deficit_kw, max_discharge_kw)
        available_kwh = prev_soc_kwh
        max_dischargeable_kw = (available_kwh * discharge_eff) / dt_hours if dt_hours > 0 else 0
        discharge_kw = max(0.0, min(discharge_kw, max_dischargeable_kw))

        new_soc_kwh = prev_soc_kwh - (discharge_kw / discharge_eff) * dt_hours
        residual_kw = net_kw + discharge_kw  # still negative if battery couldn't cover it all

        return BatteryResult(
            soc_kwh=round(new_soc_kwh, 4),
            soc_pct=round(100 * new_soc_kwh / capacity_kwh, 2),
            charge_kw=round(-discharge_kw, 4),
            residual_kw=round(residual_kw, 4),
        )
