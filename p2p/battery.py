"""Battery model: energy bookkeeping for one DC battery per house.

Physics only, no control policy. The caller (a baseline rule or a MARL
agent) requests a power and the battery delivers what its limits allow.
Every function works on scalars or on numpy arrays with one entry per
house.

SIGN CONVENTION
    power > 0  discharging (battery -> house bus)
    power < 0  charging    (house bus -> battery)

EFFICIENCY
    The datasheet gives a round-trip figure only. It is split equally
    between the two directions, eta_c = eta_d = sqrt(eta_rt). This split
    is our modelling step (DERIVED), not a datasheet value.
"""

import math
from dataclasses import dataclass

import numpy as np

from p2p.config import BatteryConfig


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float
    max_power_kw: float
    round_trip_efficiency: float

    @classmethod
    def from_config(cls, cfg: BatteryConfig) -> "BatterySpec":
        return cls(
            capacity_kwh=cfg.usable_energy_kwh.value,
            max_power_kw=cfg.rated_dc_power_kw.value,
            round_trip_efficiency=cfg.round_trip_efficiency.value,
        )

    @property
    def one_way_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)


@dataclass(frozen=True)
class BatteryStep:
    power_kw: np.ndarray   # delivered at the terminals, signed as above
    soc: np.ndarray        # after the step, fraction of usable energy


def step(spec: BatterySpec, soc, requested_kw, dt_hours: float) -> BatteryStep:
    """Apply a requested power for one interval, limited by the power
    rating and by the energy left (discharge) or the headroom (charge)."""

    eta = spec.one_way_efficiency
    energy = np.asarray(soc, dtype=float) * spec.capacity_kwh
    power = np.clip(np.asarray(requested_kw, dtype=float), -spec.max_power_kw, spec.max_power_kw)

    # Discharging P for dt draws P*dt/eta from storage; charging stores |P|*dt*eta.
    max_discharge = energy * eta / dt_hours
    max_charge = (spec.capacity_kwh - energy) / (eta * dt_hours)
    power = np.minimum(power, max_discharge)
    power = np.maximum(power, -max_charge)

    energy = energy - np.where(power > 0, power / eta, power * eta) * dt_hours
    new_soc = np.clip(energy / spec.capacity_kwh, 0.0, 1.0)
    return BatteryStep(power_kw=power, soc=new_soc)
