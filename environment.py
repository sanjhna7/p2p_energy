"""
Energy environment: physics models, state, observations, and topology.

This module consolidates the DC nanogrid electrical model, battery
physics, MARL observation extraction, and community topology into a
single environment definition.

SIGN CONVENTIONS (used everywhere, without exception)
-----------------------------------------------------
    net_power     = pv_power - load_power
    battery_power > 0  battery is DISCHARGING (supplying the local bus)
    battery_power < 0  battery is CHARGING (absorbing from the local bus)
    tie_power     > 0  house -> community (EXPORT)
    tie_power     < 0  community -> house (IMPORT)
    tie_current        MAGNITUDE only; the direction lives in tie_power

UNITS
-----
Power in kW, energy in kWh, voltage in V, current in A. Currents
therefore carry a factor of 1000:  I(A) = 1000 * P(kW) / V(V).
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

from config import BatteryConfig, BusConfig, NanogridConfig, TieConfig

W_PER_KW = 1000.0


# =====================================================================
# 1. Battery physics (low-level core)
# =====================================================================

DEFAULT_DT_HOURS = 0.5          # matches the 30-min Ausgrid interval
DEFAULT_CHARGE_EFFICIENCY = 0.95
DEFAULT_DISCHARGE_EFFICIENCY = 0.95


@dataclass
class BatteryResult:
    """Result from the low-level battery step equations."""
    soc_kwh: float
    soc_pct: float
    charge_kw: float     # +ve = charging, -ve = discharging, 0 = idle
    residual_kw: float   # leftover surplus/deficit after battery action


def step_battery_core(
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
    Advance one house's battery by a single timestep (low-level).

    This is a deterministic, rule-based physics model — NOT a learned
    model. Control policy: simple greedy self-consumption.

    prev_soc_kwh    : SoC carried over from the previous step
    capacity_kwh    : usable battery capacity for this house
    net_kw          : solar_kw - load_kw for this step
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


# =====================================================================
# 2. DC nanogrid physics models
# =====================================================================

def current_from_power(power_kw: float, voltage_v: float) -> float:
    """P = V x I, with kW -> W handled in one place."""
    return W_PER_KW * power_kw / voltage_v


# ----- House state -----

@dataclass
class HouseState:
    """The complete provisional state of one house at one timestep."""

    pv_voltage: float
    pv_current: float
    pv_power: float

    load_voltage: float
    load_current: float
    load_power: float

    battery_voltage: float
    battery_current: float
    battery_power: float
    battery_soc: float

    local_bus_voltage: float
    local_bus_current: float

    tie_current: float
    tie_power: float

    net_power: float
    surplus_power: float
    deficit_power: float
    battery_charge_power: float
    battery_discharge_power: float
    energy_exchange: float

    # Slack terms. Non-zero only when the bus-tie converter rating binds.
    curtailed_power: float
    unserved_power: float

    def as_dict(self) -> dict:
        return asdict(self)


# ----- PV -----

def pv_power_from_ausgrid(
    ausgrid_pv_kw: float,
    ausgrid_capacity_kwp: float,
    hardware_capacity_kw: float,
) -> float:
    """Re-scale a REAL Ausgrid generation profile to our PV array.

        pv_pu     = ausgrid_pv_kw / ausgrid_capacity_kwp
        pv_power  = pv_pu * hardware_capacity_kw
    """

    if ausgrid_capacity_kwp <= 0:
        raise ValueError(
            "customer has no Generator Capacity; cannot normalise its PV profile"
        )

    pv_pu = ausgrid_pv_kw / ausgrid_capacity_kwp
    return max(0.0, pv_pu * hardware_capacity_kw)


def load_power_from_ausgrid(ausgrid_load_kw: float, dc_load_fraction: float) -> float:
    """Map the REAL household demand (GC + CL) onto the DC load."""
    return max(0.0, ausgrid_load_kw * dc_load_fraction)


# ----- Battery (high-level, config-aware wrapper) -----

@dataclass
class BatteryStep:
    """High-level battery step result with SoC as fraction."""
    soc: float                 # fraction of nameplate capacity, in [min_soc, max_soc]
    power_kw: float            # > 0 discharging, < 0 charging
    charge_kw: float           # magnitude, >= 0
    discharge_kw: float        # magnitude, >= 0
    residual_kw: float         # surplus (+) / deficit (-) left for the tie


def step_battery(prev_soc: float, net_power_kw: float, cfg: BatteryConfig,
                 dt_hours: float) -> BatteryStep:
    """Advance the battery one timestep under greedy self-consumption.

    Implemented on top of step_battery_core so there is one set of
    charge/discharge equations. That function cycles between 0 and its
    capacity argument, so it is given the USABLE window
    (max_soc - min_soc) x capacity and an offset state of charge.
    """

    usable_kwh = cfg.usable_capacity_kwh

    if usable_kwh <= 0 or cfg.capacity_kwh <= 0:
        return BatteryStep(prev_soc, 0.0, 0.0, 0.0, net_power_kw)

    # SoC fraction -> energy above the floor.
    prev_usable_kwh = (prev_soc - cfg.min_soc) * cfg.capacity_kwh
    prev_usable_kwh = min(max(prev_usable_kwh, 0.0), usable_kwh)

    result = step_battery_core(
        prev_soc_kwh=prev_usable_kwh,
        capacity_kwh=usable_kwh,
        net_kw=net_power_kw,
        max_charge_kw=cfg.max_charge_kw,
        max_discharge_kw=cfg.max_discharge_kw,
        dt_hours=dt_hours,
        charge_eff=cfg.charge_efficiency,
        discharge_eff=cfg.discharge_efficiency,
    )

    # Take the core's power decision as the single source of truth and
    # re-derive the SoC and residual from it, so the row balances to
    # machine precision.
    power_kw = result.charge_kw       # +ve charging, at this point only

    if power_kw > 0:
        headroom_kwh = usable_kwh - prev_usable_kwh
        power_kw = min(power_kw, headroom_kwh / (cfg.charge_efficiency * dt_hours))
        power_kw = max(power_kw, 0.0)
        new_usable_kwh = prev_usable_kwh + power_kw * cfg.charge_efficiency * dt_hours
        charge_kw, discharge_kw = power_kw, 0.0
    elif power_kw < 0:
        discharge_kw = min(
            -power_kw, prev_usable_kwh * cfg.discharge_efficiency / dt_hours
        )
        discharge_kw = max(discharge_kw, 0.0)
        new_usable_kwh = (
            prev_usable_kwh - discharge_kw / cfg.discharge_efficiency * dt_hours
        )
        charge_kw = 0.0
    else:
        new_usable_kwh = prev_usable_kwh
        charge_kw = discharge_kw = 0.0

    soc = cfg.min_soc + new_usable_kwh / cfg.capacity_kwh

    return BatteryStep(
        soc=soc,
        power_kw=discharge_kw - charge_kw,
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        residual_kw=net_power_kw - charge_kw + discharge_kw,
    )


# ----- Bus tie -----

@dataclass
class TieStep:
    power_kw: float        # signed: > 0 export, < 0 import
    curtailed_kw: float    # surplus the converter could not export
    unserved_kw: float     # deficit the converter could not import


def step_tie(residual_kw: float, cfg: TieConfig) -> TieStep:
    """Push whatever the battery could not absorb through the bus tie,
    limited by the converter rating."""

    limit = cfg.max_power_kw
    power = min(max(residual_kw, -limit), limit)

    spill = residual_kw - power
    return TieStep(
        power_kw=power,
        curtailed_kw=spill if spill > 0 else 0.0,
        unserved_kw=-spill if spill < 0 else 0.0,
    )


# ----- Local DC bus -----

def local_bus_voltage(tie_power_kw: float, cfg: BusConfig) -> float:
    """Provisional droop approximation of the local DC bus voltage.

        V_local = V_nominal - R_virtual * P_imported
    """
    imported_kw = -tie_power_kw
    voltage = cfg.local_nominal_voltage - cfg.local_virtual_droop * imported_kw
    return min(max(voltage, cfg.voltage_min), cfg.voltage_max)


def local_bus_current(pv_power_kw, battery_discharge_kw, imported_kw, voltage_v):
    """Total power injected into the local DC bus, expressed as current."""
    injected_kw = pv_power_kw + battery_discharge_kw + max(imported_kw, 0.0)
    return abs(current_from_power(injected_kw, voltage_v))


# ----- Community DC bus -----

@dataclass
class CommunityStep:
    voltage: float
    total_export_kw: float
    total_import_kw: float
    network_loss_kw: float
    slack_kw: float
    aggregate_imbalance_kw: float


def community_step(tie_powers, bus_cfg: BusConfig, loss_fraction: float) -> CommunityStep:
    """One shared community DC bus voltage for every house at this step."""

    total_export = sum(p for p in tie_powers if p > 0)
    total_import = sum(-p for p in tie_powers if p < 0)

    network_loss = total_export * loss_fraction
    delivered = total_export - network_loss
    slack = total_import - delivered

    imbalance = total_import - total_export
    voltage = bus_cfg.community_nominal_voltage - bus_cfg.community_droop * imbalance
    voltage = min(
        max(voltage, bus_cfg.community_voltage_min),
        bus_cfg.community_voltage_max,
    )

    return CommunityStep(
        voltage=voltage,
        total_export_kw=total_export,
        total_import_kw=total_import,
        network_loss_kw=network_loss,
        slack_kw=slack,
        aggregate_imbalance_kw=imbalance,
    )


# ----- One house, one timestep -----

def step_house(
    ausgrid_pv_kw: float,
    ausgrid_load_kw: float,
    ausgrid_capacity_kwp: float,
    prev_soc: float,
    cfg: NanogridConfig,
    dt_hours: float,
) -> HouseState:
    """Assemble one house's complete provisional electrical state."""

    pv_power = pv_power_from_ausgrid(
        ausgrid_pv_kw, ausgrid_capacity_kwp, cfg.pv.capacity_kw
    )
    load_power = load_power_from_ausgrid(ausgrid_load_kw, cfg.load.dc_load_fraction)

    net_power = pv_power - load_power

    battery = step_battery(prev_soc, net_power, cfg.battery, dt_hours)
    tie = step_tie(battery.residual_kw, cfg.tie)

    v_local = local_bus_voltage(tie.power_kw, cfg.bus)
    imported_kw = max(-tie.power_kw, 0.0)

    return HouseState(
        pv_voltage=cfg.pv.nominal_voltage,
        pv_current=current_from_power(pv_power, cfg.pv.nominal_voltage),
        pv_power=pv_power,

        load_voltage=cfg.load.nominal_voltage,
        load_current=current_from_power(load_power, cfg.load.nominal_voltage),
        load_power=load_power,

        battery_voltage=cfg.battery.nominal_voltage,
        battery_current=current_from_power(
            battery.power_kw, cfg.battery.nominal_voltage
        ),
        battery_power=battery.power_kw,
        battery_soc=battery.soc,

        local_bus_voltage=v_local,
        local_bus_current=local_bus_current(
            pv_power, battery.discharge_kw, imported_kw, v_local
        ),

        tie_current=abs(current_from_power(tie.power_kw, v_local)),
        tie_power=tie.power_kw,

        net_power=net_power,
        surplus_power=max(net_power, 0.0),
        deficit_power=max(-net_power, 0.0),
        battery_charge_power=battery.charge_kw,
        battery_discharge_power=battery.discharge_kw,
        energy_exchange=tie.power_kw * dt_hours,

        curtailed_power=tie.curtailed_kw,
        unserved_power=tie.unserved_kw,
    )


# =====================================================================
# 3. MARL observation extraction
# =====================================================================

# Columns that identify a row rather than describe it.
INDEX_COLUMNS = ["timestamp", "house_id"]


def validate_features(features, available):
    unknown = [f for f in features if f not in available]
    if unknown:
        raise ValueError(
            f"observation.features refers to column(s) not in the dataset: "
            f"{unknown}"
        )
    if not features:
        raise ValueError("observation.features is empty")
    return list(features)


def observation_frame(dataset, features):
    """Slice the observation columns (plus the index) out of the dataset."""
    features = validate_features(features, dataset.columns)
    return dataset[INDEX_COLUMNS + features].copy()


def observation_shape(features, n_houses):
    """(per-agent obs dim, joint obs dim) for the configured features."""
    return len(features), len(features) * n_houses


def observations_at(dataset, timestamp, features):
    """Joint observation at one timestep: (n_houses, n_features) array."""
    rows = dataset[dataset["timestamp"] == timestamp].sort_values("house_id")
    return np.asarray(rows[list(features)], dtype=np.float32)


# =====================================================================
# 4. Community topology
# =====================================================================

def build_edges(n_nodes: int, topology: str) -> List[Tuple[int, int]]:
    """Edges for the configured community layout.

    chain : House0 - House1 - ... - HouseN-1
    ring  : chain with the ends joined
    star  : House0 at the centre, every other house connected to it
    full  : every house connected to every other house
    """

    if n_nodes < 1:
        raise ValueError("a community needs at least one house")

    if topology == "chain":
        return [(i, i + 1) for i in range(n_nodes - 1)]

    if topology == "ring":
        if n_nodes < 3:
            return [(i, i + 1) for i in range(n_nodes - 1)]
        return [(i, (i + 1) % n_nodes) for i in range(n_nodes)]

    if topology == "star":
        return [(0, i) for i in range(1, n_nodes)]

    if topology == "full":
        return [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    raise ValueError(
        f"unknown topology {topology!r}; expected chain, ring, star or full"
    )


def build_topology(house_ids, customer_ids, topology: str) -> dict:
    nodes = list(range(len(house_ids)))
    return {
        "provenance": "provisional_simulation",
        "note": (
            "Preliminary community layout. NOT derived from Ausgrid - the "
            "source dataset has no network topology, and customer ids are "
            "not spatial. Replace with the Simulink/hardware layout."
        ),
        "topology": topology,
        "nodes": nodes,
        "edges": [list(e) for e in build_edges(len(nodes), topology)],
        "node_attributes": [
            {"node": n, "house_id": h, "ausgrid_customer_id": c}
            for n, h, c in zip(nodes, house_ids, customer_ids)
        ],
    }


def save_topology(topology: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(topology, fh, indent=2)


def load_topology(path) -> dict:
    with open(path) as fh:
        return json.load(fh)
