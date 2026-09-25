"""
Provisional electrical model of one household DC nanogrid.

                      PV
                       |
                  Local DC Bus
                 /            \
           Battery           DC Load
                 \            /
             Bus Tie Converter
                       |
              Community DC Bus

NONE of the voltages, currents or battery quantities produced here come
from the Ausgrid dataset - that dataset contains energy meter readings
only. They are a deliberately simple, physically consistent
approximation, standing in for the Simulink/hardware model until it is
finished. Every constant they depend on comes from
config/nanogrid_config.yaml.

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

from dataclasses import dataclass, asdict

from . import battery_model
from .config import BatteryConfig, BusConfig, NanogridConfig, TieConfig

W_PER_KW = 1000.0


def current_from_power(power_kw: float, voltage_v: float) -> float:
    """P = V x I, with kW -> W handled in one place.

    Callers pass a positive nominal voltage (config._check enforces it),
    so this never divides by zero.
    """
    return W_PER_KW * power_kw / voltage_v


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

    # Slack terms. Non-zero only when the bus-tie converter rating binds:
    # surplus that cannot be exported is curtailed, deficit that cannot
    # be imported goes unserved. Recorded rather than silently dropped,
    # because the household energy balance in validate.py needs them.
    curtailed_power: float
    unserved_power: float

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------
# PV
# ---------------------------------------------------------------------

def pv_power_from_ausgrid(
    ausgrid_pv_kw: float,
    ausgrid_capacity_kwp: float,
    hardware_capacity_kw: float,
) -> float:
    """Re-scale a REAL Ausgrid generation profile to our PV array.

        pv_pu     = ausgrid_pv_kw / ausgrid_capacity_kwp
        pv_power  = pv_pu * hardware_capacity_kw

    Normalising by the customer's own installed capacity is what makes
    this a re-scaling rather than a re-invention: the per-unit series is
    that household's real, weather-driven, seasonal generation shape, and
    only its magnitude changes. No generic clear-sky/solar-geometry
    equation is used anywhere.
    """

    if ausgrid_capacity_kwp <= 0:
        raise ValueError(
            "customer has no Generator Capacity; cannot normalise its PV profile"
        )

    pv_pu = ausgrid_pv_kw / ausgrid_capacity_kwp
    # Ausgrid's gross generation channel is non-negative by definition;
    # clamp defensively so a stray negative can never become "negative
    # sunlight" downstream.
    return max(0.0, pv_pu * hardware_capacity_kw)


def load_power_from_ausgrid(ausgrid_load_kw: float, dc_load_fraction: float) -> float:
    """Map the REAL household demand (GC + CL) onto the DC load."""
    return max(0.0, ausgrid_load_kw * dc_load_fraction)


# ---------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------

@dataclass
class BatteryStep:
    soc: float                 # fraction of nameplate capacity, in [min_soc, max_soc]
    power_kw: float            # > 0 discharging, < 0 charging
    charge_kw: float           # magnitude, >= 0
    discharge_kw: float        # magnitude, >= 0
    residual_kw: float         # surplus (+) / deficit (-) left for the tie


def step_battery(prev_soc: float, net_power_kw: float, cfg: BatteryConfig,
                 dt_hours: float) -> BatteryStep:
    """Advance the battery one timestep under greedy self-consumption.

    Surplus charges the battery first and only the remainder is exported;
    deficit discharges the battery first and only the remainder is
    imported. That coupling is the point: SoC, tie power and bus voltage
    are all consequences of the same real PV/load series, never
    independent random columns.

    Implemented on top of the shared battery_model.step_battery
    so there is one set of charge/discharge equations in the repository.
    That function cycles between 0 and its capacity argument, so it is
    given the USABLE window (max_soc - min_soc) x capacity and an offset
    state of charge; the equations it applies,

        charging:    E(t+1) = E(t) + P_chg * eff * dt
        discharging: E(t+1) = E(t) - P_dis / eff * dt

    are exactly the SoC update this provisional model specifies, so the
    result is bounded by min_soc and max_soc by construction.
    """

    usable_kwh = cfg.usable_capacity_kwh

    if usable_kwh <= 0 or cfg.capacity_kwh <= 0:
        return BatteryStep(prev_soc, 0.0, 0.0, 0.0, net_power_kw)

    # SoC fraction -> energy above the floor, which is what the shared
    # battery model understands.
    prev_usable_kwh = (prev_soc - cfg.min_soc) * cfg.capacity_kwh
    prev_usable_kwh = min(max(prev_usable_kwh, 0.0), usable_kwh)

    result = battery_model.step_battery(
        prev_soc_kwh=prev_usable_kwh,
        capacity_kwh=usable_kwh,
        net_kw=net_power_kw,
        max_charge_kw=cfg.max_charge_kw,
        max_discharge_kw=cfg.max_discharge_kw,
        dt_hours=dt_hours,
        charge_eff=cfg.charge_efficiency,
        discharge_eff=cfg.discharge_efficiency,
    )

    # battery_model rounds its power, its SoC and its residual to 4 dp
    # INDEPENDENTLY, so those three no longer satisfy the energy balance
    # exactly (they disagree at ~1e-4 kW). Take its power decision as the
    # single source of truth and re-derive the SoC and the residual from
    # it, so the row balances to machine precision. The rounding itself
    # is kept - 0.1 W is a sane converter resolution - it just has to be
    # applied once rather than three times.
    #
    # battery_model reports +ve = charging; this package reports
    # +ve = discharging (section 10). The flip happens below.
    power_kw = result.charge_kw       # +ve charging, at this point only

    if power_kw > 0:
        # Cap by the energy the window can still absorb, so the rounding
        # can never push SoC past max_soc.
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
        # Exact by construction: whatever the battery did not take is
        # what the bus tie must carry.
        residual_kw=net_power_kw - charge_kw + discharge_kw,
    )


# ---------------------------------------------------------------------
# Bus tie
# ---------------------------------------------------------------------

@dataclass
class TieStep:
    power_kw: float        # signed: > 0 export, < 0 import
    curtailed_kw: float    # surplus the converter could not export
    unserved_kw: float     # deficit the converter could not import


def step_tie(residual_kw: float, cfg: TieConfig) -> TieStep:
    """Push whatever the battery could not absorb through the bus tie,
    limited by the converter rating.

    Whatever exceeds the rating is not quietly discarded: surplus is
    reported as curtailment and deficit as unserved load, so the
    household energy balance still closes exactly.
    """

    limit = cfg.max_power_kw
    power = min(max(residual_kw, -limit), limit)

    spill = residual_kw - power
    return TieStep(
        power_kw=power,
        # Written as explicit branches rather than max(): max(-0.0, 0.0)
        # returns -0.0, which would litter the CSV with negative zeros.
        curtailed_kw=spill if spill > 0 else 0.0,
        unserved_kw=-spill if spill < 0 else 0.0,
    )


# ---------------------------------------------------------------------
# Local DC bus
# ---------------------------------------------------------------------

def local_bus_voltage(tie_power_kw: float, cfg: BusConfig) -> float:
    """Provisional droop approximation of the local DC bus voltage.

        V_local = V_nominal - R_virtual * P_imported

    where P_imported = -tie_power is the power the local bus draws from
    the community through the tie. Importing loads the bus and pulls its
    voltage down; exporting pushes it up. Bounded by voltage_min /
    voltage_max, which validate.py re-checks and reports on.

    This stands in for the DC power-flow solution of the Simulink model.
    """

    imported_kw = -tie_power_kw
    voltage = cfg.local_nominal_voltage - cfg.local_virtual_droop * imported_kw
    return min(max(voltage, cfg.voltage_min), cfg.voltage_max)


def local_bus_current(pv_power_kw, battery_discharge_kw, imported_kw, voltage_v):
    """Total power injected into the local DC bus, expressed as current.

    Injection (PV + battery discharge + import from the community) equals
    withdrawal (load + battery charging + export) by construction, so
    either side gives the same figure; the injection side is used.
    Reported as a magnitude, like the bus current a meter would read.
    """
    injected_kw = pv_power_kw + battery_discharge_kw + max(imported_kw, 0.0)
    return abs(current_from_power(injected_kw, voltage_v))


# ---------------------------------------------------------------------
# Community DC bus
# ---------------------------------------------------------------------

@dataclass
class CommunityStep:
    voltage: float
    total_export_kw: float      # sum of positive tie powers
    total_import_kw: float      # sum of negative tie powers, as magnitude
    network_loss_kw: float      # loss on the exported energy
    slack_kw: float             # external grid top-up (+) / absorption (-)
    aggregate_imbalance_kw: float


def community_step(tie_powers, bus_cfg: BusConfig, loss_fraction: float) -> CommunityStep:
    """One shared community DC bus voltage for every house at this step.

        V_community = V_nominal - droop * aggregate_imbalance
        aggregate_imbalance = total_import - total_export   (net demand)

    Net community demand pulls the shared bus down; net community surplus
    pushes it up. One value per timestep, so every connected house
    observes the SAME community voltage - it is a shared bus, not a
    per-house random number.

    The community rarely balances exactly on real data, so the residual
    is reported explicitly as `slack_kw` (the external grid connection)
    rather than being hidden:

        delivered = total_export * (1 - loss_fraction)
        slack     = total_import - delivered
        slack > 0 : community imports from the external grid
        slack < 0 : community exports to the external grid
    """

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


# ---------------------------------------------------------------------
# One house, one timestep
# ---------------------------------------------------------------------

def step_house(
    ausgrid_pv_kw: float,
    ausgrid_load_kw: float,
    ausgrid_capacity_kwp: float,
    prev_soc: float,
    cfg: NanogridConfig,
    dt_hours: float,
) -> HouseState:
    """Assemble one house's complete provisional electrical state.

    Order matters and encodes the energy flow: real PV and load first,
    then the battery reacts to their difference, then the bus tie carries
    whatever is left, and only then do the voltages and currents follow
    from those powers. Nothing here is drawn at random.
    """

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

        # Magnitude only; the direction is carried by tie_power's sign.
        tie_current=abs(current_from_power(tie.power_kw, v_local)),
        tie_power=tie.power_kw,

        net_power=net_power,
        surplus_power=max(net_power, 0.0),
        deficit_power=max(-net_power, 0.0),
        battery_charge_power=battery.charge_kw,
        battery_discharge_power=battery.discharge_kw,
        # Energy traded with the community over this interval, signed like
        # tie_power: + exported kWh, - imported kWh.
        energy_exchange=tie.power_kw * dt_hours,

        curtailed_power=tie.curtailed_kw,
        unserved_power=tie.unserved_kw,
    )
