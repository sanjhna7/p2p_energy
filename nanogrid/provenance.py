"""
Per-variable data provenance, for research transparency.

Two labels, and the distinction must survive into any paper written on
this dataset:

    "ausgrid_derived"        the variable's temporal behaviour comes from
                             the real Ausgrid Solar Home Electricity
                             Dataset (2010-2011), with only a documented
                             hardware scaling or mapping applied.

    "provisional_simulation" the variable does not exist in the Ausgrid
                             dataset and is produced by the simplified
                             nanogrid model in models.py. It will be
                             REPLACED by the Simulink/hardware simulation
                             output.
"""

AUSGRID = "ausgrid_derived"
PROVISIONAL = "provisional_simulation"

RESEARCH_STATEMENT = (
    "The preliminary MARL dataset uses real temporal household PV "
    "generation and consumption profiles derived from the Ausgrid Solar "
    "Home Electricity Dataset. Electrical variables unavailable in the "
    "source dataset, including battery state-of-charge, DC-bus voltage, "
    "converter/tie power and current, are generated using a simplified "
    "physically consistent nanogrid model for algorithm-development "
    "purposes. These provisional variables will subsequently be replaced "
    "by measurements from the completed Simulink/hardware model."
)

PROVENANCE = {
    "timestamp": (AUSGRID, "Original Ausgrid half-hourly interval timestamp."),
    "house_id": (PROVISIONAL, "Index of the house/agent in this simulation."),
    "ausgrid_customer_id": (
        AUSGRID,
        "Real Ausgrid customer whose profile backs this house.",
    ),

    "pv_power": (
        AUSGRID,
        "Ausgrid GG (gross generation), kWh/interval -> kW, normalised by "
        "the customer's Generator Capacity and re-scaled to pv.capacity_kw.",
    ),
    "pv_voltage": (PROVISIONAL, "Assumed nominal PV operating voltage."),
    "pv_current": (PROVISIONAL, "pv_power / pv_voltage."),

    "load_power": (
        AUSGRID,
        "Ausgrid GC + CL, kWh/interval -> kW, scaled by load.dc_load_fraction.",
    ),
    "load_voltage": (PROVISIONAL, "Assumed nominal DC load voltage."),
    "load_current": (PROVISIONAL, "load_power / load_voltage."),

    "battery_power": (
        PROVISIONAL,
        "Greedy self-consumption battery. > 0 discharging, < 0 charging.",
    ),
    "battery_soc": (PROVISIONAL, "Modelled state of charge, fraction of capacity."),
    "battery_voltage": (PROVISIONAL, "Assumed nominal battery terminal voltage."),
    "battery_current": (PROVISIONAL, "battery_power / battery_voltage."),
    "battery_charge_power": (PROVISIONAL, "Charging magnitude, kW."),
    "battery_discharge_power": (PROVISIONAL, "Discharging magnitude, kW."),

    "local_bus_voltage": (
        PROVISIONAL,
        "Droop approximation: V_nominal - virtual_droop x imported power.",
    ),
    "local_bus_current": (
        PROVISIONAL,
        "Total power injected into the local DC bus / local_bus_voltage.",
    ),

    "community_bus_voltage": (
        PROVISIONAL,
        "Shared community bus droop on the aggregate community imbalance. "
        "Identical for every house at a given timestep.",
    ),

    "tie_power": (
        PROVISIONAL,
        "Bidirectional bus-tie power. > 0 house -> community, "
        "< 0 community -> house. Limited by tie.max_power_kw.",
    ),
    "tie_current": (
        PROVISIONAL,
        "|tie_power| / local_bus_voltage. Magnitude; direction is in tie_power.",
    ),

    "net_power": (AUSGRID, "pv_power - load_power; both Ausgrid-derived."),
    "surplus_power": (AUSGRID, "max(net_power, 0)."),
    "deficit_power": (AUSGRID, "max(-net_power, 0)."),
    "energy_exchange": (PROVISIONAL, "tie_power x interval hours, in kWh."),

    "curtailed_power": (
        PROVISIONAL,
        "Surplus above the bus-tie rating; recorded so the household "
        "energy balance closes exactly.",
    ),
    "unserved_power": (
        PROVISIONAL,
        "Deficit above the bus-tie rating; recorded for the same reason.",
    ),
}


def provenance_for(columns):
    """Provenance entries for the columns actually written, in order."""
    return {
        c: {"provenance": PROVENANCE[c][0], "description": PROVENANCE[c][1]}
        for c in columns
        if c in PROVENANCE
    }


def summary(columns):
    """Count of columns per provenance label."""
    counts = {AUSGRID: 0, PROVISIONAL: 0}
    for c in columns:
        if c in PROVENANCE:
            counts[PROVENANCE[c][0]] += 1
    return counts
