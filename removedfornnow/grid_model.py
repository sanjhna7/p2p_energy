"""
Baseline grid / bus-voltage model.

IMPORTANT LIMITATION (call this out to anyone reading this file):
The Ausgrid CSV has no network topology — no line impedances, no bus
connectivity, no transformer ratings. Real power-flow (or a GNN trained
on power-flow) needs that topology. Until we have it (e.g. a synthetic
feeder like an IEEE 13/34-bus test case commonly paired with this exact
dataset in papers), this module uses a simplified linear sensitivity
proxy: voltage droops proportionally to net power drawn at a bus.

This is a *baseline*, not a real power-flow solver. Its only job right
now is to give every part of the data layer (state.py, simulation.py)
something to read/write with the right shape, so nothing downstream is
blocked. When real topology or a GNN surrogate is ready, only the body
of `compute_bus_voltage` (and friends) should need to change — the
function signature is the swap-in contract:

    net_injection_kw (per bus) -> voltage_pu, transformer_loading_pct, losses_kw

Do not extend this file with anything ML-related. GNN work is
explicitly a separate, later step.
"""

from dataclasses import dataclass

NOMINAL_VOLTAGE_PU = 1.0
VOLTAGE_SENSITIVITY = 0.01     # pu drop per kW net draw at a bus (placeholder)
TRANSFORMER_RATED_KW = 500.0    # placeholder feeder/transformer rating
LOSS_COEFFICIENT = 0.02         # fraction of |net power| assumed lost (placeholder)


@dataclass
class BusResult:
    voltage_pu: float
    transformer_loading_pct: float
    losses_kw: float


def compute_bus_voltage(
    net_injection_kw: float,
    sensitivity: float = VOLTAGE_SENSITIVITY,
    nominal_voltage_pu: float = NOMINAL_VOLTAGE_PU,
) -> float:
    """
    net_injection_kw > 0 means the bus is exporting (net generation),
    < 0 means the bus is importing (net load) after battery buffering.

    Baseline proxy only: voltage rises slightly on export, droops
    slightly on import. Replace with a real power-flow / GNN call later.
    """
    return round(nominal_voltage_pu + sensitivity * net_injection_kw, 4)


def compute_transformer_loading(total_load_kw: float, rated_kw: float = TRANSFORMER_RATED_KW) -> float:
    if rated_kw <= 0:
        return 0.0
    return round(100 * abs(total_load_kw) / rated_kw, 2)


def compute_losses(total_flow_kw: float, loss_coefficient: float = LOSS_COEFFICIENT) -> float:
    return round(abs(total_flow_kw) * loss_coefficient, 4)


def compute_bus_state(net_injection_kw: float) -> BusResult:
    """Convenience wrapper bundling the three bus-level outputs together.

    All three figures are derived from the same net_injection_kw - the
    power this bus actually exchanges with the transformer/upstream
    grid *after* any local battery buffering (and, once multiple houses
    share a bus, after any local P2P netting between them). Feeding
    transformer_loading a different, pre-netting number would overstate
    loading: two houses on the same bus trading surplus/deficit locally
    never touch the transformer for that portion of their power at all.
    """
    return BusResult(
        voltage_pu=compute_bus_voltage(net_injection_kw),
        transformer_loading_pct=compute_transformer_loading(net_injection_kw),
        losses_kw=compute_losses(net_injection_kw),
    )