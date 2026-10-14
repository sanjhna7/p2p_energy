"""P2P market clearing at the mid-market rate (MMR).

Each interval, after the batteries have acted, every house is left with a
residual power: > 0 surplus to sell, < 0 deficit to buy. Surplus and
deficit are matched inside the community first, then the remainder goes
to the grid.

    traded   = min(total surplus, total deficit)
    sellers  share `traded` in proportion to their surplus
    buyers   share `traded` in proportion to their deficit
    P2P price = MMR = (P_grid_buy + P_feed_in) / 2

A seller earns MMR instead of the feed-in tariff and a buyer pays MMR
instead of the retail price, so with P_feed_in < MMR < P_grid_buy every
P2P trade leaves both sides better off than trading with the grid.
This is financial settlement only; the physical flows are in powerflow.py.
"""

from dataclasses import dataclass

import numpy as np

from p2p.config import TariffConfig

CENTS_PER_AUD = 100.0


@dataclass(frozen=True)
class Clearing:
    p2p_sold_kw: np.ndarray      # per house, >= 0
    p2p_bought_kw: np.ndarray    # per house, >= 0
    grid_export_kw: np.ndarray   # per house, >= 0
    grid_import_kw: np.ndarray   # per house, >= 0
    cost_aud: np.ndarray         # per house for the interval; negative = earned


def clear(residual_kw, tariff: TariffConfig, dt_hours: float, p2p: bool = True) -> Clearing:
    residual = np.asarray(residual_kw, dtype=float)
    surplus = np.maximum(residual, 0.0)
    deficit = np.maximum(-residual, 0.0)

    supply, demand = surplus.sum(), deficit.sum()
    traded = min(supply, demand) if p2p else 0.0
    sold = surplus * (traded / supply) if supply > 0 else np.zeros_like(surplus)
    bought = deficit * (traded / demand) if demand > 0 else np.zeros_like(deficit)

    export, imported = surplus - sold, deficit - bought
    buy, fit, mmr = (tariff.grid_buy_c_per_kwh.value, tariff.feed_in_c_per_kwh.value, tariff.mmr_c_per_kwh)
    cents = (imported * buy - export * fit + (bought - sold) * mmr) * dt_hours
    return Clearing(sold, bought, export, imported, cents / CENTS_PER_AUD)
