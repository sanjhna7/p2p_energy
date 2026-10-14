"""Evaluation measures (slide 27) from an environment log.

Energy is power x dt summed over the episode. Community-level ratios:

  self_sufficiency   share of household load NOT bought from the grid
  pv_local_use       share of PV generation NOT exported to the grid
                     (used at home, stored, or shared with neighbours)
  savings_vs_grid    reduction in total cost against the grid_only log
  fairness_jain      Jain's index of per-house savings, 1 = everyone gains
                     equally; houses that lose money count as zero gain
"""

import numpy as np


def jain(x) -> float:
    """Jain's fairness index; undefined (nan) when nobody gains anything."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return float(x.sum() ** 2 / (len(x) * (x**2).sum())) if (x**2).sum() > 0 else float("nan")


def summarise(log: dict, dt_hours: float, baseline: dict = None) -> dict:
    e = lambda k: float(log[k].sum() * dt_hours)   # noqa: E731  kWh over all houses and steps
    load, pv = e("load_kw"), e("pv_kw")
    out = {
        "load_kwh": load,
        "pv_kwh": pv,
        "grid_import_kwh": e("grid_import_kw"),
        "grid_export_kwh": e("grid_export_kw"),
        "p2p_traded_kwh": e("p2p_sold_kw"),
        "self_sufficiency": 1 - e("grid_import_kw") / load,
        "pv_local_use": 1 - e("grid_export_kw") / pv if pv > 0 else float("nan"),
        "cost_aud": float(log["cost_aud"].sum()),
        # Network figures cover converged steps only; see non_converged_steps.
        "network_loss_kwh": float(np.nansum(log["loss_kw"]) * dt_hours),
        "v_min": float(np.nanmin(log["v_min"])) if log["converged"].any() else float("nan"),
        "v_max": float(np.nanmax(log["v_max"])) if log["converged"].any() else float("nan"),
        "i_max_a": float(np.nanmax(log["i_max_abs"])) if log["converged"].any() else float("nan"),
        "non_converged_steps": int((~log["converged"]).sum()),
        "blocked_steps": int(log["blocked"].sum()),
        "inherent_violation_steps": int(log["inherent_violation"].sum()),
    }
    if baseline is not None:
        per_house = baseline["cost_aud"].sum(axis=0) - log["cost_aud"].sum(axis=0)
        out["savings_vs_grid_aud"] = float(per_house.sum())
        out["savings_vs_grid_pct"] = float(per_house.sum() / baseline["cost_aud"].sum() * 100)
        out["fairness_jain"] = jain(per_house)
        out["houses_worse_off"] = int((per_house < -1e-9).sum())
    return out
