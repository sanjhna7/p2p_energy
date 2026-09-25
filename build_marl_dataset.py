"""
Entry point for the provisional MARL development dataset.

    python build_marl_dataset.py

Generates the dataset from the real Ausgrid profiles, validates it, and
writes the visual checks. Everything it assumes lives in
config/nanogrid_config.yaml; see docs/PROVISIONAL_DATASET.md for what is
real data and what is provisional model output.

This does NOT touch the earlier SQLite pipeline (now parked in
removedfornnow/) - it only reads meter_readings from the database that
pipeline produced.
"""

import argparse
import sys

from nanogrid import plots, validate
from nanogrid.build_dataset import build
from nanogrid.config import DEFAULT_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    dataset, observations, community, meta, cfg = build(args.config)

    print("\nSample rows (full simulation state):")
    print(dataset.head(3).to_string(index=False))

    print("\nSample rows (agent observations):")
    print(observations.head(3).to_string(index=False))

    print(f"\nObservation shape (per agent) : "
          f"{tuple(meta['observation']['per_agent_shape'])}")
    print(f"Joint observation shape       : "
          f"{tuple(meta['observation']['joint_shape'])}")
    print(f"Per-house state dim           : {meta['state']['per_house_state_dim']}")
    print(f"Joint state shape             : "
          f"{tuple(meta['state']['joint_state_shape'])}")

    print()
    report = validate.validate(
        dataset, community, cfg, profiles=None
    )
    print("=" * 62)
    print("Validation")
    print("=" * 62)
    print(report.render())

    # The profile-fidelity checks need the original source, reloaded here
    # so the fast path above stays independent of it.
    from nanogrid.ausgrid_source import load_profiles
    profile_report = validate.Report()
    validate._check_profiles(dataset, load_profiles(cfg), cfg, profile_report)
    print(profile_report.render())

    failures = report.failed + profile_report.failed
    total = len(report.results) + len(profile_report.results)
    print(f"\n{total - len(failures)}/{total} checks passed.")

    if not args.no_plots:
        print()
        plots.run(args.config)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
