"""
P2P Energy — main entry point.

Usage:
    python main.py                    Build the MARL dataset (default)
    python main.py --no-plots         Build without generating plots
    python main.py --synthetic        Generate synthetic DC microgrid data
    python main.py --gat              Train the GAT voltage prediction model

Generates the dataset from real Ausgrid profiles, validates it, and
writes the visual checks. Everything it assumes lives in
config/nanogrid_config.yaml; see docs/PROVISIONAL_DATASET.md for what is
real data and what is provisional model output.
"""

import argparse
import sys

import simulation
import data
from config import DEFAULT_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser(
        description="P2P Energy Management — dataset generation and model training"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Path to nanogrid config YAML")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic DC microgrid data instead")
    parser.add_argument("--gat", action="store_true",
                        help="Train the GAT voltage prediction model")
    args = parser.parse_args()

    # --- Synthetic data generation ---
    if args.synthetic:
        data.run_synthetic()
        return 0

    # --- GAT model training ---
    if args.gat:
        import model
        model.run_gat()
        return 0

    # --- Default: MARL dataset pipeline ---
    dataset, observations, community, meta, cfg = simulation.build_dataset(
        args.config
    )

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
    report = simulation.validate(
        dataset, community, cfg, profiles=None
    )
    print("=" * 62)
    print("Validation")
    print("=" * 62)
    print(report.render())

    # Profile-fidelity checks need the original source.
    profile_report = simulation.Report()
    profiles = data.load_profiles(cfg)
    simulation._check_profiles(dataset, profiles, cfg, profile_report)
    print(profile_report.render())

    failures = report.failed + profile_report.failed
    total = len(report.results) + len(profile_report.results)
    print(f"\n{total - len(failures)}/{total} checks passed.")

    if not args.no_plots:
        print()
        simulation.run_plots(args.config)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
