"""
MARL observation extraction.

The generated dataset holds the COMPLETE provisional simulation state.
The environment shows an agent only a subset of it - by default:

    [pv_power, load_power, battery_soc,
     local_bus_voltage, tie_power, community_bus_voltage]

Changing the observation space means editing `observation.features` in
config/nanogrid_config.yaml and nothing else. Every module that needs to
know the observation shape asks this one.
"""

import numpy as np

# Columns that identify a row rather than describe it. Always written
# alongside the observation so a row can be joined back to the full
# state, but never part of the observation vector itself.
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
    """Joint observation at one timestep: (n_houses, n_features) array,
    ordered by house_id - the shape a MARL environment's reset()/step()
    returns."""
    rows = dataset[dataset["timestamp"] == timestamp].sort_values("house_id")
    return np.asarray(rows[list(features)], dtype=np.float32)
