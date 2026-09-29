"""
Configuration loading.

Every assumption of the provisional model lives in
config/nanogrid_config.yaml. This module turns that file into typed
dataclasses so the rest of the package never hardcodes a number and
never guesses a key name.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "nanogrid_config.yaml"


def _build(cls, data):
    """Instantiate a config dataclass, rejecting unknown/missing keys.

    A typo in the YAML would otherwise be silently ignored and the code
    would run on a default that nobody chose - exactly the kind of
    invisible assumption this package exists to avoid.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown config key(s) {sorted(unknown)}"
        )
    return cls(**data)


@dataclass(frozen=True)
class SourceConfig:
    db_path: str
    csv_path: str
    interval_hours: float
    n_houses: int
    customer_ids: Optional[Sequence[str]] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    max_days: Optional[int] = None


@dataclass(frozen=True)
class PVConfig:
    capacity_kw: float
    nominal_voltage: float


@dataclass(frozen=True)
class LoadConfig:
    dc_load_fraction: float
    nominal_voltage: float


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float
    initial_soc: float
    min_soc: float
    max_soc: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    nominal_voltage: float

    @property
    def usable_capacity_kwh(self) -> float:
        """Energy between min_soc and max_soc - the window the battery
        is actually allowed to cycle within."""
        return self.capacity_kwh * (self.max_soc - self.min_soc)


@dataclass(frozen=True)
class BusConfig:
    local_nominal_voltage: float
    community_nominal_voltage: float
    local_virtual_droop: float
    community_droop: float
    voltage_min: float
    voltage_max: float
    community_voltage_min: float
    community_voltage_max: float


@dataclass(frozen=True)
class TieConfig:
    max_power_kw: float


@dataclass(frozen=True)
class NetworkConfig:
    loss_fraction: float
    topology: str
    topology_path: str


@dataclass(frozen=True)
class ObservationConfig:
    features: Sequence[str]


@dataclass(frozen=True)
class OutputConfig:
    dataset_path: str
    observations_path: str
    community_path: str
    metadata_path: str
    plots_dir: str


@dataclass(frozen=True)
class ValidationConfig:
    power_tolerance_kw: float
    energy_balance_tolerance_kw: float
    ohm_tolerance_a: float
    profile_correlation_min: float


@dataclass(frozen=True)
class NanogridConfig:
    source: SourceConfig
    pv: PVConfig
    load: LoadConfig
    battery: BatteryConfig
    bus: BusConfig
    tie: TieConfig
    network: NetworkConfig
    observation: ObservationConfig
    output: OutputConfig
    validation: ValidationConfig
    path: Path = field(default=DEFAULT_CONFIG_PATH)
    raw: dict = field(default_factory=dict, repr=False)

    def resolve(self, relative_path: str) -> Path:
        """Config paths are written relative to the project root, so a
        run from any working directory lands in the same place."""
        p = Path(relative_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


_SECTIONS = {
    "source": SourceConfig,
    "pv": PVConfig,
    "load": LoadConfig,
    "battery": BatteryConfig,
    "bus": BusConfig,
    "tie": TieConfig,
    "network": NetworkConfig,
    "observation": ObservationConfig,
    "output": OutputConfig,
    "validation": ValidationConfig,
}


def load_config(path=DEFAULT_CONFIG_PATH) -> NanogridConfig:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    missing = set(_SECTIONS) - set(raw)
    if missing:
        raise ValueError(f"config is missing section(s) {sorted(missing)}")

    sections = {name: _build(cls, raw[name]) for name, cls in _SECTIONS.items()}
    cfg = NanogridConfig(path=path, raw=raw, **sections)
    _check(cfg)
    return cfg


def _check(cfg: NanogridConfig) -> None:
    """Fail fast on configurations that cannot describe a real nanogrid."""

    b = cfg.battery
    if not 0.0 <= b.min_soc < b.max_soc <= 1.0:
        raise ValueError("battery: require 0 <= min_soc < max_soc <= 1")
    if not b.min_soc <= b.initial_soc <= b.max_soc:
        raise ValueError("battery: initial_soc must lie in [min_soc, max_soc]")
    for name in ("charge_efficiency", "discharge_efficiency"):
        if not 0.0 < getattr(b, name) <= 1.0:
            raise ValueError(f"battery: {name} must lie in (0, 1]")

    for name, voltage in (
        ("pv.nominal_voltage", cfg.pv.nominal_voltage),
        ("load.nominal_voltage", cfg.load.nominal_voltage),
        ("battery.nominal_voltage", b.nominal_voltage),
        ("bus.local_nominal_voltage", cfg.bus.local_nominal_voltage),
        ("bus.community_nominal_voltage", cfg.bus.community_nominal_voltage),
    ):
        if voltage <= 0:
            raise ValueError(f"{name} must be positive (used as P / V)")

    if cfg.bus.voltage_min >= cfg.bus.voltage_max:
        raise ValueError("bus: voltage_min must be below voltage_max")
    if cfg.bus.community_voltage_min >= cfg.bus.community_voltage_max:
        raise ValueError("bus: community_voltage_min must be below max")
    if not 0.0 <= cfg.network.loss_fraction < 1.0:
        raise ValueError("network: loss_fraction must lie in [0, 1)")
    if cfg.source.n_houses < 1:
        raise ValueError("source: n_houses must be >= 1")
    if cfg.source.interval_hours <= 0:
        raise ValueError("source: interval_hours must be positive")
