"""Configuration loading with provenance.

Every number in config/p2p.yaml is a Param: a value plus where it came
from. The loader rejects unknown keys, unknown sources, and SPEC values
whose reference is missing, so no number can enter the code without a
stated origin.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "p2p.yaml"

SOURCES = {"REAL", "SPEC", "SCENARIO", "PENDING"}


@dataclass(frozen=True)
class Param:
    value: object
    source: str
    ref: Optional[str] = None
    note: str = ""


@dataclass(frozen=True)
class Reference:
    citation: str
    url: str
    sha256: str = ""


@dataclass(frozen=True)
class ScenarioConfig:
    n_houses: Param
    start: Param
    days: Param


@dataclass(frozen=True)
class BatteryConfig:
    model: str
    usable_energy_kwh: Param
    nominal_voltage_v: Param
    operating_voltage_min_v: Param
    operating_voltage_max_v: Param
    max_continuous_current_a: Param
    rated_dc_power_kw: Param
    round_trip_efficiency: Param
    initial_soc: Param


@dataclass(frozen=True)
class TariffConfig:
    grid_buy_c_per_kwh: Param
    feed_in_c_per_kwh: Param

    @property
    def mmr_c_per_kwh(self) -> float:
        """Mid-market rate, (P_buy + P_feed-in) / 2 (slide 17)."""
        return (self.grid_buy_c_per_kwh.value + self.feed_in_c_per_kwh.value) / 2


@dataclass(frozen=True)
class NetworkConfig:
    bus_voltage_v: Param
    voltage_min_v: Param
    voltage_max_v: Param
    line_current_max_a: Param
    grid_tie_max_kw: Param


@dataclass(frozen=True)
class RewardConfig:
    own_voltage_aud_per_v: Param
    network_voltage_aud_per_v: Param
    line_overload_aud_per_a: Param
    grid_overload_aud_per_kw: Param
    collapse_aud: Param
    blocked_aud: Param
    infeasible_aud_per_kwh: Param


@dataclass(frozen=True)
class TrainingConfig:
    start: Param
    days: Param
    episode_steps: Param
    iterations: Param
    episodes_per_iteration: Param
    learning_rate: Param
    gamma: Param
    gae_lambda: Param
    clip: Param
    ppo_epochs: Param
    minibatch_steps: Param
    entropy_coef: Param
    value_coef: Param
    reward_scale: Param
    seed: Param


@dataclass(frozen=True)
class Config:
    processed_dir: Path
    scenario: ScenarioConfig
    battery: BatteryConfig
    tariff: TariffConfig
    network: NetworkConfig
    reward: RewardConfig
    training: TrainingConfig
    references: dict
    path: Path

    def provenance(self):
        """(name, value, source, citation) for every Param, for metadata and docs."""
        rows = []
        for section in ("scenario", "battery", "tariff", "network", "reward", "training"):
            obj = getattr(self, section)
            for f in fields(obj):
                p = getattr(obj, f.name)
                if isinstance(p, Param):
                    cite = self.references[p.ref].citation if p.ref else ""
                    rows.append((f"{section}.{f.name}", p.value, p.source, cite))
        return rows


def _check_keys(cls, data, where):
    known = {f.name for f in fields(cls)}
    unknown, missing = set(data) - known, known - set(data)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    if missing:
        raise ValueError(f"{where}: missing key(s) {sorted(missing)}")


def _param(data, where, references):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected {{value, source, ...}}, got {data!r}")
    _check_keys(Param, {"note": "", "ref": None, **data}, where)
    p = Param(**data)
    if p.source not in SOURCES:
        raise ValueError(f"{where}: source {p.source!r} not in {sorted(SOURCES)}")
    if p.source == "SPEC" and p.ref not in references:
        raise ValueError(f"{where}: SPEC value needs a ref from `references`, got {p.ref!r}")
    if p.ref is not None and p.ref not in references:
        raise ValueError(f"{where}: unknown ref {p.ref!r}")
    if p.source == "PENDING" and not p.note:
        raise ValueError(f"{where}: PENDING value needs a note saying what replaces it")
    if p.value is None and p.source != "PENDING":
        raise ValueError(f"{where}: only a PENDING value may be null")
    return p


def _section(cls, data, where, references):
    _check_keys(cls, data, where)
    values = {}
    for f in fields(cls):
        raw = data[f.name]
        values[f.name] = _param(raw, f"{where}.{f.name}", references) if f.type is Param else raw
    return cls(**values)


def load_config(path=DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    expected = {"references", "data", "scenario", "battery", "tariff", "network", "reward", "training"}
    if set(raw) != expected:
        raise ValueError(f"config sections must be {sorted(expected)}, got {sorted(raw)}")

    references = {k: Reference(**v) for k, v in raw["references"].items()}
    processed = Path(raw["data"]["processed_dir"])
    cfg = Config(
        processed_dir=processed if processed.is_absolute() else PROJECT_ROOT / processed,
        scenario=_section(ScenarioConfig, raw["scenario"], "scenario", references),
        battery=_section(BatteryConfig, raw["battery"], "battery", references),
        tariff=_section(TariffConfig, raw["tariff"], "tariff", references),
        network=_section(NetworkConfig, raw["network"], "network", references),
        reward=_section(RewardConfig, raw["reward"], "reward", references),
        training=_section(TrainingConfig, raw["training"], "training", references),
        references=references,
        path=path,
    )
    _check(cfg)
    return cfg


def _check(cfg: Config) -> None:
    b = cfg.battery
    if b.usable_energy_kwh.value <= 0 or b.rated_dc_power_kw.value <= 0:
        raise ValueError("battery: energy and power ratings must be positive")
    if not 0.0 < b.round_trip_efficiency.value <= 1.0:
        raise ValueError("battery: round_trip_efficiency must lie in (0, 1]")
    if not 0.0 <= b.initial_soc.value <= 1.0:
        raise ValueError("battery: initial_soc must lie in [0, 1]")
    if not b.operating_voltage_min_v.value < b.nominal_voltage_v.value < b.operating_voltage_max_v.value:
        raise ValueError("battery: nominal voltage must lie inside the operating range")
    t = cfg.tariff
    if not 0 <= t.feed_in_c_per_kwh.value < t.grid_buy_c_per_kwh.value:
        raise ValueError("tariff: require 0 <= feed-in < grid buy price")
    n = cfg.network
    if not n.voltage_min_v.value < n.bus_voltage_v.value < n.voltage_max_v.value:
        raise ValueError("network: bus voltage must lie inside [voltage_min_v, voltage_max_v]")
    for f in fields(cfg.reward):
        if getattr(cfg.reward, f.name).value < 0:
            raise ValueError(f"reward.{f.name}: penalty weights must be >= 0")
    if cfg.scenario.n_houses.value < 1 or cfg.scenario.days.value < 1:
        raise ValueError("scenario: n_houses and days must be >= 1")
