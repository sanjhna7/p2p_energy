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

SOURCES = {"REAL", "SPEC", "SCENARIO"}


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
class Config:
    processed_dir: Path
    scenario: ScenarioConfig
    battery: BatteryConfig
    references: dict
    path: Path

    def provenance(self):
        """(name, value, source, citation) for every Param, for metadata and docs."""
        rows = []
        for section in ("scenario", "battery"):
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

    expected = {"references", "data", "scenario", "battery"}
    if set(raw) != expected:
        raise ValueError(f"config sections must be {sorted(expected)}, got {sorted(raw)}")

    references = {k: Reference(**v) for k, v in raw["references"].items()}
    processed = Path(raw["data"]["processed_dir"])
    cfg = Config(
        processed_dir=processed if processed.is_absolute() else PROJECT_ROOT / processed,
        scenario=_section(ScenarioConfig, raw["scenario"], "scenario", references),
        battery=_section(BatteryConfig, raw["battery"], "battery", references),
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
    if cfg.scenario.n_houses.value < 1 or cfg.scenario.days.value < 1:
        raise ValueError("scenario: n_houses and days must be >= 1")
