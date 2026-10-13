"""Configuration loading: YAML files under config/ plus a few environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(os.environ.get("IMMS_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    label: str
    adstock_max_lag: int
    prior_decay_mean: float
    prior_roi_mean: float
    prior_roi_sd: float


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@cache
def channels() -> list[ChannelSpec]:
    raw = _load_yaml("channels.yaml")["channels"]
    return [ChannelSpec(**row) for row in raw]


@cache
def channel_names() -> list[str]:
    return [c.name for c in channels()]


@cache
def channel_labels() -> dict[str, str]:
    return {c.name: c.label for c in channels()}


@cache
def control_names() -> list[str]:
    return list(_load_yaml("channels.yaml")["controls"])


@cache
def model_config() -> dict:
    return _load_yaml("model.yaml")


@cache
def geo_config() -> dict:
    return _load_yaml("geo_tests.yaml")


@cache
def tv_config() -> dict:
    return _load_yaml("tv.yaml")


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql+psycopg://imms:imms@localhost:5432/imms")


def value_per_conversion() -> float:
    return float(os.environ.get("VALUE_PER_CONVERSION", "180"))
