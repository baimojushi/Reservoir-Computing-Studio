"""Versioned RC experiment configuration and strict input validation."""
import copy
import math

DEFAULT = {
    "dataset": {
        "kind": "sine",
        "length": 2000,
        "noise": 0.01,
        "seed": 42,
        "dataset_id": None,
        "column": None,
    },
    "model": {
        "units": 120,
        "spectral_radius": 0.90,
        "leak_rate": 0.35,
        "density": 0.08,
        "input_scaling": 0.70,
        "ridge": 0.00001,
        "seed": 42,
        "washout": 50,
        "horizon": 1,
    },
    "split": {"train": 0.70, "validation": 0.15},
}

class ConfigurationError(ValueError):
    pass

def _numeric(value, name, minimum, maximum, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name}: expected a number")
    if not math.isfinite(value) or not (minimum <= value <= maximum):
        raise ConfigurationError(f"{name}: must be in [{minimum}, {maximum}]")
    if integer and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigurationError(f"{name}: expected an integer")

def normalize_config(incoming=None):
    if incoming is None:
        incoming = {}
    if not isinstance(incoming, dict):
        raise ConfigurationError("config: expected a JSON object")
    cfg = copy.deepcopy(DEFAULT)
    for section, values in incoming.items():
        if section not in cfg or not isinstance(values, dict):
            raise ConfigurationError(f"unknown/invalid config section: {section}")
        for key, value in values.items():
            if key not in cfg[section]:
                raise ConfigurationError(f"unknown config key: {section}.{key}")
            cfg[section][key] = value

    data, model, split = cfg["dataset"], cfg["model"], cfg["split"]
    if data["kind"] not in ("sine", "mackey_glass", "csv"):
        raise ConfigurationError("dataset.kind: sine, mackey_glass or csv")
    _numeric(data["length"], "dataset.length", 150, 20000, integer=True)
    _numeric(data["noise"], "dataset.noise", 0.0, 1.0)
    _numeric(data["seed"], "dataset.seed", 0, 2147483647, integer=True)
    if data["kind"] == "csv":
        if not isinstance(data["dataset_id"], str) or not (
            len(data["dataset_id"]) == 32 and
            all(char in "0123456789abcdef" for char in data["dataset_id"])
        ):
            raise ConfigurationError("CSV experiments require a valid imported dataset_id")
        if data["column"] is not None and (
            not isinstance(data["column"], str) or len(data["column"]) > 200
        ):
            raise ConfigurationError("dataset.column: expected a column name")
    _numeric(model["units"], "model.units", 10, 350, integer=True)
    _numeric(model["spectral_radius"], "model.spectral_radius", 0.01, 2.0)
    _numeric(model["leak_rate"], "model.leak_rate", 0.01, 1.0)
    _numeric(model["density"], "model.density", 0.01, 1.0)
    _numeric(model["input_scaling"], "model.input_scaling", 0.001, 10.0)
    _numeric(model["ridge"], "model.ridge", 0.0, 100.0)
    _numeric(model["seed"], "model.seed", 0, 2147483647, integer=True)
    _numeric(model["washout"], "model.washout", 0, 2000, integer=True)
    _numeric(model["horizon"], "model.horizon", 1, 100, integer=True)
    _numeric(split["train"], "split.train", 0.50, 0.90)
    _numeric(split["validation"], "split.validation", 0.05, 0.30)
    if split["train"] + split["validation"] >= 0.95:
        raise ConfigurationError("train + validation must be < 0.95")
    if data["kind"] != "csv":
        count = data["length"] - model["horizon"]
        if model["washout"] + 20 >= int(count * split["train"]):
            raise ConfigurationError("not enough train samples after washout")
    return cfg
