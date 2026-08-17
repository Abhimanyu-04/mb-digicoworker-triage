"""Task 1: generate the synthetic factory telemetry dataset.

Simulates 1000 units over 48h and injects one root-cause anomaly (picked in
config.py or via --scenario). Failure is probabilistic - each unit gets a fail
probability from a logistic model, no hard thresholds anywhere.

Usage:
    python data_generator.py [--scenario material_batch|machine_drift|operator_shift]
                             [--output factory_telemetry.csv] [--seed 42]
"""

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config import GENERATOR_CONFIG

BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(path_str):
    p = Path(path_str)
    return str(p if p.is_absolute() else BASE_DIR / p)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _build_timeline(cfg, rng):
    n = cfg["n_units"]
    duration_s = cfg["duration_hours"] * 3600
    base = np.linspace(0, duration_s, n, endpoint=False)
    elapsed = np.clip(base + rng.normal(0, 20, n), 0, duration_s - 1)
    elapsed.sort()
    start = pd.Timestamp(cfg["start_time"])
    df = pd.DataFrame({
        "unit_id": [f"UNIT_{i + 1:04d}" for i in range(n)],
        "timestamp": start + pd.to_timedelta(elapsed, unit="s"),
    })
    df["elapsed_hours"] = elapsed / 3600.0
    return df


def _assign_shifts(df, cfg):
    shift_cfg = cfg["shifts"]
    block = (df["elapsed_hours"] // shift_cfg["block_hours"]).astype(int)
    ids = shift_cfg["ids"]
    df["operator_shift_id"] = [ids[b % len(ids)] for b in block]


def _assign_batches(df, cfg):
    # drums are consumed sequentially -> contiguous unit blocks per batch.
    # this is what makes a bad batch naturally overlap one shift's window.
    ids = cfg["batches"]["ids"]
    per_batch = int(np.ceil(len(df) / len(ids)))
    df["sealant_batch_id"] = [ids[min(i // per_batch, len(ids) - 1)]
                              for i in range(len(df))]


def _generate_telemetry(df, cfg, rng):
    for param, p in cfg["machine_params"].items():
        df[param] = rng.normal(p["setpoint"], p["std"], len(df))


def _apply_machine_drift(df, cfg):
    # failing regulator: linear ramp down from drift_start_hour to the end.
    # no extra fail-probability term needed, the physics weight picks it up.
    a = cfg["anomaly"]["machine_drift"]
    duration = cfg["duration_hours"]
    frac = (df["elapsed_hours"] - a["drift_start_hour"]) / (duration - a["drift_start_hour"])
    df[a["param"]] = df[a["param"]] - frac.clip(lower=0.0) * a["drop_at_end"]


def _fail_probability(df, cfg):
    fm = cfg["failure_model"]
    w = fm["weights"]
    mp = cfg["machine_params"]

    logit = np.full(len(df), fm["baseline_logit"])
    # low pressure -> thin bead, low oven -> undercure, nozzle bad either way
    logit += w["dispenser_pressure_psi"] * np.maximum(
        0.0, mp["dispenser_pressure_psi"]["setpoint"] - df["dispenser_pressure_psi"])
    logit += w["nozzle_temp_c"] * np.abs(
        df["nozzle_temp_c"] - mp["nozzle_temp_c"]["setpoint"])
    logit += w["oven_temp_c"] * np.maximum(
        0.0, mp["oven_temp_c"]["setpoint"] - df["oven_temp_c"])

    scenario = cfg["anomaly"]["scenario"]
    if scenario == "material_batch":
        a = cfg["anomaly"]["material_batch"]
        logit += np.where(df["sealant_batch_id"] == a["bad_batch"], a["effect_logit"], 0.0)
    elif scenario == "operator_shift":
        a = cfg["anomaly"]["operator_shift"]
        logit += np.where(df["operator_shift_id"] == a["bad_shift"], a["effect_logit"], 0.0)

    return _sigmoid(logit)


def _corrupt_manual_entries(df, cfg, rng):
    # applied after pass/fail is decided - these are recording errors,
    # not physical ones
    dq = cfg["data_quality"]
    n_bad = int(len(df) * dq["entry_error_rate"])
    rows = rng.choice(len(df), size=n_bad, replace=False)
    cols = rng.choice(["operator_shift_id", "sealant_batch_id"], size=n_bad)
    styles = rng.random(n_bad)
    for row, col, style in zip(rows, cols, styles):
        val = df.at[row, col]
        if style < dq["blank_fraction"]:
            df.at[row, col] = ""
        elif style < 0.6:
            df.at[row, col] = val.lower()
        else:
            df.at[row, col] = f"  {val} "
    return n_bad


def generate(cfg):
    scenario = cfg["anomaly"]["scenario"]
    valid = ("material_batch", "machine_drift", "operator_shift")
    if scenario not in valid:
        raise ValueError(f"Unknown anomaly scenario '{scenario}'; expected one of {valid}")

    rng = np.random.default_rng(cfg["random_seed"])
    df = _build_timeline(cfg, rng)
    _assign_shifts(df, cfg)
    _assign_batches(df, cfg)
    _generate_telemetry(df, cfg, rng)
    if scenario == "machine_drift":
        _apply_machine_drift(df, cfg)

    p_fail = _fail_probability(df, cfg)
    df["pressure_test_result"] = np.where(rng.random(len(df)) < p_fail, "FAIL", "PASS")

    n_corrupted = _corrupt_manual_entries(df, cfg, rng)

    for param in cfg["machine_params"]:
        df[param] = df[param].round(2)
    df = df.drop(columns=["elapsed_hours"])

    fail_rate = (df["pressure_test_result"] == "FAIL").mean()
    print(f"Scenario           : {scenario}")
    print(f"Units generated    : {len(df)}")
    print(f"Overall fail rate  : {fail_rate:.1%}")
    print(f"Corrupted entries  : {n_corrupted}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic factory telemetry.")
    parser.add_argument("--scenario", choices=["material_batch", "machine_drift", "operator_shift"])
    parser.add_argument("--output", default="factory_telemetry.csv")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg = copy.deepcopy(GENERATOR_CONFIG)
    if args.scenario:
        cfg["anomaly"]["scenario"] = args.scenario
    if args.seed is not None:
        cfg["random_seed"] = args.seed

    df = generate(cfg)
    output_path = _resolve_path(args.output)
    df.to_csv(output_path, index=False)
    print(f"Dataset written to : {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
