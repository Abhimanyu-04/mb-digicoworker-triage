"""Load, validate and clean the telemetry CSV."""

import pandas as pd

REQUIRED_COLUMNS = [
    "dispenser_pressure_psi",
    "nozzle_temp_c",
    "oven_temp_c",
    "operator_shift_id",
    "sealant_batch_id",
    "pressure_test_result",
]
NUMERIC_COLUMNS = ["dispenser_pressure_psi", "nozzle_temp_c", "oven_temp_c"]
CATEGORICAL_COLUMNS = ["operator_shift_id", "sealant_batch_id"]


class IngestError(Exception):
    pass


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def ingest(path):
    """Returns (clean_df, quality_report). Raises IngestError on fatal problems."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        fallback = BASE_DIR / p
        if fallback.exists():
            p = fallback
    try:
        df = pd.read_csv(p)
    except FileNotFoundError:
        raise IngestError(f"Input file not found: {path}")
    except Exception as exc:
        raise IngestError(f"Could not parse CSV '{path}': {exc}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise IngestError("Dataset contains no rows.")

    quality = {"rows_loaded": len(df), "entries_normalized": 0,
               "rows_excluded": 0, "exclusion_reasons": {}}
    bad_mask = pd.Series(False, index=df.index)

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(coerced.isna().sum())
        if n_bad:
            quality["exclusion_reasons"][f"invalid {col}"] = n_bad
            bad_mask |= coerced.isna()
        df[col] = coerced

    # fix casing/whitespace typos, drop blanks (astype(str) turns NaN into "nan")
    for col in CATEGORICAL_COLUMNS:
        original = df[col].astype(str)
        normalized = original.str.strip().str.upper()
        repaired = int(((normalized != original) & (normalized != "")).sum())
        quality["entries_normalized"] += repaired
        blank = normalized.isin(["", "NAN", "NONE", "NULL"])
        n_blank = int(blank.sum())
        if n_blank:
            quality["exclusion_reasons"][f"blank/unusable {col}"] = n_blank
            bad_mask |= blank
        df[col] = normalized

    result = df["pressure_test_result"].astype(str).str.strip().str.upper()
    invalid_result = ~result.isin(["PASS", "FAIL"])
    if invalid_result.any():
        quality["exclusion_reasons"]["invalid pressure_test_result"] = int(invalid_result.sum())
        bad_mask |= invalid_result
    df["pressure_test_result"] = result
    df["fail"] = (result == "FAIL").astype(int)

    # timestamp is optional, it enables the time-drift (METHOD) check
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().sum() > 0:
            df["elapsed_hours"] = (ts - ts.min()).dt.total_seconds() / 3600.0
        else:
            quality["exclusion_reasons"]["unparseable timestamps (METHOD check skipped)"] = len(df)

    clean = df[~bad_mask].reset_index(drop=True)
    quality["rows_excluded"] = int(bad_mask.sum())
    quality["rows_analyzed"] = len(clean)

    if clean.empty:
        raise IngestError("No analyzable rows remain after validation.")
    return clean, quality
