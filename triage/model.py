"""Root-cause isolation with one logistic regression over all 4M factors.

Fitting everything together means each coefficient is that factor's effect
with the others held constant - a real cause keeps a big significant
coefficient, a factor that only looked guilty by association collapses.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

NUMERIC_COLUMNS = ["dispenser_pressure_psi", "nozzle_temp_c", "oven_temp_c"]
CATEGORICAL_COLUMNS = ["operator_shift_id", "sealant_batch_id"]

DIMENSION_OF = {
    "operator_shift_id": "MAN",
    "sealant_batch_id": "MATERIAL",
    "dispenser_pressure_psi": "MACHINE",
    "nozzle_temp_c": "MACHINE",
    "oven_temp_c": "MACHINE",
    "elapsed_hours": "METHOD",
}

PHYSICAL_MEANING = {
    "dispenser_pressure_psi": ("low dispenser pressure produces a thin or "
                               "discontinuous sealant bead, leaving micro-leak paths"),
    "nozzle_temp_c": ("off-nominal nozzle temperature degrades sealant viscosity "
                      "and surface wetting"),
    "oven_temp_c": "low oven temperature leaves the sealant undercured and weak",
}


@dataclass
class Feature:
    name: str
    column: str
    kind: str        # 'categorical' | 'continuous'
    dimension: str   # MAN | MACHINE | MATERIAL | METHOD
    coef: float
    p_value: float
    level: str = None
    confidence: float = None


def observe(df):
    """Plain fail-rate tables. No decisions here, just context for the report."""
    overall = df["fail"].mean()
    obs = {"overall_rate": overall, "categorical": {}, "numeric": {}}
    for col in CATEGORICAL_COLUMNS:
        table = df.groupby(col)["fail"].agg(n="count", fails="sum", rate="mean")
        table["lift_vs_line"] = table["rate"] / max(overall, 1e-9)
        obs["categorical"][col] = table.round(4)
    for col in NUMERIC_COLUMNS + (["elapsed_hours"] if "elapsed_hours" in df.columns else []):
        obs["numeric"][col] = {
            "mean_fail": float(df.loc[df["fail"] == 1, col].mean()),
            "mean_pass": float(df.loc[df["fail"] == 0, col].mean()),
        }
    return obs


def _standardize(series):
    std = series.std()
    return (series - series.mean()) / (std if std > 0 else 1.0)


def _build_features(df):
    # elapsed_hours stays out on purpose: sequential batch consumption makes
    # time near-collinear with batch identity, which splits the effect between
    # them and destabilizes the coefficients. METHOD gets method_check instead.
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        X[col] = _standardize(df[col])
    dummies = pd.get_dummies(df[CATEGORICAL_COLUMNS], drop_first=True)
    return pd.concat([X, dummies.astype(float)], axis=1)


def fit(df):
    X = _build_features(df)
    try:
        model = sm.Logit(df["fail"], sm.add_constant(X)).fit(disp=0, maxiter=200)
    except Exception as exc:
        return None, f"model did not converge ({exc})"
    coefs = pd.DataFrame({"coef": model.params, "p_value": model.pvalues}).drop(index="const")
    return coefs, None


def _to_feature(feature_name, row):
    for col in CATEGORICAL_COLUMNS:
        if feature_name.startswith(col + "_"):
            level = feature_name[len(col) + 1:]
            return Feature(name=level, column=col, kind="categorical",
                           dimension=DIMENSION_OF[col], coef=row["coef"],
                           p_value=row["p_value"], level=level)
    name = "process_time_drift" if feature_name == "elapsed_hours" else feature_name
    return Feature(name=name, column=feature_name, kind="continuous",
                   dimension=DIMENSION_OF[feature_name], coef=row["coef"],
                   p_value=row["p_value"])


def _winner_indicator(df, winner):
    if winner.kind == "categorical":
        return (df[winner.column] == winner.level).astype(float)
    return _standardize(df[winner.column])


def method_check(df, winner):
    """Fit fail ~ time + winner. If time collapses next to the winner, the
    apparent drift was just the winner's exposure window."""
    if "elapsed_hours" not in df.columns:
        return None
    X = pd.DataFrame({
        "elapsed_hours": _standardize(df["elapsed_hours"]),
        "winner": _winner_indicator(df, winner),
    }, index=df.index)
    try:
        m = sm.Logit(df["fail"], sm.add_constant(X)).fit(disp=0, maxiter=200)
    except Exception:
        return None
    return {"coef": float(m.params["elapsed_hours"]),
            "p_value": float(m.pvalues["elapsed_hours"])}


def time_only_fit(df):
    """Bare time-trend model, used when nothing in the main model wins."""
    if "elapsed_hours" not in df.columns:
        return None
    X = pd.DataFrame({"elapsed_hours": _standardize(df["elapsed_hours"])}, index=df.index)
    try:
        m = sm.Logit(df["fail"], sm.add_constant(X)).fit(disp=0, maxiter=200)
    except Exception:
        return None
    return Feature(name="process_time_drift", column="elapsed_hours",
                   kind="continuous", dimension="METHOD",
                   coef=float(m.params["elapsed_hours"]),
                   p_value=float(m.pvalues["elapsed_hours"]))


def _score(winner, cfg):
    w = cfg["confidence_weights"]
    sig = min(1.0, -np.log10(max(winner.p_value, 1e-300)) / cfg["significance_log10_cap"])
    effect = min(1.0, abs(winner.coef) / cfg["coef_norm"][winner.kind])
    return min(w["significance"] * sig + w["effect_size"] * effect,
               cfg["confidence_cap"])


def _univariate_rates(obs, feat):
    table = obs["categorical"][feat.column]
    rate_in = float(table.loc[feat.level, "rate"])
    n_in = int(table.loc[feat.level, "n"])
    fails_in = int(table.loc[feat.level, "fails"])
    total_n = int(table["n"].sum())
    total_fails = int(table["fails"].sum())
    rate_out = (total_fails - fails_in) / max(total_n - n_in, 1)
    return rate_in, rate_out


def _build_summary(winner, decoys, time_result, obs, cfg):
    parts = []
    odds = np.exp(abs(winner.coef))
    if winner.kind == "categorical":
        rate_in, rate_out = _univariate_rates(obs, winner)
        parts.append(
            f"{winner.name} is the primary root cause: with every other factor held "
            f"constant in the regression, it carries the only dominant significant "
            f"coefficient ({winner.coef:+.2f}, p = {winner.p_value:.1e}), raising the "
            f"odds of a pressure-test failure about {odds:.0f}x. Units using it failed "
            f"at {rate_in:.1%} versus {rate_out:.1%} for everything else.")
    else:
        m = obs["numeric"].get(winner.column, {})
        direction = "drop" if winner.coef < 0 else "rise"
        physical = PHYSICAL_MEANING.get(winner.column,
                                        "parameter deviation degrades seal integrity")
        parts.append(
            f"{winner.name} is the primary root cause: with every other factor held "
            f"constant in the regression, each one-standard-deviation {direction} "
            f"raises the odds of a pressure-test failure about {odds:.1f}x "
            f"(coefficient {winner.coef:+.2f}, p = {winner.p_value:.1e}). Failing "
            f"units average {m.get('mean_fail', float('nan')):.2f} versus "
            f"{m.get('mean_pass', float('nan')):.2f} for passing units. "
            f"Physically, {physical}.")

    if decoys:
        described = []
        for d in decoys:
            rate_in, _ = _univariate_rates(obs, d)
            described.append(f"{d.name} ({rate_in:.1%} fail rate)")
        parts.append(
            f"Raw fail-rate tables make {', '.join(described)} look guilty, but the "
            f"model does not confirm any of them as a risk factor once {winner.name} "
            f"is accounted for - secondary symptoms, not causes.")

    if time_result is not None and winner.column != "elapsed_hours":
        if time_result["p_value"] >= cfg["significance_alpha"]:
            parts.append(
                f"Modeled alongside {winner.name}, the time trend collapses to "
                f"non-significance (p = {time_result['p_value']:.2f}), so the apparent "
                f"drift was the primary cause's exposure window (METHOD ruled out).")
        else:
            parts.append(
                f"Caution: an independent time trend survives alongside the primary "
                f"cause (p = {time_result['p_value']:.1e}) - a process/method drift "
                f"may also be present and is flagged for review.")

    return " ".join(parts)


def resolve(df, cfg):
    obs = observe(df)
    n_fails = int(df["fail"].sum())

    if n_fails == 0 or n_fails == len(df):
        return {"winner": None, "features": [], "obs": obs, "coefs": None,
                "model_note": None,
                "summary": ("No usable failure contrast exists (all units "
                            f"{'passed' if n_fails == 0 else 'failed'}), nothing "
                            "for the model to isolate.")}

    coefs, note = fit(df)
    if coefs is None:
        return {"winner": None, "features": [], "obs": obs, "coefs": None,
                "model_note": note,
                "summary": f"No verdict: {note}. Raw fail-rate tables are in the "
                           f"evidence log for manual review."}

    features = [_to_feature(name, row) for name, row in coefs.iterrows()]
    alpha = cfg["significance_alpha"]

    # must be significant, and categorical levels must raise risk - a negative
    # dummy coef means "safer than baseline", which can't cause a spike
    eligible = [f for f in features
                if f.p_value < alpha and (f.kind == "continuous" or f.coef > 0)]

    if not eligible:
        time_winner = time_only_fit(df)
        if time_winner is not None and time_winner.p_value < alpha:
            time_winner.confidence = _score(time_winner, cfg)
            return {"winner": time_winner, "features": features, "obs": obs,
                    "coefs": coefs, "model_note": None, "decoys": [],
                    "time_check": None,
                    "summary": ("No shift, batch or telemetry factor explains the "
                                "failures, but failure odds trend strongly with time "
                                f"(coefficient {time_winner.coef:+.2f}, p = "
                                f"{time_winner.p_value:.1e}), consistent with a "
                                "process/recipe change during the window (METHOD).")}
        return {"winner": None, "features": features, "obs": obs, "coefs": coefs,
                "model_note": None, "time_check": None,
                "summary": ("No statistically significant root cause was isolated: "
                            "the failure pattern is consistent with normal process "
                            "variation across all 4M dimensions.")}

    winner = max(eligible, key=lambda f: abs(f.coef))
    winner.confidence = _score(winner, cfg)

    # decoys: elevated in the raw tables but not confirmed by the model.
    # "not (significant and positive)" on purpose - a pure decoy's coef usually
    # goes negative once the real cause is in the model
    decoys = []
    for f in features:
        if f.kind != "categorical" or f is winner:
            continue
        if f.p_value < alpha and f.coef > 0:
            continue
        rate_in, _ = _univariate_rates(obs, f)
        if rate_in > obs["overall_rate"]:
            decoys.append(f)

    time_result = method_check(df, winner)
    summary = _build_summary(winner, decoys, time_result, obs, cfg)

    return {"winner": winner, "features": features, "obs": obs, "coefs": coefs,
            "model_note": None, "decoys": decoys, "time_check": time_result,
            "summary": summary}
