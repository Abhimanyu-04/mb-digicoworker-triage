"""Writes root_cause_report.json (strict schema) and the evidence log."""

import json


def build_report(quality, verdict):
    winner = verdict["winner"]
    if winner is None:
        primary = {
            "category": "NONE",
            "suspect_attribute": "none",
            "confidence_score": 0.0,
            "summary": verdict["summary"],
        }
    else:
        primary = {
            "category": winner.dimension,
            "suspect_attribute": winner.name,
            "confidence_score": round(winner.confidence, 2),
            "summary": verdict["summary"],
        }
    return {
        "total_units_analyzed": quality["rows_loaded"],
        "failure_rate_percentage": round(verdict.get("failure_rate", 0.0) * 100, 1),
        "primary_root_cause": primary,
    }


def write_json(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def write_log(path, quality, verdict):
    lines = ["=" * 78, "MB DigiCoworker - Triage Evidence Log", "=" * 78, ""]

    lines.append("[Data quality]")
    lines.append(f"  rows loaded: {quality['rows_loaded']}   "
                 f"analyzed: {quality['rows_analyzed']}   "
                 f"excluded: {quality['rows_excluded']}")
    lines.append(f"  manual-entry values repaired (casing/whitespace): "
                 f"{quality['entries_normalized']}")
    for reason, n in quality["exclusion_reasons"].items():
        lines.append(f"  excluded - {reason}: {n}")
    lines.append("")

    obs = verdict.get("obs")
    if obs:
        lines.append("[Raw fail-rate tables]")
        lines.append(f"  overall fail rate: {obs['overall_rate']:.1%}")
        for col, table in obs["categorical"].items():
            lines.append(f"  by {col}:")
            for line in str(table).splitlines():
                lines.append(f"    {line}")
        lines.append("  numeric means (fail vs pass):")
        for col, m in obs["numeric"].items():
            lines.append(f"    {col}: {m['mean_fail']:.2f} vs {m['mean_pass']:.2f}")
        lines.append("")

    if verdict.get("coefs") is not None:
        lines.append("[Joint logistic regression]")
        for line in str(verdict["coefs"].round(4)).splitlines():
            lines.append(f"  {line}")
        if verdict.get("decoys"):
            names = ", ".join(d.name for d in verdict["decoys"])
            lines.append(f"  decoys nullified by the model: {names}")
    elif verdict.get("model_note"):
        lines.append(f"[Joint logistic regression] {verdict['model_note']}")
    lines.append("")

    lines.append("[Verdict]")
    lines.append(f"  {verdict['summary']}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
