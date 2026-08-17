"""Task 2: root-cause triage engine.

Usage:
    python run_triage.py [factory_telemetry.csv]
                         [--output root_cause_report.json]
                         [--log triage_analysis.log]
"""

import argparse
import json
from pathlib import Path
import sys

from config import TRIAGE_CONFIG
from triage.ingest import ingest, IngestError
from triage.model import resolve
from triage.report import build_report, write_json, write_log

BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(path_str):
    p = Path(path_str)
    return str(p if p.is_absolute() else BASE_DIR / p)


def main():
    parser = argparse.ArgumentParser(description="Automated 4M root-cause triage.")
    parser.add_argument("input", nargs="?", default="factory_telemetry.csv")
    parser.add_argument("--output", default="root_cause_report.json")
    parser.add_argument("--log", default="triage_analysis.log")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    log_path = _resolve_path(args.log)

    try:
        df, quality = ingest(input_path)
    except IngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    verdict = resolve(df, TRIAGE_CONFIG)
    verdict["failure_rate"] = df["fail"].mean()

    report = build_report(quality, verdict)
    write_json(report, output_path)
    write_log(log_path, quality, verdict)

    print(json.dumps(report, indent=2))
    print(f"\nReport written to  : {output_path}")
    print(f"Evidence log       : {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
