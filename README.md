# MB DigiCoworker - Automated 4M Root Cause Triage

My submission for the MB DigiCoworker case study. Two parts: a synthetic data
generator for the EV battery enclosure sealant line (Task 1), and a triage
engine that finds the root cause of the failure spike across the 4M dimensions
(Task 2).

## Setup and run

```bash
pip install -r requirements.txt

python data_generator.py                  # writes factory_telemetry.csv
python run_triage.py factory_telemetry.csv
```

This produces `root_cause_report.json` (the required schema) and
`triage_analysis.log`, which has the supporting numbers: data quality counts,
raw fail-rate tables and the fitted model's coefficient table.

Stack: Python 3.11+, pandas, numpy, statsmodels. Everything runs in-memory on
a single node, which fits the centralized on-prem constraint from the case
study. The 15-minute MES/ERP batch cadence would be the natural re-run
interval in production.

## Files

```
config.py            # all tunable numbers for both parts
data_generator.py    # Task 1
run_triage.py        # Task 2 entry point
triage/ingest.py     # load, validate, clean
triage/model.py      # fail-rate tables + the regression
triage/report.py     # JSON report + evidence log
model_walkthrough.ipynb  # notebook stepping through model.py with outputs
```

## Task 1 - the dataset

1000 units over 48 hours (one every ~3 min). Telemetry is drawn from normal
distributions around realistic setpoints (90 psi / 45 C / 180 C). Shifts run
in real 8-hour blocks, and sealant batches are consumed sequentially, ~250
contiguous units per drum, the way material actually flows on a line.

Failure is probabilistic: every unit gets
`P(fail) = sigmoid(baseline + physics deviations + anomaly effect)` and a
weighted coin flip decides PASS/FAIL. The physics terms follow sealing
physics (low pressure = thin bead, low oven temp = undercure, nozzle temp bad
in both directions). There are no hard pass/fail threshold rules anywhere.

In the default scenario BATCH_B102 is contaminated and fails at ~50% against
a ~3.5% baseline, putting the line at ~15% overall - the spike from the case
study. Because a drum is consumed in one contiguous window, the shifts on
duty during that window also show elevated fail rates through no fault of
their own. I didn't hand-wire this decoy; it falls out of sequential
consumption crossing the shift blocks, which is exactly how real shop-floor
confounding happens. A naive analysis blames the shift.

About 2% of the hand-entered shift/batch fields are corrupted (casing typos,
whitespace, blanks) to reflect the legacy MES entry described in the case
study. The engine has to clean these.

Two other scenarios can be injected with `--scenario machine_drift` or
`--scenario operator_shift` (see the table below).

## Task 2 - the engine

The core is a single logistic regression fitted over every factor at once:

```
P(fail) ~ standardized telemetry + one-hot shift + one-hot batch
```

Since all factors are in the model together, each coefficient measures that
factor's effect on failure odds with everything else held constant - which is
exactly the primary-cause vs secondary-symptom question. The winner is the
largest significant risk-raising coefficient. Factors that look bad in the
raw fail-rate tables but aren't confirmed by the model get named as decoys
and exonerated in the summary.

A few decisions worth explaining:

- I picked regression over feature-importance style ML because importance
  ranks predictive power, and a correlated symptom predicts almost as well as
  the cause. Regression coefficients answer the right question, and a plant
  manager can read them.
- Elapsed time is kept out of the main model. Sequential batch consumption
  makes time near-collinear with batch identity, and when I tried including
  it the effect split between the two and destabilized the coefficients.
  Instead, METHOD gets a separate two-factor check (`fail ~ time + winner`):
  if the time trend collapses next to the winner, the apparent drift was just
  the winner's exposure window. If nothing wins at all, a time-only model
  runs last so a genuine process-drift cause can still be caught.
- Categorical levels only qualify as a root cause with a positive
  coefficient. A negative dummy coefficient means "safer than baseline",
  which cannot explain a failure spike.
- Confidence = 0.5 * significance + 0.5 * magnitude, capped at 0.95. The
  magnitude bar differs by feature kind (2.5 for dummies, 1.25 for
  standardized continuous) because a dummy coefficient carries a whole-group
  contrast while a continuous one is per standard deviation.
- If nothing is significant the report honestly says so (`category: NONE`)
  instead of forcing a suspect.

On the sample dataset: MATERIAL / BATCH_B102, confidence 0.95. Its
coefficient (+2.64, p = 7.5e-10) is about 14x odds, while both
suspicious-looking shifts (26.1% and 17.1% raw fail rates) collapse to
non-significance.

## Does it generalize?

Same engine, no changes, three different injected causes:

| scenario | injected cause | verdict | confidence | decoys exonerated |
|---|---|---|---|---|
| material_batch | contaminated BATCH_B102 | MATERIAL / BATCH_B102 | 0.95 | both shifts, time |
| machine_drift | regulator ramps down from hour 30 | MACHINE / dispenser_pressure_psi | 0.77 | OP_SHIFT_C, BATCH_D104, time |
| operator_shift | OP_SHIFT_C workmanship | MAN / OP_SHIFT_C | 0.88 | BATCH_B102, BATCH_D104, time |

```bash
python data_generator.py --scenario machine_drift --output drift.csv
python run_triage.py drift.csv --output drift_report.json --log drift.log
```

## Error handling

Missing file, missing columns or an empty dataset exit with code 1 and a
clear message. Bad values (unparseable numbers, blank categoricals, invalid
test results) get their rows excluded and itemized in the log. Missing
timestamps skip the METHOD check with a note. A non-converging model reports
itself and still delivers the raw tables.

## Known simplifications

- Telemetry values are drawn independently, no sensor autocorrelation.
- The 15-minute MES latency is a design rationale, not simulated lag.
- The confidence weights and normalizers are documented judgment calls, not
  fitted quantities.
