"""All tunable numbers for the generator and the triage engine."""

GENERATOR_CONFIG = {
    "n_units": 1000,
    "duration_hours": 48,
    "start_time": "2026-08-15 06:00:00",
    "random_seed": 42,

    "machine_params": {
        "dispenser_pressure_psi": {"setpoint": 90.0, "std": 3.0},
        "nozzle_temp_c": {"setpoint": 45.0, "std": 1.5},
        "oven_temp_c": {"setpoint": 180.0, "std": 4.0},
    },

    "shifts": {
        "ids": ["OP_SHIFT_A", "OP_SHIFT_B", "OP_SHIFT_C"],
        "block_hours": 8,
    },

    "batches": {
        "ids": ["BATCH_A101", "BATCH_B102", "BATCH_C103", "BATCH_D104"],
    },

    # P(fail) = sigmoid(baseline + physics deviations + anomaly effect)
    "failure_model": {
        "baseline_logit": -3.7,  # ~2.4% nominal fail rate
        "weights": {
            "dispenser_pressure_psi": 0.12,  # per psi below setpoint
            "nozzle_temp_c": 0.10,           # per degC off setpoint, either way
            "oven_temp_c": 0.05,             # per degC below setpoint
        },
    },

    "anomaly": {
        "scenario": "material_batch",  # material_batch | machine_drift | operator_shift
        "material_batch": {
            "bad_batch": "BATCH_B102",
            "effect_logit": 3.4,
        },
        "machine_drift": {
            "param": "dispenser_pressure_psi",
            "drift_start_hour": 30.0,
            "drop_at_end": 24.0,
        },
        "operator_shift": {
            "bad_shift": "OP_SHIFT_C",
            "effect_logit": 2.6,
        },
    },

    # shift/batch ids are hand-entered by supervisors, some come in dirty
    "data_quality": {
        "entry_error_rate": 0.02,
        "blank_fraction": 0.15,
    },
}

TRIAGE_CONFIG = {
    "significance_alpha": 0.01,
    "confidence_weights": {"significance": 0.5, "effect_size": 0.5},
    "significance_log10_cap": 10.0,
    # dummy coefs carry a whole-group contrast, continuous coefs are per-sigma
    "coef_norm": {"categorical": 2.5, "continuous": 1.25},
    "confidence_cap": 0.95,
}
