
"""One physical-consistency layer, calibrated on normal training inputs only.

The validator accepts exactly the nine process/actual-state features. Labels and
event metadata are deliberately absent from both calibration and evaluation.
Original source-row indices identify consecutive one-minute observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulate import TANK_CAPACITY_L, TIMESTEP_MIN
from train_baseline import FEATURES


QUANTILE = 0.99
STATE_CHECKS = {
    "inlet_flow_state": ("inlet_flow", "inlet_valve_actual", "L/min"),
    "transfer_flow_state": ("transfer_flow", "pump1_actual", "L/min"),
    "transfer_pressure_state": ("pressure", "pump1_actual", "bar"),
    "outlet_flow_state": ("outlet_flow", "pump2_actual", "L/min"),
}
BALANCE_CHECKS = {
    "tank1_balance": ("tank1_level", "inlet_flow", "transfer_flow"),
    "tank2_balance": ("tank2_level", "transfer_flow", "outlet_flow"),
}
RULES = tuple(STATE_CHECKS) + tuple(BALANCE_CHECKS)
RULE_COLUMNS = tuple(f"rule_{rule}" for rule in RULES)


def validate_inputs(inputs: pd.DataFrame) -> None:
    """Reject metadata, missing values and unordered observations at the boundary."""
    if set(inputs.columns) != set(FEATURES) or len(inputs.columns) != len(FEATURES):
        raise ValueError("Consistency inputs must contain exactly the nine sensor/actual-state features.")
    if inputs.empty or not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError("Consistency inputs must be nonempty and finite.")
    if (not pd.api.types.is_integer_dtype(inputs.index.dtype)
            or not inputs.index.is_monotonic_increasing or not inputs.index.is_unique):
        raise ValueError("Retain strictly increasing integer source-row indices.")
    for actuator in FEATURES[6:]:
        if not inputs[actuator].isin([0, 1]).all():
            raise ValueError(f"{actuator} must be a binary actual state.")


def balance_residuals(inputs: pd.DataFrame) -> pd.DataFrame:
    """End-of-minute change uses the CURRENT row's flows, without looking ahead."""
    consecutive = pd.Series(inputs.index, index=inputs.index).diff().eq(1)
    result = pd.DataFrame({"balance_pair_available": consecutive.astype(int)}, index=inputs.index)
    for rule, (level, incoming, outgoing) in BALANCE_CHECKS.items():
        observed = inputs[level].diff().where(consecutive)
        expected = 100.0 * TIMESTEP_MIN / TANK_CAPACITY_L * (inputs[incoming] - inputs[outgoing])
        result[f"{rule}_observed_change"] = observed
        result[f"{rule}_expected_change"] = expected
        result[f"{rule}_residual"] = observed - expected
    return result


def numerical_floor(scale: float) -> float:
    """An eight-ULP round-off allowance; this is not a fitted noise tolerance."""
    return float(8 * np.spacing(max(1.0, abs(scale))))


def calibrate_thresholds(normal_training_inputs: pd.DataFrame) -> pd.DataFrame:
    """Learn ten thresholds from preselected normal training rows, once.

    Keep original indices: differences spanning an excluded abnormal interval
    are NOT normal one-minute balance observations and must not be calibrated.
    """
    validate_inputs(normal_training_inputs)
    records = []
    for rule, (feature, actuator, units) in STATE_CHECKS.items():
        for state in (0, 1):
            sample = normal_training_inputs.loc[normal_training_inputs[actuator].eq(state), feature]
            if sample.empty:
                raise ValueError(f"No normal training observations for {rule}, state={state}.")
            center = float(sample.median())
            residual = sample - center
            raw = float(residual.abs().quantile(QUANTILE))
            floor = numerical_floor(float(sample.abs().max()))
            threshold = max(raw, floor)
            records.append({
                "rule": rule, "feature": feature, "actuator": actuator,
                "actuator_state": state, "training_samples": len(sample),
                "quantile": QUANTILE, "center": center, "absolute_residual_quantile": raw,
                "numerical_floor": floor, "threshold": threshold,
                "lower_allowed": max(0.0, center - threshold),
                "upper_allowed": center + threshold, "units": units,
                "residual_definition": "reported value - state-conditional normal median",
            })
    balances = balance_residuals(normal_training_inputs)
    # Levels are percentages bounded by 100. Eight representable increments at
    # this scale cover subtraction, scaling and CSV float round-trip error.
    floor = numerical_floor(100.0)
    for rule, (level, incoming, outgoing) in BALANCE_CHECKS.items():
        sample = balances[f"{rule}_residual"].dropna()
        if sample.empty:
            raise ValueError("Consecutive normal training pairs are required for water balance.")
        raw = float(sample.abs().quantile(QUANTILE))
        threshold = max(raw, floor)
        records.append({
            "rule": rule, "feature": level, "actuator": "none", "actuator_state": np.nan,
            "training_samples": len(sample), "quantile": QUANTILE, "center": 0.0,
            "absolute_residual_quantile": raw, "numerical_floor": floor, "threshold": threshold,
            "lower_allowed": -threshold, "upper_allowed": threshold,
            "units": "percentage points (one minute)",
            "residual_definition": f"delta({level}) - 0.1 * ({incoming} - {outgoing})",
        })
    return pd.DataFrame(records)


def apply_consistency(inputs: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Apply frozen rules to every row, with no labels, metadata or RF decisions.

    The first sample and any gap lack a previous reading, so only state checks
    apply there. Every subsequent balance uses the preceding REPORTED level,
    including manipulated readings; no truthful history is substituted.
    """
    validate_inputs(inputs)
    if len(thresholds) != 10 or set(thresholds.rule) != set(RULES):
        raise ValueError("Expected the ten calibrated state/balance thresholds.")
    result = balance_residuals(inputs)
    for rule, (feature, actuator, _) in STATE_CHECKS.items():
        rows = thresholds.loc[thresholds.rule.eq(rule)].set_index("actuator_state")
        if len(rows) != 2 or set(rows.index) != {0, 1}:
            raise ValueError(f"Missing or duplicate state thresholds for {rule}.")
        center = inputs[actuator].map(rows.center)
        tolerance = inputs[actuator].map(rows.threshold)
        residual = inputs[feature] - center
        result[f"{rule}_residual"] = residual
        result[f"rule_{rule}"] = residual.abs().gt(tolerance).astype(int)
    for rule in BALANCE_CHECKS:
        rows = thresholds.loc[thresholds.rule.eq(rule)]
        if len(rows) != 1:
            raise ValueError(f"Expected one threshold for {rule}.")
        result[f"rule_{rule}"] = result[f"{rule}_residual"].abs().gt(rows.threshold.iloc[0]).astype(int)
    flags = result[list(RULE_COLUMNS)].to_numpy(dtype=bool)
    result["consistency_rule_count"] = flags.sum(axis=1)
    result["consistency_alert"] = flags.any(axis=1).astype(int)
    result["triggered_rules"] = [
        "|".join(rule for rule, active in zip(RULES, row) if active) for row in flags
    ]
    return result


def hybrid_prediction(random_forest_prediction, consistency_alert) -> np.ndarray:
    forest, alert = np.asarray(random_forest_prediction), np.asarray(consistency_alert)
    if forest.shape != alert.shape or not np.isin(forest, [0, 1]).all() or not np.isin(alert, [0, 1]).all():
        raise ValueError("Hybrid predictions require aligned binary RF predictions and alerts.")
    return np.maximum(forest, alert).astype(int)
