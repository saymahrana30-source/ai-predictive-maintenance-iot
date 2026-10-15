"""
predict.py - Step 6: use the trained model on new sensor data.

IMPORTANT: prediction must compute features EXACTLY like training did (same window,
same columns, same order). That is why this file reuses add_rolling_features() instead
of writing the maths again. If the two differ, the model gets confused.
"""
import pandas as pd

from src import config
from src.alerts import classify_alert
from src.features import add_rolling_features
from src.preprocess import preprocess


def predict_dataframe(bundle, clean_df):
    """
    Predict a failure probability for EVERY row of a cleaned sensor table.
    clean_df must be sorted by machine and time (preprocess() takes care of this).
    """
    features = add_rolling_features(clean_df, bundle["window_hours"])
    probability = bundle["model"].predict_proba(features[bundle["feature_columns"]])[:, 1]

    result = clean_df.copy()
    result["failure_probability"] = probability.round(4)
    result["prediction"] = (probability >= config.PREDICTION_THRESHOLD).astype(int)
    result["alert_level"] = [classify_alert(p) for p in probability]
    return result


def validate_readings(readings):
    """Check a request before using it. Raises ValueError with a helpful message."""
    required = ["timestamp", *config.SENSOR_COLUMNS]
    if not isinstance(readings, list) or len(readings) < config.WINDOW_HOURS:
        raise ValueError(f"'readings' must be a list of at least {config.WINDOW_HOURS} hourly readings.")
    for i, row in enumerate(readings):
        if not isinstance(row, dict):
            raise ValueError(f"reading {i} must be an object with the sensor values.")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"reading {i} is missing: {', '.join(missing)}")


def predict_latest(bundle, machine_id, readings):
    """
    Predict failure risk for one machine using its most recent readings.
    `readings` is a list of hourly dictionaries, oldest first.
    """
    validate_readings(readings)
    raw = pd.DataFrame(readings)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    for col in config.SENSOR_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")  # bad text becomes missing
    raw["machine_id"] = machine_id
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    clean = preprocess(raw)  # the same cleaning used during training
    if clean[config.SENSOR_COLUMNS].isna().any().any():
        raise ValueError("Not enough valid sensor values to make a prediction.")

    latest = predict_dataframe(bundle, clean).iloc[-1]
    return {
        "machine_id": machine_id,
        "timestamp": latest["timestamp"].isoformat(),
        "failure_probability": float(latest["failure_probability"]),
        "alert_level": latest["alert_level"],
        "prediction": (f"Failure likely within {config.PREDICTION_HORIZON_HOURS} hours"
                       if latest["prediction"] == 1 else "Machine running normally"),
    }
