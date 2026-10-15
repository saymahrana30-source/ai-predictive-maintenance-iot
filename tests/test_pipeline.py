"""
Tests check that the building blocks behave correctly.
Run them from the project folder with:   python -m pytest -q
"""
import numpy as np
import pandas as pd

from src import config
from src.alerts import classify_alert
from src.features import add_failure_label, add_rolling_features, get_feature_columns
from src.preprocess import preprocess
from src.simulate_data import simulate_sensor_data
from src.train import time_split


def _one_machine(n_hours, failure_hours, machine_id="M01", temperature=50.0):
    """A tiny hand-made machine so we know the correct answers in advance."""
    failure = np.zeros(n_hours, dtype=int)
    failure[failure_hours] = 1
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h"),
        "machine_id": machine_id,
        "temperature": temperature,
        "vibration": 2.0,
        "current": 8.0,
        "failure_event": failure,
    })


def test_simulation_has_expected_columns_and_failures():
    df = simulate_sensor_data(num_machines=2, days=30)
    expected = {"timestamp", "machine_id", "failure_event", *config.SENSOR_COLUMNS}
    assert expected <= set(df.columns)
    assert df["machine_id"].nunique() == 2
    assert df["failure_event"].sum() > 0


def test_label_is_one_only_within_the_horizon():
    # Failure at hour 20. With a 5-hour horizon, hours 15-19 should be labelled 1.
    out = add_failure_label(_one_machine(n_hours=40, failure_hours=[20]), horizon=5)
    assert out.loc[15:19, "label"].eq(1).all()
    assert out.loc[20, "label"] == 0   # the failure has already started, so it is not "in the future"
    assert out.loc[14, "label"] == 0
    assert 39 not in out.index         # the last hours cannot look 5 hours ahead, so they are dropped


def test_labels_never_cross_between_machines():
    a = _one_machine(30, [28], "M01")
    b = _one_machine(30, [], "M02")      # M02 never fails
    out = add_failure_label(pd.concat([a, b], ignore_index=True), horizon=5)
    assert out.loc[out["machine_id"] == "M02", "label"].sum() == 0


def test_rolling_features_only_use_their_own_machine():
    a = _one_machine(10, [], "M01", temperature=50.0)
    b = _one_machine(10, [], "M02", temperature=100.0)
    out = add_rolling_features(pd.concat([a, b], ignore_index=True))
    m01 = out.loc[out["machine_id"] == "M01", "temperature_mean_6h"]
    assert np.allclose(m01, 50.0)        # M02's 100 degrees must not leak into M01
    assert set(get_feature_columns()) <= set(out.columns)


def test_alert_levels_follow_the_thresholds():
    assert classify_alert(0.10) == "NORMAL"
    assert classify_alert(0.50) == "WARNING"
    assert classify_alert(0.90) == "CRITICAL"


def test_preprocess_repairs_bad_and_missing_values():
    df = _one_machine(10, [])
    df.loc[3, "temperature"] = -127.0     # sensor error code
    df.loc[5, "vibration"] = np.nan       # dropped reading
    clean = preprocess(df)
    assert clean[config.SENSOR_COLUMNS].notna().all().all()
    assert clean["temperature"].between(*config.VALID_RANGES["temperature"]).all()
    assert clean.loc[3, "temperature"] == 50.0   # estimated from the neighbouring hours
    assert clean.loc[5, "vibration"] == 2.0


def test_time_split_is_chronological():
    df = simulate_sensor_data(num_machines=2, days=20)
    train, test = time_split(df)
    assert len(train) > 0 and len(test) > 0
    assert train["timestamp"].max() < test["timestamp"].min()   # the future is never used for training
