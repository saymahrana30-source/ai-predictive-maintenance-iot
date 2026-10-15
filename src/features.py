"""
features.py - Step 3: feature engineering and labels.

One temperature reading tells us very little. But "the average temperature over the
last 6 hours is climbing and vibration is getting less stable" is an early-warning sign.
Summaries of recent history like that are called FEATURES.

This file also creates the LABEL, the answer the model learns to predict:
    label = 1  if the machine fails within the next 24 hours
    label = 0  otherwise
"""
import pandas as pd

from src import config


def add_rolling_features(df, window=config.WINDOW_HOURS):
    """
    For every sensor, add three features that summarise the last `window` hours:
      <sensor>_mean_<window>h    average level                 (smooths out noise)
      <sensor>_std_<window>h     how jumpy the readings are    (instability)
      <sensor>_change_<window>h  how far it moved in `window` h (trend)
    Each machine is handled on its own, so one machine's history never leaks into another.
    """
    df = df.copy()
    for sensor in config.SENSOR_COLUMNS:
        df[f"{sensor}_mean_{window}h"] = df.groupby("machine_id")[sensor].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{sensor}_std_{window}h"] = df.groupby("machine_id")[sensor].transform(
            lambda s: s.rolling(window, min_periods=2).std()
        ).fillna(0.0)
        df[f"{sensor}_change_{window}h"] = df.groupby("machine_id")[sensor].transform(
            lambda s: s.diff(window)
        ).fillna(0.0)
    return df


def get_feature_columns(window=config.WINDOW_HOURS):
    """The exact columns the model uses. Their order matters, so it is saved with the model."""
    columns = []
    for sensor in config.SENSOR_COLUMNS:
        columns += [sensor,
                    f"{sensor}_mean_{window}h",
                    f"{sensor}_std_{window}h",
                    f"{sensor}_change_{window}h"]
    return columns


def add_failure_label(df, horizon=config.PREDICTION_HORIZON_HOURS):
    """
    Add the 'label' column: 1 if a failure happens in the NEXT `horizon` hours, else 0.

    Method: for every hour we look 1, 2, ... 24 hours ahead with shift(-k).
    If any of those future hours has a failure, the label is 1.
    """
    df = df.copy()
    labels = []
    for _, machine in df.groupby("machine_id", sort=False):
        future_checks = [machine["failure_event"].shift(-k) for k in range(1, horizon + 1)]
        labels.append(pd.concat(future_checks, axis=1).max(axis=1))
    df["label"] = pd.concat(labels)

    # The last `horizon` hours of each machine cannot look that far ahead, so drop them
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def build_training_table(clean_df):
    """Features plus labels, ready for training."""
    return add_failure_label(add_rolling_features(clean_df))
