"""
preprocess.py - Step 2: clean the raw sensor data.

Real sensors are messy:
  * readings go missing (loose wire, Wi-Fi drop, ...)
  * sensors send error codes that are physically impossible (e.g. -127 °C)

Cleaning rules:
  1. Impossible values (outside config.VALID_RANGES) are treated as missing.
  2. Missing values are estimated from the neighbouring hours of the SAME machine.
"""
import pandas as pd

from src import config


def load_raw_data(path=config.RAW_DATA_PATH):
    """Read the CSV and put the rows in time order for each machine."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)


def count_bad_readings(df):
    """How many sensor values are missing or impossible? (For the progress report.)"""
    bad = 0
    for col, (low, high) in config.VALID_RANGES.items():
        bad += int((~df[col].between(low, high)).sum())  # NaN counts as "not between"
    return bad


def remove_impossible_values(df):
    """Replace physically impossible readings with NaN so they count as missing."""
    df = df.copy()
    for col, (low, high) in config.VALID_RANGES.items():
        df[col] = df[col].where(df[col].between(low, high))
    return df


def fill_missing_values(df):
    """Estimate missing readings from the neighbouring hours of the same machine."""
    df = df.copy()
    for col in config.SENSOR_COLUMNS:
        df[col] = df.groupby("machine_id")[col].transform(
            lambda s: s.interpolate(limit_direction="both").ffill().bfill()
        )
    return df


def preprocess(df):
    """Full cleaning pipeline: remove impossible values, then fill the gaps."""
    return fill_missing_values(remove_impossible_values(df))
