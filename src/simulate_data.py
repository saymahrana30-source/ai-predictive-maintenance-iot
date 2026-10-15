"""
simulate_data.py - Step 1: create VIRTUAL IoT sensor data.

No real machines? No problem. We simulate a small factory where:
  * 5 machines each have 3 sensors (temperature, vibration, current)
  * every machine reports one reading per hour
  * each machine slowly WEARS OUT before it fails, like a real motor would
  * after a failure the machine is repaired and starts a new healthy life

Because we control the simulation we KNOW exactly when each failure happened.
That is what lets us train the AI model and check how well it works.
"""
import numpy as np
import pandas as pd

from src import config

# ---- How each sensor behaves ----------------------------------------------
DEGRADATION_HOURS = 72     # wear-out starts this many hours before a failure
SUDDEN_FAILURE_SHARE = 0.3 # 3 in 10 failures give NO warning (e.g. a broken wire)

BASELINE = {"temperature": 55.0, "vibration": 2.0, "current": 8.0}            # normal level
BASELINE_SPREAD = {"temperature": 1.0, "vibration": 0.1, "current": 0.2}      # machine-to-machine difference
DEGRADATION_EFFECT = {"temperature": 12.0, "vibration": 3.0, "current": 2.5}  # rise at the moment of failure
NOISE_STD = {"temperature": 1.5, "vibration": 0.35, "current": 0.4}           # random sensor noise
DAILY_SWING = {"temperature": 3.0, "vibration": 0.0, "current": 0.5}          # day/night load cycle


def _schedule_failures(n_hours, rng):
    """Return the hour numbers at which one machine fails."""
    failure_hours = []
    t = 0
    while True:
        t += int(rng.integers(12 * 24, 25 * 24 + 1))  # 12 to 25 days of healthy running
        if t >= n_hours:
            break
        failure_hours.append(t)
        t += 1  # the repair takes one hour
    return np.array(failure_hours, dtype=int)


def _wear_level(hours, failure_hours):
    """
    How worn-out is the machine at each hour?
      0.0 = healthy, 1.0 = failing right now.
    Wear builds up over the 72 hours before each failure. It starts slowly and speeds up
    near the end, so the last few hours show the strongest signal.
    """
    # A sentinel "failure" far in the future means every hour has a next failure to compare with
    upcoming = np.append(failure_hours, 10**9)
    next_index = np.searchsorted(upcoming, hours, side="left")
    hours_until_failure = upcoming[next_index] - hours
    progress = np.clip(1 - hours_until_failure / DEGRADATION_HOURS, 0, 1)
    return progress ** 2


def simulate_machine(machine_id, n_hours, rng, start):
    """Simulate one machine's sensor readings, hour by hour."""
    hours = np.arange(n_hours)
    failure_hours = _schedule_failures(n_hours, rng)

    failure_event = np.zeros(n_hours, dtype=int)
    failure_event[failure_hours] = 1

    # Gradual failures show wear beforehand. Sudden ones do not, so the AI cannot warn about them.
    sudden = rng.random(len(failure_hours)) < SUDDEN_FAILURE_SHARE
    wear = _wear_level(hours, failure_hours[~sudden])
    wear_speed = rng.uniform(0.8, 1.2)                  # some machines wear out faster than others
    daily_cycle = np.sin(2 * np.pi * ((hours % 24) - 8) / 24)

    def sensor_reading(name):
        baseline = BASELINE[name] + rng.normal(0, BASELINE_SPREAD[name])
        return (
            baseline
            + DAILY_SWING[name] * daily_cycle
            + wear_speed * DEGRADATION_EFFECT[name] * wear
            + rng.normal(0, NOISE_STD[name], n_hours)
        )

    temperature = sensor_reading("temperature")
    vibration = sensor_reading("vibration")
    current = sensor_reading("current")

    # Occasional one-hour spikes (for example a sudden load change).
    # They are NOT failures, so they act as false alarms the model must learn to ignore.
    spikes = rng.random(n_hours) < 0.003
    temperature = temperature + spikes * 8.0
    vibration = vibration + spikes * 2.0

    return pd.DataFrame({
        "timestamp": start + pd.to_timedelta(hours, unit="h"),
        "machine_id": machine_id,
        "temperature": np.round(temperature, 2),
        "vibration": np.round(vibration, 2),
        "current": np.round(current, 2),
        "failure_event": failure_event,
    })


def simulate_sensor_data(num_machines=config.NUM_MACHINES,
                         days=config.SIMULATION_DAYS,
                         seed=config.RANDOM_SEED):
    """Simulate every machine and return one table of all readings."""
    rng = np.random.default_rng(seed)
    n_hours = days * 24
    start = pd.Timestamp("2026-01-01 00:00")

    frames = [simulate_machine(f"M{i:02d}", n_hours, rng, start)
              for i in range(1, num_machines + 1)]
    df = pd.concat(frames, ignore_index=True)

    # Realistic imperfections that the cleaning step has to handle:
    glitch = rng.random(len(df)) < 0.002          # sensor error code of -127 °C
    df.loc[glitch, "temperature"] = -127.0
    for col in config.SENSOR_COLUMNS:             # about 1% of readings silently go missing
        df.loc[rng.random(len(df)) < 0.01, col] = np.nan
    return df


def save_sensor_data(df, path=config.RAW_DATA_PATH):
    """Write the simulated readings to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    save_sensor_data(simulate_sensor_data())
    print(f"Simulated sensor data saved to {config.RAW_DATA_PATH}")
