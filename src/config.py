"""
config.py - every setting in one place.

Want to simulate more machines, predict further ahead, or change where files go?
Change the value HERE and the whole project follows.
"""
from pathlib import Path

# ---- Folders ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

# ---- Files -----------------------------------------------------------------
RAW_DATA_PATH = DATA_RAW_DIR / "iot_sensor_data.csv"
MODEL_PATH = MODELS_DIR / "predictive_maintenance_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"

# ---- Sensors (these names must match the CSV column names) -----------------
SENSOR_COLUMNS = ["temperature", "vibration", "current"]
SENSOR_UNITS = {"temperature": "°C", "vibration": "mm/s", "current": "A"}

# ---- Simulation ------------------------------------------------------------
NUM_MACHINES = 8
SIMULATION_DAYS = 150
RANDOM_SEED = 42            # same seed = same data every time (reproducible)

# ---- Feature engineering ---------------------------------------------------
WINDOW_HOURS = 6                # how many past hours each feature looks back
PREDICTION_HORIZON_HOURS = 24   # the question: "will it fail in the next 24 hours?"

# ---- Training / evaluation -------------------------------------------------
TEST_FRACTION = 0.2             # the newest 20% of the timeline is held out for testing
PREDICTION_THRESHOLD = 0.5      # probability at or above this = "failure predicted"

# ---- Alert levels ----------------------------------------------------------
WARNING_THRESHOLD = 0.4         # plan an inspection
CRITICAL_THRESHOLD = 0.7        # stop the machine and call maintenance

# ---- Data quality: physically possible ranges ------------------------------
VALID_RANGES = {
    "temperature": (0.0, 120.0),   # °C
    "vibration": (0.0, 30.0),      # mm/s
    "current": (0.0, 30.0),        # A
}
