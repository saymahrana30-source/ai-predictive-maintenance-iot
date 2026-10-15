"""
train.py - Step 4: train the AI models.

We train two models so we can compare them:
  1. Logistic Regression: a simple, fast baseline. A fancier model should beat it.
  2. Random Forest: many decision trees that vote together. Strong on sensor data.

The Random Forest is saved as the production model that the API uses.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def time_split(df, test_fraction=config.TEST_FRACTION):
    """
    Train on the PAST, test on the FUTURE.

    We never shuffle time-series data randomly. That would let the model train on hours
    that happen AFTER the test hours, which is cheating (called "data leakage").
    """
    times = np.sort(df["timestamp"].unique())
    cutoff = times[int(len(times) * (1 - test_fraction))]
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]
    return train, test


def build_models():
    """Create (but do not train yet) the two models."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),  # Logistic Regression works best when inputs are on a similar scale
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",   # failures are rare, so give them more weight
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def train_models(train_df, feature_columns):
    """Train every model on the training data. Returns a dictionary of trained models."""
    models = build_models()
    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    for name, model in models.items():
        print(f"      training {name} ...")
        model.fit(X_train, y_train)
    return models


def save_model_bundle(model, feature_columns, path=config.MODEL_PATH):
    """
    Save the model AND everything needed to use it later: the exact feature order,
    the window size and the prediction horizon. Forgetting any of these is a classic bug.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "window_hours": config.WINDOW_HOURS,
        "horizon_hours": config.PREDICTION_HORIZON_HOURS,
    }
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path=config.MODEL_PATH):
    """Load a model bundle saved by save_model_bundle()."""
    return joblib.load(path)
