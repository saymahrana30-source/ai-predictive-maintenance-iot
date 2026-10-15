"""
evaluate.py - Step 5: how good are the predictions on data the model NEVER trained on?

Why not just look at accuracy? Failures are rare (around 5% of hours). A model that always
answers "no failure" would be about 95% accurate and completely useless. So we also use:

  Precision  When the model raises an alarm, how often is a failure really coming?
  Recall     Of all the real upcoming failures, how many did the model catch?
  F1         One number that balances precision and recall (1.0 is perfect)
  ROC-AUC    How well the model ranks risky hours above safe ones
             (0.5 = guessing, 1.0 = perfect)
"""
import json
import math

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src import config


def compute_metrics(y_true, y_pred, y_proba):
    """Return the five metrics as a dictionary, rounded to 4 decimal places."""
    roc_auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else math.nan
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}


def print_metrics_table(all_metrics):
    """Print one row per model so the models can be compared side by side."""
    header = f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}{'roc_auc':>9}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        print(f"{name:<22}{m['accuracy']:>10.3f}{m['precision']:>11.3f}"
              f"{m['recall']:>9.3f}{m['f1']:>8.3f}{m['roc_auc']:>9.3f}")


def save_metrics(all_metrics, path=config.METRICS_PATH):
    """Save the metrics as JSON so they can be shown in the README."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    return path
