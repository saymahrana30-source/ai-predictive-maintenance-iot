"""
alerts.py - Step 7: turn a probability into an action a maintenance team understands.

  NORMAL    keep running, nothing to do
  WARNING   plan an inspection soon
  CRITICAL  stop the machine and call maintenance now
"""
from src import config


def classify_alert(probability):
    """Map a failure probability (0 to 1) to an alert level."""
    if probability >= config.CRITICAL_THRESHOLD:
        return "CRITICAL"
    if probability >= config.WARNING_THRESHOLD:
        return "WARNING"
    return "NORMAL"
