"""
visualize.py - Step 8: charts that show what the system sees and predicts.

The "Agg" backend lets matplotlib save PNG files on any computer, even one with no
screen. Charts are saved to the screenshots/ folder for your README and portfolio.
"""
import matplotlib
matplotlib.use("Agg")  # save to files instead of opening windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402


def plot_sensor_history(clean_df, machine_id, path):
    """Three sensor lines for one machine. Red dashed lines mark the real failures."""
    data = clean_df[clean_df["machine_id"] == machine_id]
    failure_times = data.loc[data["failure_event"] == 1, "timestamp"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, sensor in zip(axes, config.SENSOR_COLUMNS):
        ax.plot(data["timestamp"], data[sensor], linewidth=0.7, color="tab:blue")
        for t in failure_times:
            ax.axvline(t, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel(f"{sensor}\n({config.SENSOR_UNITS[sensor]})")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{machine_id}: sensor readings over time (red dashed line = real failure)")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_failure_probability(predictions, machine_id, path):
    """The model's failure probability over time, with the real failure windows shaded."""
    data = predictions[predictions["machine_id"] == machine_id]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(data["timestamp"], data["failure_probability"], color="tab:blue", linewidth=1,
            label="Model's predicted failure probability")
    ax.axhline(config.CRITICAL_THRESHOLD, color="red", linestyle=":", label="CRITICAL threshold")
    ax.axhline(config.WARNING_THRESHOLD, color="orange", linestyle=":", label="WARNING threshold")
    # Shade the hours where a real failure happened within the next 24 hours
    ax.fill_between(data["timestamp"].to_numpy(), 0, 1,
                    where=(data["label"] == 1).to_numpy(),
                    color="red", alpha=0.12, transform=ax.get_xaxis_transform(),
                    label="Actual: failure within 24 h")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Failure probability")
    ax.set_title(f"{machine_id}: AI failure warning on unseen (test) data")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, path):
    """Show how many predictions were right and wrong, in each category."""
    labels = ["No failure", "Failure"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted by AI")
    ax.set_ylabel("What really happened")
    ax.set_title("Confusion matrix (test data)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_columns, path):
    """Which sensor signals does the Random Forest rely on most?"""
    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot.barh(ax=ax, color="tab:green")
    ax.set_xlabel("Importance (higher = the model relies on it more)")
    ax.set_title("What the AI looks at to predict failures")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
