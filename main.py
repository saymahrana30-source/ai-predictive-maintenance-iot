"""
main.py - run the whole project from start to finish.

    python main.py                 use existing simulated data (created on the first run)
    python main.py --resimulate    create brand-new simulated data first

Pipeline:
  1. Simulate IoT sensor data (no hardware needed)
  2. Clean the data
  3. Engineer features and failure labels
  4. Split by time, then train two models
  5. Evaluate on unseen future data
  6. Save the model, metrics and predictions
  7. Draw the charts
"""
import argparse

from sklearn.metrics import confusion_matrix

from src import config
from src.evaluate import compute_metrics, print_metrics_table, save_metrics
from src.features import build_training_table, get_feature_columns
from src.predict import predict_dataframe
from src.preprocess import count_bad_readings, load_raw_data, preprocess
from src.simulate_data import save_sensor_data, simulate_sensor_data
from src.train import load_model_bundle, save_model_bundle, time_split, train_models
from src.visualize import (plot_confusion_matrix, plot_failure_probability,
                           plot_feature_importance, plot_sensor_history)


def run_pipeline(resimulate=False):
    print("=" * 64)
    print("AI-Powered Predictive Maintenance for IoT Devices")
    print("=" * 64)

    # 1. Virtual sensor data
    print("\n[1/7] Virtual IoT sensor data")
    if resimulate or not config.RAW_DATA_PATH.exists():
        save_sensor_data(simulate_sensor_data())
        print("      Simulated new sensor data ->", config.RAW_DATA_PATH.relative_to(config.PROJECT_ROOT))
    raw = load_raw_data()
    print(f"      {len(raw):,} hourly readings from {raw['machine_id'].nunique()} machines")
    print(f"      {int(raw['failure_event'].sum())} real failures happened in the simulation")

    # 2. Cleaning
    print("\n[2/7] Cleaning the data")
    print(f"      {count_bad_readings(raw):,} missing or impossible sensor values found")
    clean = preprocess(raw)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.DATA_PROCESSED_DIR / "clean_sensor_data.csv", index=False)
    print(f"      Repaired. Gaps left: {int(clean[config.SENSOR_COLUMNS].isna().sum().sum())}")
    print("      Cleaned data saved -> data/processed/clean_sensor_data.csv")

    # 3. Features and labels
    print("\n[3/7] Feature engineering")
    table = build_training_table(clean)
    feature_cols = get_feature_columns()
    print(f"      {len(feature_cols)} features per reading")
    print(f"      Hours labelled 'failure within 24h': {int(table['label'].sum()):,} "
          f"({table['label'].mean():.1%} of all hours)")

    # 4. Time-based split and training
    print("\n[4/7] Training (on the past only)")
    train_df, test_df = time_split(table)
    print(f"      Train: {len(train_df):,} rows | Test (future): {len(test_df):,} rows")
    models = train_models(train_df, feature_cols)

    # 5. Evaluation
    print("\n[5/7] Evaluation on unseen future data")
    all_metrics = {}
    for name, model in models.items():
        proba = model.predict_proba(test_df[feature_cols])[:, 1]
        pred = (proba >= config.PREDICTION_THRESHOLD).astype(int)
        all_metrics[name] = compute_metrics(test_df["label"], pred, proba)
    print()
    print_metrics_table(all_metrics)
    save_metrics(all_metrics)

    rf = models["random_forest"]
    save_model_bundle(rf, feature_cols)
    print(f"\n      Production model (Random Forest) saved -> models/{config.MODEL_PATH.name}")

    # 6. Predictions and alerts
    print("\n[6/7] Predictions and alerts")
    bundle = load_model_bundle()  # reload from disk to prove the saved model works
    predictions = predict_dataframe(bundle, clean).loc[test_df.index]
    predictions["label"] = test_df["label"]
    columns = ["timestamp", "machine_id", *config.SENSOR_COLUMNS,
               "label", "failure_probability", "prediction", "alert_level"]
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions[columns].to_csv(config.PREDICTIONS_PATH, index=False)
    counts = predictions["alert_level"].value_counts()
    print("      Alerts on test data: " + ", ".join(f"{k}: {v:,}" for k, v in counts.items()))
    print(f"      Predictions saved -> outputs/{config.PREDICTIONS_PATH.name}")

    # 7. Charts
    print("\n[7/7] Charts")
    config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(predictions["label"], predictions["prediction"])
    busiest_machine = predictions.groupby("machine_id")["label"].sum().idxmax()
    charts = [
        plot_sensor_history(clean, busiest_machine, config.SCREENSHOTS_DIR / "sensor_history.png"),
        plot_failure_probability(predictions, busiest_machine,
                                 config.SCREENSHOTS_DIR / "failure_probability.png"),
        plot_confusion_matrix(cm, config.SCREENSHOTS_DIR / "confusion_matrix.png"),
        plot_feature_importance(rf, feature_cols, config.SCREENSHOTS_DIR / "feature_importance.png"),
    ]
    for chart in charts:
        print("      saved", chart.relative_to(config.PROJECT_ROOT))

    print("\nDone! Next steps:")
    print("  - Look at the screenshots/ folder and outputs/metrics.json")
    print("  - Start the API with:  python -m src.api")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the predictive maintenance pipeline.")
    parser.add_argument("--resimulate", action="store_true",
                        help="create brand-new simulated sensor data first")
    args = parser.parse_args()
    run_pipeline(resimulate=args.resimulate)
