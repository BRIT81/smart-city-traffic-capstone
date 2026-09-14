"""
deep_learning_explainability.py

Task 3 (Part 3): Deep Learning with Explainability.

Implements an LSTM for next-hour traffic volume (demand) prediction, as
recommended by the brief given the data's naturally sequential structure.
Each input is a 24-hour lookback window; any window that crosses a gap in
the dataset's hourly coverage (see the exploratory gap analysis in the
notebook, roughly 23% of hours are missing overall) is excluded rather
than silently treated as consecutive.

For explainability, SHAP is applied to Task 1's random forest regressor
(same target, same problem) rather than to the LSTM directly. SHAP's
deep-learning explainers are version-fragile with Keras/TensorFlow, and
the brief explicitly permits applying the explainability method to a
comparable tree-based or linear model when the chosen model is difficult
to explain directly, as documented here.

Outputs: the trained LSTM and its two scalers (fit on the training period
only, never on test-period data) are saved to
part3_machine_learning/models/; the SHAP bar chart, beeswarm plot, and an
hour-by-hour SHAP contribution table are saved to
part3_machine_learning/figures/ and part3_machine_learning/ respectively.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import shap

from data_prep import prepare_data
from unsupervised_models import WEATHER_SEVERITY_RANK
from supervised_models import FEATURE_COLS

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
FIGURES_DIR = SCRIPT_DIR / "figures"

LOOKBACK = 24
SEQ_FEATURES = ["traffic_volume", "hour_sin", "hour_cos", "is_weekend", "weather_severity"]
TEST_FRACTION = 0.2

# Task 1's random forest regressor, reused here for the explainability
# requirement rather than the LSTM itself.
RF_REGRESSOR_PATH = MODELS_DIR / "random_forest_traffic_volume.joblib"


def build_gap_aware_windows(df, lookback=LOOKBACK):
    """Build (X, y, target_dates) from every window of `lookback` hours
    that is truly consecutive. A window's total time span must equal
    exactly `lookback` hours; since the data has at most one row per hour
    and is sorted ascending, any span larger than that means a gap sits
    somewhere inside the window, so it is excluded rather than treated as
    a valid sequence."""
    df = df.sort_values("date_time").reset_index(drop=True)
    dt = df["date_time"].values

    span = dt[lookback:] - dt[:-lookback]
    valid_starts = np.where(span == np.timedelta64(lookback, "h"))[0]

    feature_matrix = df[SEQ_FEATURES].values
    target_vector = df["traffic_volume"].values

    X_raw = np.stack([feature_matrix[s : s + lookback] for s in valid_starts])
    y_raw = target_vector[valid_starts + lookback]
    target_dates = df["date_time"].values[valid_starts + lookback]

    logger.info(
        "Built %d gap-respecting %d-hour windows out of %d candidates (%.1f%% valid).",
        len(X_raw), lookback, len(span), 100 * len(X_raw) / len(span),
    )
    return df, X_raw, y_raw, target_dates


def chronological_split_windows(df, X_raw, y_raw, target_dates, test_fraction=TEST_FRACTION):
    """Same 80% chronological cutoff used throughout Part 3, applied here
    by each window's target timestamp, so train/test period boundaries
    mean the same thing for every model in this project."""
    split_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df.loc[split_idx, "date_time"]

    train_mask = target_dates < np.datetime64(cutoff_date)
    test_mask = ~train_mask

    logger.info(
        "Chronological split at %s: %d train windows, %d test windows.",
        cutoff_date, train_mask.sum(), test_mask.sum(),
    )
    return (
        X_raw[train_mask], X_raw[test_mask],
        y_raw[train_mask], y_raw[test_mask],
        cutoff_date,
    )


def fit_scalers(df, cutoff_date):
    """Scalers are fit only on training-period rows, never on test-period
    data, consistent with the leakage-avoidance principle applied
    throughout this project."""
    train_row_mask = df["date_time"] < cutoff_date
    scaler_volume = StandardScaler().fit(df.loc[train_row_mask, ["traffic_volume"]].values)
    scaler_severity = StandardScaler().fit(df.loc[train_row_mask, ["weather_severity"]].values)

    logger.info(
        "Scalers fit on training period only: traffic_volume mean=%.1f std=%.1f, "
        "weather_severity mean=%.3f std=%.3f",
        scaler_volume.mean_[0], scaler_volume.scale_[0],
        scaler_severity.mean_[0], scaler_severity.scale_[0],
    )
    return scaler_volume, scaler_severity


def scale_sequences(X_raw, y_raw, scaler_volume, scaler_severity):
    X_scaled = X_raw.copy().astype(float)
    n_samples, n_steps, _ = X_scaled.shape

    X_scaled[:, :, 0] = scaler_volume.transform(X_raw[:, :, 0].reshape(-1, 1)).reshape(n_samples, n_steps)
    X_scaled[:, :, 4] = scaler_severity.transform(X_raw[:, :, 4].reshape(-1, 1)).reshape(n_samples, n_steps)

    y_scaled = scaler_volume.transform(y_raw.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled


def build_lstm(input_shape):
    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(model, X_train, y_train):
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )
    n_epochs_run = len(history.history["loss"])
    logger.info(
        "LSTM training stopped after %d epochs. Final val_loss=%.4f, val_mae=%.4f.",
        n_epochs_run, history.history["val_loss"][-1], history.history["val_mae"][-1],
    )
    return history


def evaluate_lstm(model, X_test, y_test, scaler_volume):
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_volume.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_volume.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info("LSTM traffic_volume regressor: MAE=%.2f, R2=%.4f", mae, r2)
    return mae, r2


def build_tabular_test_set(df, cutoff_date):
    """Rebuilds Task 1's tabular test set fresh, for use with the reloaded
    random forest regressor in the SHAP explainability step."""
    test_df_tab = df[df["date_time"] >= cutoff_date].copy()
    return test_df_tab[FEATURE_COLS]


def save_lstm_artifacts(model, scaler_volume, scaler_severity, models_dir=MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "lstm_traffic_volume.keras")
    joblib.dump(scaler_volume, models_dir / "lstm_scaler_volume.joblib")
    joblib.dump(scaler_severity, models_dir / "lstm_scaler_severity.joblib")
    logger.info("Saved LSTM model and scalers to %s", models_dir)


def run_task3():
    logger.info("Starting Task 3: Deep Learning with Explainability.")
    ml_df = prepare_data()
    ml_df["weather_severity"] = ml_df["weather_main"].map(WEATHER_SEVERITY_RANK)

    gap_df, X_raw, y_raw, target_dates = build_gap_aware_windows(ml_df)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, cutoff_date = chronological_split_windows(
        gap_df, X_raw, y_raw, target_dates
    )

    scaler_volume, scaler_severity = fit_scalers(gap_df, cutoff_date)
    X_train, y_train = scale_sequences(X_train_raw, y_train_raw, scaler_volume, scaler_severity)
    X_test, y_test = scale_sequences(X_test_raw, y_test_raw, scaler_volume, scaler_severity)

    lstm_model = build_lstm(input_shape=(LOOKBACK, len(SEQ_FEATURES)))
    train_lstm(lstm_model, X_train, y_train)
    lstm_mae, lstm_r2 = evaluate_lstm(lstm_model, X_test, y_test, scaler_volume)

    save_lstm_artifacts(lstm_model, scaler_volume, scaler_severity)

    X_test_tab = build_tabular_test_set(gap_df, cutoff_date)
    rf_reg_loaded = joblib.load(RF_REGRESSOR_PATH)
    logger.info("Reloaded random forest regressor from %s for SHAP explainability.", RF_REGRESSOR_PATH)

    explainer = shap.TreeExplainer(rf_reg_loaded)
    shap_values = explainer.shap_values(X_test_tab)
    logger.info("Computed SHAP values: shape %s", shap_values.shape)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X_test_tab, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_traffic_volume.png", dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_test_tab, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm_traffic_volume.png", dpi=150)
    plt.close()
    logger.info("Saved SHAP bar and beeswarm plots to %s", FIGURES_DIR)

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    logger.info("Top 5 features by mean |SHAP value|:\n%s", mean_abs_shap.head(5).round(1).to_string())

    hour_sin_idx = FEATURE_COLS.index("hour_sin")
    hour_cos_idx = FEATURE_COLS.index("hour_cos")
    time_shap_contribution = shap_values[:, hour_sin_idx] + shap_values[:, hour_cos_idx]

    hour_effect_df = pd.DataFrame({
        "hour": gap_df.loc[X_test_tab.index, "hour"].values,
        "time_shap_contribution": time_shap_contribution,
    }).groupby("hour")["time_shap_contribution"].mean()

    hour_effect_path = SCRIPT_DIR / "task3_hour_shap_effect.csv"
    hour_effect_df.to_csv(hour_effect_path)
    logger.info("Saved hour-by-hour SHAP time contribution to %s", hour_effect_path)

    logger.info("Task 3 complete.")
    print(f"\nLSTM  — MAE: {lstm_mae:.2f}   R2: {lstm_r2:.4f}")
    print(f"Random Forest (Task 1) — MAE: 268.78   R2: 0.9411")
    print(f"\nTop 8 features by mean |SHAP value|:")
    print(mean_abs_shap.head(8).round(1))
    print(f"\nHour-by-hour SHAP time-of-day contribution:")
    print(hour_effect_df.round(1))

    return lstm_mae, lstm_r2, mean_abs_shap, hour_effect_df


if __name__ == "__main__":
    # Re-assert this script's own directory at the front of sys.path: importing
    # data_prep above inserts part2_python ahead of it (so pipeline.py and
    # feature_engineering.py can be found), which would otherwise cause this
    # bare import to resolve to part2_python's same-named logging_config.py
    # instead of this file's own, silently redirecting the log file there.
    sys.path.insert(0, str(SCRIPT_DIR))
    from logging_config import configure_logging
    configure_logging()
    run_task3()