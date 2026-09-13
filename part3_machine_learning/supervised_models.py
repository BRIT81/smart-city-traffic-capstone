"""
supervised_models.py

Task 1 (Part 3): Supervised Machine Learning Models.

Builds a chronological train/test split over the Part 3 dataset (see
data_prep.py), then fits and evaluates two algorithms for each of two
targets:

- Classification target `high_risk`: logistic regression (linear baseline)
  and a random forest (tree-based ensemble), evaluated with accuracy,
  precision, recall, F1, and ROC AUC.
- Regression target `traffic_volume`: linear regression (linear baseline)
  and a random forest (tree-based ensemble), evaluated with MAE and
  R-squared.

A chronological split, not a random shuffle, is used throughout, since the
data is a genuine hourly time series (2012-10-02 to 2018-09-30) and a
random split would let a model train on rows that come after some of its
own test points, information a real deployed model would never have.

Both random forests are capped at max_depth=15 and min_samples_leaf=10.
Unconstrained trees (scikit-learn's defaults) grow until every leaf is
essentially a single row, which barely changes accuracy but produces a
huge serialized model, the regressor alone reached 882MB before any cap
was applied, and 92MB even after a first, milder cap (max_depth=20,
min_samples_leaf=5). This tighter cap keeps file size well within
GitHub's limits for version control and for the deployment simulation in
a later task, with no meaningful cost to predictive performance.

All four fitted models are saved to part3_machine_learning/models/ via
joblib, so later tasks (SHAP explainability, the deployment simulation)
can reuse them without retraining.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, r2_score,
)

from data_prep import prepare_data

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_holiday",
    "wx_Clear", "wx_Clouds", "wx_Drizzle", "wx_Fog", "wx_Haze", "wx_Mist",
    "wx_Rain", "wx_Smoke", "wx_Snow", "wx_Squall", "wx_Thunderstorm",
    "is_severe_weather", "is_low_visibility",
    "temp_scaled", "rain_1h", "snow_1h", "clouds_all",
]

TEST_FRACTION = 0.2


def chronological_split(df, test_fraction=TEST_FRACTION):
    """Sort by date_time and split at a fixed point in the timeline, not a
    random shuffle, so the test set is always the most recent slice of the
    data and never leaks into training."""
    df = df.sort_values("date_time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df.loc[split_idx, "date_time"]

    train_df = df[df["date_time"] < cutoff_date].copy()
    test_df = df[df["date_time"] >= cutoff_date].copy()

    logger.info(
        "Chronological split at %s: train=%d rows (%s to %s), test=%d rows (%s to %s).",
        cutoff_date,
        len(train_df), train_df["date_time"].min(), train_df["date_time"].max(),
        len(test_df), test_df["date_time"].min(), test_df["date_time"].max(),
    )
    return train_df, test_df


def build_feature_targets(train_df, test_df):
    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]
    y_train_clf, y_test_clf = train_df["high_risk"], test_df["high_risk"]
    y_train_reg, y_test_reg = train_df["traffic_volume"], test_df["traffic_volume"]

    logger.info(
        "Feature matrix built: %d features, %d train rows, %d test rows.",
        len(FEATURE_COLS), len(X_train), len(X_test),
    )
    return X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg


def train_and_evaluate_classifiers(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba),
        }
        results.append(metrics)
        logger.info(
            "high_risk classifier '%s': accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, roc_auc=%.4f",
            name, metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1"], metrics["ROC AUC"],
        )

    results_df = pd.DataFrame(results).set_index("Model")
    return models, results_df


def train_and_evaluate_regressors(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=15, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
        results.append(metrics)
        logger.info(
            "traffic_volume regressor '%s': mae=%.4f, r2=%.4f",
            name, metrics["MAE"], metrics["R2"],
        )

    results_df = pd.DataFrame(results).set_index("Model")
    return models, results_df


def log_top_features(model, feature_cols, label, top_n=8):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.debug(
        "Top %d feature importances for %s:\n%s",
        top_n, label, importances.head(top_n).round(4).to_string(),
    )


def save_models(clf_models, reg_models, models_dir=MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        ("clf", "Logistic Regression"): "logistic_regression_high_risk.joblib",
        ("clf", "Random Forest"): "random_forest_high_risk.joblib",
        ("reg", "Linear Regression"): "linear_regression_traffic_volume.joblib",
        ("reg", "Random Forest"): "random_forest_traffic_volume.joblib",
    }

    for name, model in clf_models.items():
        path = models_dir / filenames[("clf", name)]
        joblib.dump(model, path)
        logger.info("Saved classifier '%s' to %s", name, path)

    for name, model in reg_models.items():
        path = models_dir / filenames[("reg", name)]
        joblib.dump(model, path)
        logger.info("Saved regressor '%s' to %s", name, path)


def run_task1():
    logger.info("Starting Task 1: Supervised Machine Learning Models.")
    ml_df = prepare_data()
    train_df, test_df = chronological_split(ml_df)
    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = build_feature_targets(train_df, test_df)

    clf_models, clf_results = train_and_evaluate_classifiers(X_train, y_train_clf, X_test, y_test_clf)
    reg_models, reg_results = train_and_evaluate_regressors(X_train, y_train_reg, X_test, y_test_reg)

    log_top_features(clf_models["Random Forest"], FEATURE_COLS, "high_risk classifier")
    log_top_features(reg_models["Random Forest"], FEATURE_COLS, "traffic_volume regressor")

    save_models(clf_models, reg_models)

    logger.info("Task 1 complete.")
    print("\nClassification results (high_risk):")
    print(clf_results.round(4))
    print("\nRegression results (traffic_volume):")
    print(reg_results.round(4))

    return clf_results, reg_results


if __name__ == "__main__":
    from logging_config import configure_logging
    configure_logging()
    run_task1()