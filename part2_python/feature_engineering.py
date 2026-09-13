"""
feature_engineering.py

Task 2: Feature Engineering for the Smart City Traffic Intelligence capstone.

Builds ML-ready features (time-based, weather-based, scaled numeric, and a
data-driven congestion category) on top of the cleaned dataset produced by
pipeline.py.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import run_pipeline, DATA_FILE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def add_time_features(df):
    """Create time-based features: hour, day of week, weekend indicator, and a
    cyclical (sine/cosine) encoding of the hour."""
    df = df.copy()

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    logger.info("Added time features: hour, day_of_week, is_weekend, hour_sin, hour_cos.")
    return df


def encode_weather_features(df):
    """One-hot encode weather_main and add a derived severe-weather indicator."""
    df = df.copy()

    weather_dummies = pd.get_dummies(df["weather_main"], prefix="wx", dtype=int)
    df = pd.concat([df, weather_dummies], axis=1)
    logger.info("One-hot encoded 'weather_main' into %d columns (prefix 'wx_').", weather_dummies.shape[1])

    severe_conditions = ["Thunderstorm", "Squall"]
    df["is_severe_weather"] = df["weather_main"].isin(severe_conditions).astype(int)
    severe_count = df["is_severe_weather"].sum()
    logger.info(
        "Added 'is_severe_weather' indicator (%s); %d of %d rows flagged.",
        severe_conditions, severe_count, len(df),
    )

    return df


def scale_numeric_features(df, columns=("temp", "traffic_volume")):
    """Create standardised (z-score) versions of selected continuous variables."""
    df = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        logger.debug("'%s' scaling parameters: mean=%.4f, std=%.4f.", col, mean, std)
        df[f"{col}_scaled"] = (df[col] - mean) / std
    logger.info("Created scaled versions of: %s.", list(columns))
    return df


def create_congestion_category(df, column="traffic_volume"):
    """Create a data-driven congestion category using quartile thresholds.

    Logic: Low = at or below the 25th percentile (Q1), High = above the 75th
    percentile (Q3), Medium = everything in between. Thresholds are derived
    from the data's own distribution rather than a fixed, arbitrary cutoff.
    """
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    logger.debug("Congestion category thresholds for '%s': Q1=%.2f, Q3=%.2f.", column, q1, q3)

    conditions = [df[column] <= q1, df[column] > q3]
    choices = ["Low", "High"]
    df["congestion_category"] = np.select(conditions, choices, default="Medium")

    counts = df["congestion_category"].value_counts().to_dict()
    logger.info(
        "Created 'congestion_category' (Low <= %.2f, Medium, High > %.2f): %s",
        q1, q3, counts,
    )
    return df


def engineer_features(df):
    """Run the full Task 2 feature engineering sequence, logging shape before and after."""
    logger.info("Feature engineering started. Input shape: %d rows, %d columns.", *df.shape)
    df = add_time_features(df)
    df = encode_weather_features(df)
    df = scale_numeric_features(df)
    df = create_congestion_category(df)
    logger.info("Feature engineering completed. Output shape: %d rows, %d columns.", *df.shape)
    return df


if __name__ == "__main__":
    clean_df = run_pipeline(DATA_FILE)
    feat_df = engineer_features(clean_df)