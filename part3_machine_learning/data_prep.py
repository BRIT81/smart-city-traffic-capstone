"""
data_prep.py

Shared data preparation for Part 3 (Machine Learning and AI) of the Smart
City Traffic Intelligence capstone.

Reuses Part 2's cleaning pipeline and feature engineering, then adds the
pieces specific to Part 3: a documented proxy accident-risk label (since no
real accident dataset was provided) and the remaining common feature-set
requirements for Task 1 (a binary holiday flag and a cyclical encoding of
day of week, on top of Part 2's existing hour cyclical encoding).
"""

import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (this file lives in part3_machine_learning/, one level below the
# project root; part2_python/ is a sibling directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PART2_DIR = PROJECT_ROOT / "part2_python"
sys.path.insert(0, str(PART2_DIR))

from pipeline import run_pipeline, DATA_FILE
from feature_engineering import engineer_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# No handlers configured here: this module only calls getLogger(__name__).
# Handler setup happens once, in whichever script's entry point is actually
# run (see logging_config.configure_logging()).
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proxy accident-risk label definitions
# ---------------------------------------------------------------------------
# Reuses the exact same definition as Part 2's is_severe_weather flag, for
# consistency across the whole project.
SEVERE_WEATHER = ["Thunderstorm", "Squall"]

# Weather conditions that impair visibility.
LOW_VISIBILITY_CONDITIONS = ["Fog", "Mist", "Haze", "Smoke"]


def add_holiday_and_dow_features(df):
    """Add a clean binary holiday flag and a cyclical encoding of day of week,
    completing Task 1's common feature-set requirements (Part 2 already
    provides a cyclical encoding of hour)."""
    df = df.copy()

    df["is_holiday"] = df["holiday"].notna().astype(int)
    holiday_count = df["is_holiday"].sum()
    logger.info("Added 'is_holiday' flag; %d of %d rows flagged as a holiday.", holiday_count, len(df))

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    logger.info("Added cyclical day-of-week encoding: dow_sin, dow_cos.")

    return df


def add_risk_label(df):
    """Add is_low_visibility, a 4-bucket risk_congestion_category (distinct
    from Part 2's 3-bucket congestion_category), and the proxy high_risk
    label used throughout Part 3 in place of a real accident dataset."""
    df = df.copy()

    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_CONDITIONS).astype(int)
    low_vis_count = df["is_low_visibility"].sum()
    logger.info(
        "Added 'is_low_visibility' flag (%s); %d of %d rows flagged.",
        LOW_VISIBILITY_CONDITIONS, low_vis_count, len(df),
    )

    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values

    def bucket(v):
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Medium"
        elif v <= q3:
            return "High"
        return "Severe"

    df["risk_congestion_category"] = df["traffic_volume"].apply(bucket)
    counts = df["risk_congestion_category"].value_counts().to_dict()
    logger.info(
        "Created 'risk_congestion_category' (Low <= %.2f, Medium <= %.2f, High <= %.2f, else Severe): %s",
        q1, q2, q3, counts,
    )

    high_congestion = df["risk_congestion_category"].isin(["High", "Severe"])
    risky_weather = df["weather_main"].isin(SEVERE_WEATHER) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (high_congestion & risky_weather).astype(int)

    positive_count = df["high_risk"].sum()
    positive_rate = positive_count / len(df)
    logger.info(
        "Created proxy 'high_risk' label (High/Severe congestion AND severe/low-visibility weather): "
        "%d of %d rows flagged (%.2f%%).",
        positive_count, len(df), positive_rate * 100,
    )
    if positive_rate < 0.01 or positive_rate > 0.5:
        logger.warning(
            "'high_risk' positive rate (%.2f%%) is unusually extreme for a rare-event proxy label; "
            "double-check SEVERE_WEATHER/LOW_VISIBILITY_CONDITIONS definitions.",
            positive_rate * 100,
        )

    return df


def prepare_data(csv_path=DATA_FILE):
    """Run Part 2's full pipeline and feature engineering, then add Part 3's
    proxy risk label and remaining common features. Returns the final,
    model-ready DataFrame shared by every Part 3 task."""
    logger.info("Preparing Part 3 dataset from Part 2's pipeline and feature engineering.")
    clean_df = run_pipeline(csv_path)
    feat_df = engineer_features(clean_df)
    feat_df = add_holiday_and_dow_features(feat_df)
    feat_df = add_risk_label(feat_df)
    logger.info("Part 3 dataset ready. Final shape: %d rows, %d columns.", *feat_df.shape)
    return feat_df


if __name__ == "__main__":
    from logging_config import configure_logging
    configure_logging()
    ml_df = prepare_data()