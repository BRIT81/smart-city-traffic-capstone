"""
pipeline.py

Task 1: Data Pipeline Construction for the Smart City Traffic Intelligence capstone.

Loads the raw Metro Interstate Traffic Volume CSV, validates its schema, and applies
a series of cleaning and validation steps (standardising categorical values, parsing
and validating date/time, removing duplicate rows, and detecting/imputing physically
impossible sensor readings), all under an auditable logging trail.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# No handlers are configured here: this module only calls getLogger(__name__).
# Handler setup happens once, in whichever script's __main__ block is actually
# run (see logging_config.configure_logging()), and propagates up to catch
# every module's log messages via the root logger.
logger = logging.getLogger(__name__)


EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]


def load_data(csv_path):
    """Load the raw traffic CSV into a DataFrame, with error handling for common failure modes."""
    logger.info("Loading data from %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error("Data file not found at %s", csv_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    else:
        logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
        return df


def validate_schema(df, expected_columns=EXPECTED_COLUMNS):
    """Validate that all expected columns are present before any further processing."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        logger.error("Schema validation failed. Missing columns: %s", missing)
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Schema validation passed: all %d expected columns are present.", len(expected_columns))
    return True


def standardize_categoricals(df):
    """Standardise inconsistent categorical values before further processing."""
    df = df.copy()

    non_holiday_count = df["holiday"].isna().sum()
    logger.info(
        "'holiday' already parsed as missing (NaN) for %d non-holiday rows "
        "(pandas recognised the source CSV's literal 'None' text automatically).",
        non_holiday_count,
    )

    for col in ["weather_main", "weather_description"]:
        stripped = df[col].str.strip()
        changed = (stripped != df[col]).sum()
        if changed > 0:
            logger.warning("Stripped whitespace from %d rows in '%s'.", changed, col)
        df[col] = stripped

    before_unique = df["weather_description"].nunique()
    lowered = df["weather_description"].str.lower()
    changed = (lowered != df["weather_description"]).sum()
    if changed > 0:
        logger.warning(
            "Standardised casing to lowercase in %d rows of 'weather_description' "
            "(merging case-variant duplicates such as 'Sky is Clear' / 'sky is clear').",
            changed,
        )
    df["weather_description"] = lowered
    logger.info(
        "'weather_description' now has %d unique values (was %d before standardisation).",
        df["weather_description"].nunique(), before_unique,
    )

    return df


def parse_and_validate_datetime(df, column="date_time"):
    """Parse the date/time column to a proper datetime dtype and validate the result."""
    df = df.copy()
    original_dtype = df[column].dtype

    parsed = pd.to_datetime(df[column], errors="coerce")

    failed_mask = parsed.isna() & df[column].notna()
    failed_count = failed_mask.sum()
    if failed_count > 0:
        logger.warning(
            "Failed to parse %d values in '%s' as valid datetimes; these rows now have a missing timestamp.",
            failed_count, column,
        )
    else:
        logger.info(
            "All %d values in '%s' parsed successfully as datetimes (was dtype '%s').",
            len(df), column, original_dtype,
        )

    df[column] = parsed
    logger.info("'%s' range: %s to %s.", column, df[column].min(), df[column].max())

    return df


def remove_duplicates(df, timestamp_col="date_time", value_col="traffic_volume"):
    """Identify and remove duplicate rows, including duplicate-timestamp entries."""
    df = df.copy()

    exact_dupe_mask = df.duplicated()
    exact_dupe_count = exact_dupe_mask.sum()
    if exact_dupe_count > 0:
        logger.warning("Removed %d exact duplicate rows.", exact_dupe_count)
        df = df[~exact_dupe_mask]
    else:
        logger.info("No exact duplicate rows found.")

    dupe_ts_mask = df.duplicated(subset=timestamp_col, keep=False)
    affected_timestamps = df.loc[dupe_ts_mask, timestamp_col].nunique()

    if affected_timestamps > 0:
        inconsistent_groups = 0
        for ts, group in df.loc[dupe_ts_mask].groupby(timestamp_col):
            if group[value_col].nunique() > 1:
                inconsistent_groups += 1
                logger.warning(
                    "Duplicate timestamp %s has inconsistent '%s' values %s; keeping the first row anyway.",
                    ts, value_col, group[value_col].tolist(),
                )

        rows_before = len(df)
        df = df.drop_duplicates(subset=timestamp_col, keep="first")
        rows_removed = rows_before - len(df)

        logger.warning(
            "Collapsed %d duplicate-timestamp hours (%d rows removed) down to one row per hour; "
            "%d of these hours had inconsistent traffic_volume across duplicates.",
            affected_timestamps, rows_removed, inconsistent_groups,
        )
    else:
        logger.info("No duplicate timestamps found.")

    return df


def detect_and_impute_outliers(df):
    """Detect physically impossible sensor readings and impute them using each
    calendar month's own median, rather than a single global average."""
    df = df.copy()
    df["month"] = df["date_time"].dt.month

    temp_invalid_mask = df["temp"] <= 0
    temp_invalid_count = temp_invalid_mask.sum()
    if temp_invalid_count > 0:
        logger.warning(
            "Found %d rows with physically impossible 'temp' readings (<= 0 Kelvin).",
            temp_invalid_count,
        )
        df.loc[temp_invalid_mask, "temp"] = np.nan

    rain_invalid_mask = df["rain_1h"] > 1000
    rain_invalid_count = rain_invalid_mask.sum()
    if rain_invalid_count > 0:
        logger.warning(
            "Found %d rows with physically implausible 'rain_1h' readings (> 1000mm/hour): %s",
            rain_invalid_count, df.loc[rain_invalid_mask, "rain_1h"].tolist(),
        )
        df.loc[rain_invalid_mask, "rain_1h"] = np.nan

    for column in ["temp", "rain_1h"]:
        for month in sorted(df["month"].unique()):
            month_mask = df["month"] == month
            missing_mask = month_mask & df[column].isna()
            missing_count = missing_mask.sum()

            if missing_count > 0:
                month_median = df.loc[month_mask, column].median()
                df.loc[missing_mask, column] = month_median
                logger.warning(
                    "Imputed %d missing '%s' value(s) in month %d using that month's median (%.2f).",
                    missing_count, column, month, month_median,
                )

    df = df.drop(columns="month")
    return df


def run_pipeline(csv_path):
    """Run the full Task 1 cleaning pipeline end-to-end, with a top-level safety net
    that logs a full error trace and exits gracefully if any stage fails."""
    try:
        df = load_data(csv_path)
        validate_schema(df)
        df = standardize_categoricals(df)
        df = parse_and_validate_datetime(df)
        df = remove_duplicates(df)
        df = detect_and_impute_outliers(df)
    except Exception:
        logger.error("Pipeline failed and could not complete.", exc_info=True)
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully. Final shape: %d rows, %d columns.", *df.shape)
        return df


if __name__ == "__main__":
    from logging_config import configure_logging
    configure_logging()
    run_pipeline(DATA_FILE)