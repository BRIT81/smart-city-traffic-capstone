"""
traffic_cli.py

Task 4: Mini Traffic Analytics Application for the Smart City Traffic
Intelligence capstone.

A command-line tool for querying the processed traffic dataset. Supports
four commands: query a specific date/time, identify peak-traffic hours,
compare weekday vs. weekend traffic, and recommend low-traffic travel windows.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (mini_app/ is one level below part2_python/, so its parent must be
# added to sys.path before pipeline/feature_engineering can be imported)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PART2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PART2_DIR))

from pipeline import run_pipeline, DATA_FILE
from feature_engineering import engineer_features

LOG_FILE = PART2_DIR / "pipeline.log"

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


# ---------------------------------------------------------------------------
# Query functions (verified in dev_notebook.ipynb, Task 4)
# ---------------------------------------------------------------------------
def query_datetime(df, dt_str):
    """Look up traffic and weather conditions for a specific date/time.
    Raises ValueError if dt_str cannot be parsed as a datetime."""
    target = pd.to_datetime(dt_str, errors="raise")

    match = df[df["date_time"] == target]
    if not match.empty:
        row = match.iloc[0]
        return {
            "date_time": row["date_time"],
            "traffic_volume": row["traffic_volume"],
            "weather_main": row["weather_main"],
            "weather_description": row["weather_description"],
            "temp_kelvin": row["temp"],
            "exact_match": True,
        }

    nearest_idx = (df["date_time"] - target).abs().idxmin()
    nearest_row = df.loc[nearest_idx]
    return {
        "requested_date_time": target,
        "nearest_available": nearest_row["date_time"],
        "traffic_volume": nearest_row["traffic_volume"],
        "weather_main": nearest_row["weather_main"],
        "weather_description": nearest_row["weather_description"],
        "temp_kelvin": nearest_row["temp"],
        "exact_match": False,
    }


def peak_hours(df, top_n=5):
    """Return the top_n hours of day with the highest average traffic volume."""
    return df.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False).head(top_n)


def compare_weekday_weekend(df):
    """Return summary stats comparing weekday vs weekend traffic volume."""
    summary = df.groupby("is_weekend")["traffic_volume"].agg(["mean", "median", "std"])
    summary.index = summary.index.map({0: "Weekday", 1: "Weekend"})
    return summary


def recommend_travel_times(df, top_n=5, start_hour=5, end_hour=22):
    """Return the top_n lowest-traffic hours within [start_hour, end_hour]."""
    window = df[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)]
    return window.groupby("hour")["traffic_volume"].mean().sort_values(ascending=True).head(top_n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="traffic_cli.py",
        description="Query the processed Metro Interstate traffic dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Look up traffic/weather for a specific date/time.")
    query_parser.add_argument("--datetime", required=True, help="e.g. '2018-06-15 08:00:00'")

    peak_parser = subparsers.add_parser("peak-hours", help="Identify the highest-traffic hours of day.")
    peak_parser.add_argument("--top", type=int, default=5, help="Number of hours to show (default: 5).")

    subparsers.add_parser("compare-weekday-weekend", help="Compare weekday vs weekend traffic.")

    recommend_parser = subparsers.add_parser("recommend", help="Recommend low-traffic travel windows (05:00-22:00).")
    recommend_parser.add_argument("--top", type=int, default=5, help="Number of hours to show (default: 5).")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    command_args = {k: v for k, v in vars(args).items() if k != "command"}
    logger.info("Command '%s' invoked with arguments: %s", args.command, command_args)

    clean_df = run_pipeline(DATA_FILE)
    feat_df = engineer_features(clean_df)

    try:
        if args.command == "query":
            result = query_datetime(feat_df, args.datetime)
            print(result)
        elif args.command == "peak-hours":
            result = peak_hours(feat_df, top_n=args.top)
            print(result)
        elif args.command == "compare-weekday-weekend":
            result = compare_weekday_weekend(feat_df)
            print(result)
        elif args.command == "recommend":
            result = recommend_travel_times(feat_df, top_n=args.top)
            print(result)
    except ValueError as e:
        logger.error("Invalid input for command '%s': %s", args.command, e)
        print(f"Error: could not process '{args.command}' — {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()