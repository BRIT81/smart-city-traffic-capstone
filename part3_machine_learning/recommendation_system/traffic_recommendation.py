"""
traffic_recommendation.py

Task 5 (Part 3): Traffic Recommendation System.

Objective: turn the trained models and historical patterns from Tasks 1-3 into
a practical, plain-language travel-timing recommender. Since this dataset
covers a single corridor (Minneapolis/St Paul I-94) rather than a road
network, the system recommends when to travel rather than which route to
take -- there is no alternate-route data to recommend against.

Approach: the Task 1 random forest regressor is reused to simulate expected
traffic volume across all 24 hours of the day, for a given day type (weekday
or weekend) and, optionally, a weather condition. This is a scenario
simulation, not a lookup: every feature other than hour (and its cyclical
encoding) is held at a representative value, so the system can answer
questions about combinations that may be sparse in the raw history. Where a
weather condition is not specified, predictions are aggregated across every
observed weather category, weighted by how often each occurs historically,
to give a realistic "typical day" recommendation.

Design history worth recording, since two earlier approaches both produced
misleading results before this one:

1. Searching for the lowest average volume across a bounded "reasonable
   hours" range simply found wherever that search happened to stop, since
   traffic declines close to monotonically from the evening peak into the
   night. This produced a "recommended" window of 10 PM to 11 PM that was
   really just an arbitrary search-boundary artifact, not a meaningful low.
2. Requiring a genuine local minimum (lower than both immediate neighbours)
   fixed that, correctly recovering the expected 10 to 11 AM midday trough
   on weekdays -- but broke under snow conditions, where the same trough
   survives yet flattens into a wider, wigglier plateau rather than a sharp
   dip, so no single hour cleared a fixed prominence bar.
3. The final approach anchors the search to the day's known rush-hour
   structure instead of testing individual hours in isolation: it finds the
   morning peak (max in hours 5-11) and evening peak (max in hours 12-20),
   then searches for the minimum strictly between them. This is robust to
   internal plateau structure, since it takes a plain minimum over a
   well-defined interval. A remaining edge case (a fixed peak-search window
   can itself manufacture a false "peak" when traffic is still climbing
   past it, as happened for weekend snow) is caught with an explicit
   validity check: a genuine valley must be lower than BOTH flanking peaks,
   not just assumed from the window boundaries. When no genuine valley is
   found (weekends, which have a single midday peak rather than two rush
   hours), the system falls back to reporting the lowest point in the
   reasonable-hours range before or after that single peak, honestly framed
   as a directional trend rather than a trough.

Finding worth noting for the write-up: the recommended window's location is
stable across clear, snow, and thunderstorm conditions (10-11 AM on
weekdays in every case), while the magnitude of the benefit varies
(29-31% below the daily peak for most conditions, dropping to 26% for
snow, since snow itself dampens the rush-hour peak the recommendation is
designed to avoid). This is consistent with the dominant role of hour of
day established across Tasks 1, 2 and 3 (feature importance, clustering
and association rules, and SHAP): commute timing is structurally driven by
work schedules, not by weather.

Caveat: Smoke (~0.04% of all hours) and Squall (a handful of rows across
six years) are extremely rare in this dataset. Their simulated profiles and
recommendations are valid outputs of this pipeline but rest on very little
historical support, and should be read as illustrative rather than
statistically robust.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PART3_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PART3_DIR))

from data_prep import prepare_data, SEVERE_WEATHER, LOW_VISIBILITY_CONDITIONS
from supervised_models import FEATURE_COLS

logger = logging.getLogger(__name__)

MODELS_DIR = PART3_DIR / "models"
RF_REGRESSOR_PATH = MODELS_DIR / "random_forest_traffic_volume.joblib"
OUTPUT_PATH = SCRIPT_DIR / "task5_recommendations.csv"

WEATHER_CATEGORIES = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
    "Rain", "Smoke", "Snow", "Squall", "Thunderstorm",
]

# Reasonable travel hours: excludes the deep overnight minimum (around 2-4 AM),
# which is technically the lowest-traffic period of the day but not a
# practical travel recommendation for most journeys.
EARLIEST_HOUR = 7
LATEST_HOUR = 22


def build_weather_profile(ml_df):
    """Historical frequency and representative continuous conditions (temp,
    rain, snow, cloud cover) for each weather category, used so simulated
    rows reflect realistic combinations (e.g. Snow days actually having
    snowfall) rather than generic dataset-wide averages."""
    weather_frequency = {cat: ml_df[f"wx_{cat}"].mean() for cat in WEATHER_CATEGORIES}
    weather_profile = (
        ml_df.groupby("weather_main")[["temp_scaled", "rain_1h", "snow_1h", "clouds_all"]]
        .mean()
    )
    logger.info(
        "Built weather frequency and representative-condition lookup for %d categories.",
        len(WEATHER_CATEGORIES),
    )
    return weather_frequency, weather_profile


def build_simulation_rows(day_type, weather_profile, weather_condition=None):
    """Build one row per (weather category, hour) combination, varying only
    hour and its cyclical encoding; every other feature is held at a
    representative value for the given day type and weather category."""
    if day_type == "weekday":
        is_weekend, representative_dow = 0, 2  # Wednesday: typical midweek day
    elif day_type == "weekend":
        is_weekend, representative_dow = 1, 5  # Saturday: typical weekend day
    else:
        raise ValueError("day_type must be 'weekday' or 'weekend'")

    dow_sin = np.sin(2 * np.pi * representative_dow / 7)
    dow_cos = np.cos(2 * np.pi * representative_dow / 7)
    categories = [weather_condition] if weather_condition else WEATHER_CATEGORIES

    rows = []
    for cat in categories:
        conditions = weather_profile.loc[cat]
        for hour in range(24):
            row = {col: 0 for col in FEATURE_COLS}
            row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            row["dow_sin"] = dow_sin
            row["dow_cos"] = dow_cos
            row["is_weekend"] = is_weekend
            row["is_holiday"] = 0
            row[f"wx_{cat}"] = 1
            row["is_severe_weather"] = int(cat in SEVERE_WEATHER)
            row["is_low_visibility"] = int(cat in LOW_VISIBILITY_CONDITIONS)
            row["temp_scaled"] = conditions["temp_scaled"]
            row["rain_1h"] = conditions["rain_1h"]
            row["snow_1h"] = conditions["snow_1h"]
            row["clouds_all"] = conditions["clouds_all"]
            rows.append({"category": cat, "hour": hour, **row})
    return pd.DataFrame(rows)


def simulate_hourly_profile(day_type, rf_regressor, weather_frequency, weather_profile, weather_condition=None):
    """Predicted traffic volume for each of the 24 hours, for a given day
    type. If weather_condition is given, returns that category's profile
    directly; otherwise aggregates every category's profile, weighted by
    historical frequency, into a single "typical day" profile."""
    sim_rows = build_simulation_rows(day_type, weather_profile, weather_condition)
    sim_rows["predicted_volume"] = rf_regressor.predict(sim_rows[FEATURE_COLS])

    if weather_condition:
        return sim_rows.set_index("hour")["predicted_volume"]

    sim_rows["weight"] = sim_rows["category"].map(weather_frequency)
    sim_rows["weighted_pred"] = sim_rows["predicted_volume"] * sim_rows["weight"]
    return sim_rows.groupby("hour")["weighted_pred"].sum()


def format_hour(h):
    """Format an hour (0-23) as a 12-hour clock string, e.g. 14 -> '2:00 PM'."""
    h = h % 24
    period = "AM" if h < 12 else "PM"
    display = h % 12
    display = 12 if display == 0 else display
    return f"{display}:00 {period}"


def find_best_window(profile, window_size=1, earliest_hour=EARLIEST_HOUR, latest_hour=LATEST_HOUR):
    """Identify the recommended travel window using the day's rush-hour
    structure rather than point-by-point local-minimum detection, which is
    fragile against flat or wiggly valleys. The morning peak is the max
    within hours 5-11 and the evening peak the max within hours 12-20, both
    structural features of commute timing already established in Tasks 1
    and 3. If the two peaks are meaningfully separated AND the lowest point
    between them is genuinely below both (not just below the search window's
    boundary), the recommendation is that midday trough. Otherwise -- a
    single midday peak rather than two rush hours, as on weekends, or a
    fixed peak-search window mistaking a still-rising slope for a peak --
    the fallback compares the lowest point before the true peak against the
    lowest point after it, honestly framed as a directional trend."""
    morning_peak_hour = profile.loc[5:11].idxmax()
    evening_peak_hour = profile.loc[12:20].idxmax()

    is_trough = False
    best_hour = None

    if evening_peak_hour - morning_peak_hour >= 2:
        valley_start = max(morning_peak_hour + 1, earliest_hour)
        valley_end = min(evening_peak_hour - 1, latest_hour)
        if valley_start <= valley_end:
            candidate_hour = profile.loc[valley_start:valley_end].idxmin()
            candidate_val = profile[candidate_hour]
            if candidate_val < profile[morning_peak_hour] and candidate_val < profile[evening_peak_hour]:
                best_hour = candidate_hour
                is_trough = True

    if not is_trough:
        true_peak_hour = (
            morning_peak_hour if profile[morning_peak_hour] >= profile[evening_peak_hour] else evening_peak_hour
        )
        candidates = {}
        if true_peak_hour - 1 >= earliest_hour:
            candidates["pre"] = profile.loc[earliest_hour:true_peak_hour - 1].idxmin()
        if latest_hour >= true_peak_hour + 1:
            candidates["post"] = profile.loc[true_peak_hour + 1:latest_hour].idxmin()
        best_hour = min(candidates.values(), key=lambda h: profile[h])

    half = (window_size - 1) // 2
    start = max(earliest_hour, best_hour - half)
    end = min(latest_hour + 1, start + window_size)
    start = end - window_size
    avg_volume = profile.loc[start:end - 1].mean()
    return start, end, avg_volume, is_trough


def recommend_window(day_type, rf_regressor, weather_frequency, weather_profile, weather_condition=None,
                      window_size=1, earliest_hour=EARLIEST_HOUR, latest_hour=LATEST_HOUR):
    """Return the structured recommendation for one scenario: the recommended
    hour range, predicted and peak volume, and whether it is a genuine
    trough or a directional-trend fallback."""
    profile = simulate_hourly_profile(day_type, rf_regressor, weather_frequency, weather_profile, weather_condition)
    start, end, avg_volume, is_trough = find_best_window(profile, window_size, earliest_hour, latest_hour)
    peak_volume = profile.max()
    return {
        "day_type": day_type,
        "weather_condition": weather_condition,
        "start": start,
        "end": end,
        "avg_volume": avg_volume,
        "peak_volume": peak_volume,
        "pct_below_peak": (1 - avg_volume / peak_volume) * 100,
        "is_trough": is_trough,
    }


def format_recommendation_text(rec):
    """Render a structured recommendation (from recommend_window) as a
    plain-language sentence, in the format given by the Task 5 brief."""
    day_phrase = "weekday" if rec["day_type"] == "weekday" else "weekend"
    weather_phrase = f" during {rec['weather_condition'].lower()} conditions" if rec["weather_condition"] else ""
    start_str, end_str = format_hour(rec["start"]), format_hour(rec["end"])

    if rec["is_trough"]:
        return (
            f"For a {day_phrase} journey{weather_phrase}, consider travelling between "
            f"{start_str} and {end_str}, when historical traffic volumes dip to a local low "
            f"between the morning and evening rush periods (predicted volume approximately "
            f"{rec['avg_volume']:.0f}, {rec['pct_below_peak']:.0f}% below the daily peak of "
            f"approximately {rec['peak_volume']:.0f})."
        )
    return (
        f"For a {day_phrase} journey{weather_phrase}, there is no significant midday dip between "
        f"rush periods; the lowest point within typical travel hours is around {start_str} to "
        f"{end_str} (predicted volume approximately {rec['avg_volume']:.0f}, "
        f"{rec['pct_below_peak']:.0f}% below the daily peak of approximately {rec['peak_volume']:.0f})."
    )


def generate_recommendation(day_type, rf_regressor, weather_frequency, weather_profile,
                             weather_condition=None, **kwargs):
    """Convenience wrapper: build a structured recommendation and render it
    as plain-language text in one call."""
    rec = recommend_window(day_type, rf_regressor, weather_frequency, weather_profile, weather_condition, **kwargs)
    return format_recommendation_text(rec)


def build_recommendation_table(rf_regressor, weather_frequency, weather_profile):
    """Build the complete recommendation table across every day type and
    weather category combination, for reporting and as a saved artifact."""
    results = []
    for day_type in ["weekday", "weekend"]:
        for weather in [None] + WEATHER_CATEGORIES:
            rec = recommend_window(day_type, rf_regressor, weather_frequency, weather_profile, weather)
            results.append({
                "day_type": rec["day_type"],
                "weather_condition": rec["weather_condition"] if rec["weather_condition"] else "any (typical mix)",
                "recommended_start": format_hour(rec["start"]),
                "recommended_end": format_hour(rec["end"]),
                "predicted_volume": round(rec["avg_volume"], 1),
                "daily_peak_volume": round(rec["peak_volume"], 1),
                "pct_below_peak": round(rec["pct_below_peak"], 1),
                "is_genuine_trough": rec["is_trough"],
                "recommendation_text": format_recommendation_text(rec),
            })
    return pd.DataFrame(results)


def run_task5():
    logger.info("Starting Task 5: Traffic Recommendation System.")

    ml_df = prepare_data()
    ml_df = ml_df.sort_values("date_time").reset_index(drop=True)

    rf_regressor = joblib.load(RF_REGRESSOR_PATH)
    logger.info("Loaded random forest regressor from %s.", RF_REGRESSOR_PATH)

    weather_frequency, weather_profile = build_weather_profile(ml_df)

    recommendations_df = build_recommendation_table(rf_regressor, weather_frequency, weather_profile)
    recommendations_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved recommendation table to %s (%d rows).", OUTPUT_PATH, len(recommendations_df))

    for day_type in ["weekday", "weekend"]:
        message = generate_recommendation(day_type, rf_regressor, weather_frequency, weather_profile)
        logger.info("Example recommendation (%s): %s", day_type, message)

    logger.info("Task 5 complete.")
    print(recommendations_df.drop(columns=["recommendation_text"]).to_string(index=False))

    return recommendations_df


if __name__ == "__main__":
    # Re-assert this file's own directory (part3_machine_learning/, not this
    # script's recommendation_system/ subfolder) at the front of sys.path:
    # importing data_prep above inserts part2_python ahead of it (so
    # pipeline.py and feature_engineering.py can be found), which would
    # otherwise cause this bare import to resolve to part2_python's
    # same-named logging_config.py, silently redirecting the log file there.
    sys.path.insert(0, str(PART3_DIR))
    from logging_config import configure_logging
    configure_logging()
    run_task5()
