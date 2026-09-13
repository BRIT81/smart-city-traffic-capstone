"""
visualizations.py

Task 3: Visualise Traffic Patterns for the Smart City Traffic Intelligence capstone.

Produces Matplotlib figures revealing traffic demand by hour, weekday vs. weekend
differences, the overall traffic volume distribution, and traffic's relationship
with weather conditions. All figures are saved under figures/, and each save is
logged for auditability.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from pipeline import run_pipeline, DATA_FILE
from feature_engineering import engineer_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "pipeline.log"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

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


def plot_traffic_by_hour(df, output_dir=FIGURES_DIR):
    """Bar chart of average traffic volume by hour of day."""
    hourly_avg = df.groupby("hour")["traffic_volume"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hourly_avg.index, hourly_avg.values, color="#4C72B0")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Traffic Volume")
    ax.set_title("Average Traffic Volume by Hour of Day")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()

    output_path = output_dir / "traffic_by_hour.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


def plot_weekday_vs_weekend(df, output_dir=FIGURES_DIR):
    """Line chart comparing average traffic by hour for weekdays vs weekends."""
    grouped = df.groupby(["is_weekend", "hour"])["traffic_volume"].mean().unstack(level=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grouped.index, grouped[0], label="Weekday", color="#4C72B0", linewidth=2)
    ax.plot(grouped.index, grouped[1], label="Weekend", color="#DD8452", linewidth=2)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Traffic Volume")
    ax.set_title("Average Traffic Volume by Hour: Weekday vs. Weekend")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "weekday_vs_weekend.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


def plot_traffic_distribution(df, output_dir=FIGURES_DIR):
    """Histogram showing the overall distribution of traffic volume."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["traffic_volume"], bins=40, color="#55A868", edgecolor="white")
    ax.axvline(df["traffic_volume"].mean(), color="#C44E52", linestyle="--", linewidth=2, label="Mean")
    ax.axvline(df["traffic_volume"].median(), color="#4C72B0", linestyle="--", linewidth=2, label="Median")
    ax.set_xlabel("Traffic Volume")
    ax.set_ylabel("Number of Hours")
    ax.set_title("Distribution of Hourly Traffic Volume")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "traffic_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


def plot_traffic_by_weather(df, output_dir=FIGURES_DIR):
    """Boxplot showing the distribution of traffic volume by weather condition."""
    categories = df.groupby("weather_main")["traffic_volume"].median().sort_values(ascending=False).index
    data = [df.loc[df["weather_main"] == cat, "traffic_volume"] for cat in categories]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, tick_labels=categories, showfliers=False)
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Traffic Volume")
    ax.set_title("Traffic Volume Distribution by Weather Condition")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    output_path = output_dir / "traffic_by_weather.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


def generate_all_visualizations(df):
    """Generate and save all Task 3 visualisations."""
    plot_traffic_by_hour(df)
    plot_weekday_vs_weekend(df)
    plot_traffic_distribution(df)
    plot_traffic_by_weather(df)
    logger.info("All Task 3 visualisations generated.")


if __name__ == "__main__":
    clean_df = run_pipeline(DATA_FILE)
    feat_df = engineer_features(clean_df)
    generate_all_visualizations(feat_df)