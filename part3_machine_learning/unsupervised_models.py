"""
unsupervised_models.py

Task 2 (Part 3): Unsupervised Machine Learning.

Two techniques, both built on the Part 3 dataset (see data_prep.py):

1. K-means clustering on hour (cyclical), weather severity, and traffic
   volume, to characterise recurring traffic/weather conditions since no
   real accident dataset exists to cluster against. k=6 was chosen after
   sweeping k=2..8 and comparing inertia (elbow) and silhouette score;
   the two metrics did not point to a single decisive winner (silhouette
   stayed in a narrow 0.33-0.37 band throughout), so k=6 was picked for
   sitting at the inertia elbow with a near-peak silhouette and producing
   an interpretable number of segments. The sweep is still run and logged
   here for anyone auditing that decision.

2. Association rule mining (Apriori) over discretised time of day,
   weather condition, weekday type, and congestion level, filtered to
   rules that predict congestion_level and ranked by lift.

Outputs: the fitted KMeans model and its scaler are saved to
part3_machine_learning/models/ via joblib; the cluster profile and the
congestion-predicting rules are saved as CSVs alongside this script for
direct use in the capstone report.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

from data_prep import prepare_data, SEVERE_WEATHER, LOW_VISIBILITY_CONDITIONS

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

WEATHER_SEVERITY_RANK = {
    "Clear": 0,
    "Clouds": 1,
    "Mist": 2, "Haze": 2, "Smoke": 2, "Fog": 2,
    "Drizzle": 3, "Rain": 3,
    "Snow": 4,
    "Thunderstorm": 5, "Squall": 5,
}

CLUSTER_FEATURES = ["hour_sin", "hour_cos", "weather_severity", "traffic_volume"]
FINAL_K = 6
K_SWEEP_RANGE = range(2, 9)

SEVERE_WEATHER_SET = set(SEVERE_WEATHER)
LOW_VISIBILITY_SET = set(LOW_VISIBILITY_CONDITIONS)
MIN_SUPPORT = 0.005


def add_weather_severity(df):
    df = df.copy()
    df["weather_severity"] = df["weather_main"].map(WEATHER_SEVERITY_RANK)
    unmapped = df["weather_main"][df["weather_severity"].isna()].unique()
    if len(unmapped) > 0:
        logger.warning("Unmapped weather_main values in weather_severity: %s", unmapped)
    logger.info("Added 'weather_severity' (0=Clear to 5=Severe storms).")
    return df


def sweep_k(X_cluster, k_range=K_SWEEP_RANGE):
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster)
        sil = silhouette_score(X_cluster, labels, sample_size=5000, random_state=42)
        logger.debug("k=%d: inertia=%.1f, silhouette=%.4f", k, km.inertia_, sil)


def run_kmeans(df, k=FINAL_K):
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[CLUSTER_FEATURES])

    sweep_k(X_cluster)

    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans_model.fit_predict(X_cluster)

    cluster_profile = df.groupby("cluster").agg(
        n_rows=("cluster", "size"),
        avg_hour=("hour", "mean"),
        avg_weather_severity=("weather_severity", "mean"),
        most_common_weather=("weather_main", lambda s: s.mode()[0]),
        pct_weekend=("is_weekend", "mean"),
        avg_traffic_volume=("traffic_volume", "mean"),
    )
    cluster_profile["pct_of_data"] = (cluster_profile["n_rows"] / len(df) * 100).round(2)

    logger.info("K-means fitted with k=%d.\n%s", k, cluster_profile.round(2).to_string())
    return df, kmeans_model, scaler, cluster_profile


def bucket_time_of_day(hour):
    if hour <= 5:
        return "Night"
    elif hour <= 9:
        return "Morning Rush"
    elif hour <= 15:
        return "Midday"
    elif hour <= 19:
        return "Evening Rush"
    return "Evening"


def bucket_weather(weather_main):
    if weather_main in SEVERE_WEATHER_SET:
        return "Severe"
    elif weather_main in LOW_VISIBILITY_SET:
        return "LowVisibility"
    elif weather_main in ("Rain", "Drizzle", "Snow"):
        return "Precipitation"
    elif weather_main == "Clouds":
        return "Clouds"
    return "Clear"


def discretize_for_arm(df):
    df = df.copy()
    df["time_of_day"] = df["hour"].apply(bucket_time_of_day)
    df["weather_bucket"] = df["weather_main"].apply(bucket_weather)
    df["weekday_type"] = df["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    df["congestion_level"] = df["risk_congestion_category"]
    logger.info("Discretised time_of_day, weather_bucket, weekday_type, congestion_level for association rule mining.")
    return df


def mine_congestion_rules(df, min_support=MIN_SUPPORT):
    items_df = df[["time_of_day", "weather_bucket", "weekday_type", "congestion_level"]]
    transactions = pd.get_dummies(items_df)

    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    logger.info(
        "Apriori: %d frequent itemsets, %d rules (min_support=%.3f, lift>=1.0).",
        len(frequent_itemsets), len(rules), min_support,
    )

    congestion_items = {f"congestion_level_{level}" for level in ["Low", "Medium", "High", "Severe"]}
    congestion_rules = rules[
        rules["consequents"].apply(lambda s: len(s) == 1 and next(iter(s)) in congestion_items)
    ].copy()
    congestion_rules["antecedents"] = congestion_rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    congestion_rules["consequents"] = congestion_rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    congestion_rules = congestion_rules.sort_values("lift", ascending=False).reset_index(drop=True)

    logger.info("%d rules predict a congestion_level, top rule: %s", len(congestion_rules), congestion_rules.iloc[0].to_dict())
    return congestion_rules


def save_outputs(kmeans_model, scaler, cluster_profile, congestion_rules, models_dir=MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(kmeans_model, models_dir / "kmeans_traffic_clusters.joblib")
    joblib.dump(scaler, models_dir / "kmeans_scaler.joblib")
    logger.info("Saved KMeans model and scaler to %s", models_dir)

    cluster_profile_path = SCRIPT_DIR / "task2_cluster_profile.csv"
    rules_path = SCRIPT_DIR / "task2_congestion_rules.csv"
    cluster_profile.to_csv(cluster_profile_path)
    congestion_rules.to_csv(rules_path, index=False)
    logger.info("Saved cluster profile to %s and congestion rules to %s", cluster_profile_path, rules_path)


def run_task2():
    logger.info("Starting Task 2: Unsupervised Machine Learning.")
    ml_df = prepare_data()
    ml_df = add_weather_severity(ml_df)

    clustered_df, kmeans_model, scaler, cluster_profile = run_kmeans(ml_df)
    arm_df = discretize_for_arm(ml_df)
    congestion_rules = mine_congestion_rules(arm_df)

    save_outputs(kmeans_model, scaler, cluster_profile, congestion_rules)

    logger.info("Task 2 complete.")
    print("\nCluster profile:")
    print(cluster_profile.round(2))
    print("\nTop 15 congestion-predicting rules by lift:")
    print(congestion_rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(15).round(4).to_string(index=False))

    return cluster_profile, congestion_rules


if __name__ == "__main__":
    from logging_config import configure_logging
    configure_logging()
    run_task2()