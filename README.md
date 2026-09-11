# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

Capstone project for the NUS SOC AI, ML and Data Science programme. A Smart City Mobility
Analytics Team scenario: transforming ~48,000 hourly records of westbound I-94 traffic volume
(near Minneapolis–St Paul), together with weather and US federal holiday data, into an
end-to-end traffic intelligence solution.

Dataset: `data/Metro_Interstate_Traffic_Volume.csv` (Metro Interstate Traffic Volume dataset,
2012-10-02 to 2018-09-30, 48,204 hourly rows, 9 columns: `holiday`, `temp`, `rain_1h`,
`snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`, `traffic_volume`).

## Project structure

```
smart-city-traffic-capstone/
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
├── part1_data_analytics/
│   ├── sql/               # SQLite database + .sql query files
│   ├── statistics/        # descriptive stats, correlation, probability analysis
│   └── powerbi/           # Power BI dashboard (.pbix) + insights report
├── part2_python/
│   ├── pipeline.py
│   ├── feature_engineering.py
│   ├── visualizations.py
│   ├── mini_app/
│   ├── figures/
│   └── pipeline.log
└── part3_machine_learning/
    ├── notebooks/
    ├── models/
    ├── mlflow/
    ├── deployment/
    ├── recommendation_system/
    └── monitoring/
```

Each part builds on the previous one: Part 2 reuses the cleaning/insights from Part 1, and
Part 3 reuses the engineered features and pipeline from Part 2.

## Status

- [ ] Part 1 – Data Analytics (SQL, statistics, probability, Power BI)
- [ ] Part 2 – Python (pipeline, feature engineering, visualisation, CLI app)
- [ ] Part 3 – Machine Learning & AI (models, MLOps, recommendation system)

## Tools and technologies

Python (pandas, NumPy, Matplotlib, scikit-learn), SQLite, Power BI Desktop, Git/GitHub,
MLflow, FastAPI/Flask (deployment simulation).

## How to run

Instructions for each part are added as that part is completed (see the section below and
each part's own README once available).

### Part 1 — SQL analysis

```
sqlite3 part1_data_analytics/sql/traffic.db < part1_data_analytics/sql/load_and_analyse.sql
```

## Logging configuration

Documented in `part2_python/README.md` once the pipeline is built (Part 2 requires
`logging.getLogger(__name__)` throughout, console + file handler in the entry point only,
writing to `part2_python/pipeline.log`).

## Assumptions and limitations

- No accident dataset was provided. Part 3's classification task uses a documented proxy
  `high_risk` label derived from traffic-volume quartiles combined with severe/low-visibility
  weather, per the capstone instructions. This is for demonstrating the ML workflow only and
  is not a real accident-risk prediction.
