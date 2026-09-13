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
│   ├── part1_analysis_notes.docx   # written findings for every Part 1 task (1-4)
│   ├── sql/               # traffic.db (SQLite) + numbered .sql query files, Task 1
│   ├── statistics/        # task2_statistics_correlation.xlsx: descriptive stats,
│   │                       # correlation and probability calculations, Tasks 2-3
│   └── powerbi/           # Power BI dashboard (.pbix), Task 4
├── part2_python/
│   ├── pipeline.py              # Task 1: data cleaning pipeline
│   ├── feature_engineering.py   # Task 2: feature engineering (builds on Task 1)
│   ├── visualizations.py        # Task 3: Matplotlib visualisations (builds on Tasks 1-2)
│   ├── logging_config.py        # shared logging setup, see "Logging configuration" below
│   ├── mini_app/
│   │   └── traffic_cli.py       # Task 4: CLI app (builds on Tasks 1-2)
│   ├── figures/                 # saved PNG charts from visualizations.py
│   ├── dev_notebook.ipynb       # interactive prototyping/verification notebook
│   ├── part2_report.docx        # Task 5: 1-2 page methodology and findings report
│   └── pipeline.log             # shared log file for every Part 2 script
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

- [x] Part 1 – Data Analytics (SQL, statistics, probability, Power BI)
- [x] Part 2 – Python (pipeline, feature engineering, visualisation, CLI app)
- [ ] Part 3 – Machine Learning & AI (models, MLOps, recommendation system)

## Tools and technologies

Python (pandas, NumPy, Matplotlib, scikit-learn), SQLite, Power BI Desktop, Git/GitHub,
MLflow, FastAPI/Flask (deployment simulation).

## How to run

Instructions for each part are added as that part is completed (see the section below and
each part's own README once available).

### Part 1 — Data Analytics (SQL, statistics, probability, Power BI)

Task 1, SQL: the CSV was loaded into `part1_data_analytics/sql/traffic.db` using DB Browser
for SQLite (schema in `01_schema.sql`). Each numbered query file answers one sub task and can
be run from DB Browser's Execute SQL tab, or from a terminal, for example:

```
sqlite3 part1_data_analytics/sql/traffic.db < part1_data_analytics/sql/02_annual_trends.sql
sqlite3 part1_data_analytics/sql/traffic.db < part1_data_analytics/sql/03_holiday_temperature.sql
```

Tasks 2 to 3, statistics, correlation and probability: open
`part1_data_analytics/statistics/task2_statistics_correlation.xlsx` in Excel, every result is a
live formula over the dataset in that workbook.

Task 4, Power BI: open `part1_data_analytics/powerbi/task4_traffic_dashboard.pbix` in Power BI
Desktop, the Power Query steps used to clean and prepare the data are saved in the query's
Applied Steps.

The written findings and interpretation for all four tasks are in
`part1_data_analytics/part1_analysis_notes.docx`.

### Part 2 — Python (pipeline, feature engineering, visualisation, CLI app)

All scripts live in `part2_python/` and share one dataset, one cleaning pipeline, and one
log file (`part2_python/pipeline.log`). Each script is independently runnable and rebuilds
whatever it depends on from the previous task, so any of them can be run on its own:

```
cd part2_python
python pipeline.py              # Task 1: load, clean, and validate the raw CSV
python feature_engineering.py   # Task 2: builds on Task 1, adds ML-ready features
python visualizations.py        # Task 3: builds on Tasks 1-2, saves four charts to figures/
```

Task 4, the mini CLI app, lives in `part2_python/mini_app/` and supports four commands, each
rebuilding the cleaned, feature-engineered dataset from Tasks 1-2 before running the query:

```
cd part2_python/mini_app
python traffic_cli.py query --datetime "2018-06-15 08:00:00"
python traffic_cli.py peak-hours --top 5
python traffic_cli.py compare-weekday-weekend
python traffic_cli.py recommend --top 5
```

`part2_python/dev_notebook.ipynb` contains the interactive, cell-by-cell development and
verification of every function before it was consolidated into its final script. Task 5's
1-2 page methodology and findings report is `part2_python/part2_report.docx`.

## Logging configuration

Every module in `part2_python/` (`pipeline.py`, `feature_engineering.py`,
`visualizations.py`, `mini_app/traffic_cli.py`) calls only `logging.getLogger(__name__)` —
none of them configure a handler themselves. Handler setup lives in exactly one place,
`part2_python/logging_config.py`, and is invoked once, from the entry point of whichever
script is actually run (its `__main__` block, or `traffic_cli.py`'s `main()`). Because Python
loggers propagate up to the root logger by default, that single configuration call also
picks up log messages from every module the entry-point script imports — for example,
running `visualizations.py` captures log output from `pipeline.py` and
`feature_engineering.py` as well, all in the order they actually execute. Third-party
loggers (`matplotlib`, `PIL`) are explicitly raised to WARNING so the log stays focused on
this project's own events.

**Where logs are written:** every run appends to one shared file, `part2_python/pipeline.log`
(never overwritten), and simultaneously prints INFO-level and above to the console. The file
captures everything down to DEBUG, so it is the complete audit trail; the console is a
lighter live view.

**Format:** `<timestamp> | <level> | <module name> | <message>`, for example:
`2026-09-13 14:48:34 | INFO     | feature_engineering | Feature engineering completed. Output shape: 40575 rows, 29 columns.`

**What each log level represents in this project:**

- **DEBUG** — fine-grained internal values only useful for troubleshooting, e.g. the mean/std
  used to scale a column, or the quartile thresholds behind a congestion category. Written to
  the file only, not shown on the console.
- **INFO** — normal, expected milestones: data loaded, schema validated, a pipeline stage or
  script completed, a figure saved, a CLI command invoked together with its arguments.
- **WARNING** — unexpected but recoverable data issues the pipeline handles automatically:
  duplicate rows removed, physically impossible sensor readings detected, missing values
  imputed using a monthly median.
- **ERROR** — a problem that prevents the current operation from completing as planned: the
  data pipeline logs a full traceback (`exc_info=True`) and exits via `sys.exit(1)` if any
  cleaning stage fails; the CLI app logs a clear, single-line ERROR message (no traceback) and
  exits if the user supplies invalid input, such as an unparseable date/time.

No `print()` statements are used anywhere in Part 2 for internal progress reporting — the only
`print()` calls are in the CLI app, and only for its actual answer to a query (the requested
traffic/weather record, or a ranked list of hours), since that is end-user-facing output, not
a status message.

## Assumptions and limitations

- No accident dataset was provided. Part 3's classification task uses a documented proxy
  `high_risk` label derived from traffic-volume quartiles combined with severe/low-visibility
  weather, per the capstone instructions. This is for demonstrating the ML workflow only and
  is not a real accident-risk prediction.
