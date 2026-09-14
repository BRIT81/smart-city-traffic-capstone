"""
deployment/app.py

Task 6.3 (Part 3): Deployment Simulation.

A FastAPI mock-up of a deployed traffic volume prediction service, wired
directly to Task 6.1/6.2's MLflow Model Registry rather than to a fixed
joblib file: at startup, this app loads whichever model version currently
carries the "production" alias for "traffic_volume_regressor". Promoting a
new version in model_versioning.py and re-running this app (no code
change) is enough to serve the new version, which is the actual point of
having a registry in the first place.

Input design: the API accepts a raw, human-readable request (a timestamp,
a weather condition, temperature, precipitation, cloud cover, and an
optional holiday flag) rather than the model's 23 already-engineered
feature columns directly, since a real caller of a traffic prediction
service would have raw weather/time data, not the model's internal
one-hot/cyclical encoding. build_feature_vector() below reproduces the
exact same encoding used in training (imported constants and column order
from supervised_models.py/data_prep.py, prototyped and verified cell-by-
cell in the dev notebook first), so the two can never silently drift
apart.

temp_scaled reproduces Part 2's global mean/std z-score (fit once here at
startup over the full prepared dataset, exactly as every other Part 3
script already does) rather than a separately saved scaler object. This is
the same mild data leakage already present throughout the project, not
something introduced by deployment, and is documented as such.

Every prediction served is appended to deployment/prediction_log.csv
(timestamp, request inputs, predicted volume, model version). This is the
data source Task 6.4/6.5's drift monitoring reads to check whether live
traffic into this API is starting to look different from the data the
model was trained on.

Run with:
    cd part3_machine_learning/deployment
    uvicorn app:app --reload

Then test with:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"date_time\": \"2018-06-15T08:00:00\", \"weather_main\": \"Clear\", \"temp\": 295.0, \"rain_1h\": 0.0, \"snow_1h\": 0.0, \"clouds_all\": 20.0, \"is_holiday\": false}"

or open http://127.0.0.1:8000/docs for FastAPI's interactive Swagger UI.
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEPLOYMENT_DIR = Path(__file__).resolve().parent
PART3_DIR = DEPLOYMENT_DIR.parent
sys.path.insert(0, str(PART3_DIR))

# Configured here at module level, not inside `if __name__ == "__main__":`,
# because uvicorn imports this file as a module rather than running it as a
# script — a __main__-gated call would never fire, silently falling back to
# Python's default "last resort" handler (WARNING+ only, unformatted, never
# written to pipeline.log). configure_logging() is a no-op if handlers are
# already attached, so this is also safe if app.py is ever run directly.
from logging_config import configure_logging
configure_logging()

from data_prep import prepare_data
from supervised_models import FEATURE_COLS
from mlflow_tracking import setup_mlflow

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "traffic_volume_regressor"
MODEL_ALIAS = "production"
PREDICTION_LOG_PATH = DEPLOYMENT_DIR / "prediction_log.csv"

WEATHER_CATEGORIES = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
    "Rain", "Smoke", "Snow", "Squall", "Thunderstorm",
]
SEVERE_WEATHER = ["Thunderstorm", "Squall"]
LOW_VISIBILITY_CONDITIONS = ["Fog", "Mist", "Haze", "Smoke"]

app = FastAPI(
    title="Smart City Traffic Volume Prediction API",
    description="Task 6.3 deployment simulation: serves the MLflow-registered production traffic_volume_regressor.",
    version="1.0",
)

# ---------------------------------------------------------------------------
# Loaded once at startup: the registered model and the temp scaling
# parameters, rather than reloading either on every request.
# ---------------------------------------------------------------------------
_state = {}


@app.on_event("startup")
def load_model_and_scaling():
    setup_mlflow()
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    ml_df = prepare_data()
    temp_mean = ml_df["temp"].mean()
    temp_std = ml_df["temp"].std()

    _state["model"] = model
    _state["model_version"] = model_version.version
    _state["model_run_id"] = model_version.run_id
    _state["temp_mean"] = temp_mean
    _state["temp_std"] = temp_std

    logger.info(
        "Loaded '%s' alias '%s' -> registry version %s (run_id=%s). temp scaling: mean=%.4f, std=%.4f",
        REGISTERED_MODEL_NAME, MODEL_ALIAS, model_version.version, model_version.run_id,
        temp_mean, temp_std,
    )

    if not PREDICTION_LOG_PATH.exists():
        with open(PREDICTION_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_timestamp", "input_date_time", "weather_main", "temp",
                "rain_1h", "snow_1h", "clouds_all", "is_holiday",
                "predicted_traffic_volume", "model_version",
            ])
        logger.info("Created prediction log at %s", PREDICTION_LOG_PATH)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    date_time: datetime = Field(..., description="Timestamp to predict traffic volume for, e.g. 2018-06-15T08:00:00")
    weather_main: Literal[
        "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
        "Rain", "Smoke", "Snow", "Squall", "Thunderstorm",
    ] = Field(..., description="Dominant weather condition at that time.")
    temp: float = Field(..., description="Temperature in Kelvin.")
    rain_1h: float = Field(0.0, ge=0, description="Rainfall in the past hour, mm.")
    snow_1h: float = Field(0.0, ge=0, description="Snowfall in the past hour, mm.")
    clouds_all: float = Field(0.0, ge=0, le=100, description="Cloud cover, percent.")
    is_holiday: bool = Field(False, description="Whether this date is a US federal holiday.")


class PredictionResponse(BaseModel):
    predicted_traffic_volume: float
    model_name: str
    model_version: str
    model_alias: str


# ---------------------------------------------------------------------------
# Feature engineering, reproducing training-time logic exactly (verified
# cell-by-cell against a real training row in the dev notebook first).
# ---------------------------------------------------------------------------
def build_feature_vector(request: PredictionRequest, temp_mean: float, temp_std: float) -> pd.DataFrame:
    hour = request.date_time.hour
    day_of_week = request.date_time.weekday()
    is_weekend = int(day_of_week >= 5)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    wx_flags = {f"wx_{cat}": int(request.weather_main == cat) for cat in WEATHER_CATEGORIES}
    is_severe_weather = int(request.weather_main in SEVERE_WEATHER)
    is_low_visibility = int(request.weather_main in LOW_VISIBILITY_CONDITIONS)

    temp_scaled = (request.temp - temp_mean) / temp_std

    row = {
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "is_weekend": is_weekend, "is_holiday": int(request.is_holiday),
        **wx_flags,
        "is_severe_weather": is_severe_weather, "is_low_visibility": is_low_visibility,
        "temp_scaled": temp_scaled, "rain_1h": request.rain_1h,
        "snow_1h": request.snow_1h, "clouds_all": request.clouds_all,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def log_prediction(request: PredictionRequest, predicted_volume: float, model_version: str):
    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), request.date_time.isoformat(), request.weather_main,
            request.temp, request.rain_1h, request.snow_1h, request.clouds_all, request.is_holiday,
            round(predicted_volume, 2), model_version,
        ])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": REGISTERED_MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": _state.get("model_version"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    features = build_feature_vector(request, _state["temp_mean"], _state["temp_std"])
    predicted_volume = float(_state["model"].predict(features)[0])

    log_prediction(request, predicted_volume, _state["model_version"])
    logger.info(
        "Prediction served: %s, %s, temp=%.1fK -> %.1f (model v%s)",
        request.date_time, request.weather_main, request.temp, predicted_volume, _state["model_version"],
    )

    return PredictionResponse(
        predicted_traffic_volume=round(predicted_volume, 1),
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(_state["model_version"]),
        model_alias=MODEL_ALIAS,
    )


if __name__ == "__main__":
    # Logging is already configured above at module import time (needed for
    # the uvicorn CLI path); this direct-execution path is a convenience
    # fallback, e.g. `python app.py`, and does not need to configure it again.
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
