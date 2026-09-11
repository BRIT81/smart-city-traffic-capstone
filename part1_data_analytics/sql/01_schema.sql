CREATE TABLE IF NOT EXISTS traffic_volume (
    holiday              TEXT,
    temp                 REAL,
    rain_1h              REAL,
    snow_1h              REAL,
    clouds_all           INTEGER,
    weather_main         TEXT,
    weather_description  TEXT,
    date_time            TEXT,
    traffic_volume       INTEGER
);