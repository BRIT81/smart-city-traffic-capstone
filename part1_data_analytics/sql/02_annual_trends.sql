DROP VIEW IF EXISTS hourly_traffic;
CREATE VIEW hourly_traffic AS
SELECT DISTINCT date_time, traffic_volume
FROM traffic_volume;

SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS hours_recorded,
    ROUND(100.0 * COUNT(*) /
        CASE WHEN CAST(strftime('%Y', date_time) AS INTEGER) % 4 = 0
             THEN 8784 ELSE 8760 END, 1) AS pct_hours_covered,
    SUM(traffic_volume) AS total_traffic_volume,
    ROUND(AVG(traffic_volume), 1) AS avg_traffic_volume_per_hour
FROM hourly_traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

WITH yearly AS (
    SELECT
        strftime('%Y', date_time) AS year,
        COUNT(*)                  AS hours_recorded,
        SUM(traffic_volume)       AS total_traffic_volume,
        AVG(traffic_volume)       AS avg_traffic_volume
    FROM hourly_traffic
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
)
SELECT
    year,
    hours_recorded,
    total_traffic_volume,
    total_traffic_volume - LAG(total_traffic_volume) OVER (ORDER BY year)                       AS total_change_vs_prev_year,
    ROUND(100.0 * (total_traffic_volume - LAG(total_traffic_volume) OVER (ORDER BY year))
        / LAG(total_traffic_volume) OVER (ORDER BY year), 2)                                     AS total_pct_change_vs_prev_year,
    ROUND(avg_traffic_volume, 1)                                                                 AS avg_traffic_volume,
    ROUND(avg_traffic_volume - LAG(avg_traffic_volume) OVER (ORDER BY year), 1)                  AS avg_change_vs_prev_year,
    ROUND(100.0 * (avg_traffic_volume - LAG(avg_traffic_volume) OVER (ORDER BY year))
        / LAG(avg_traffic_volume) OVER (ORDER BY year), 2)                                       AS avg_pct_change_vs_prev_year
FROM yearly
ORDER BY year;