SELECT 'New Years Day 2015 (Jan 1)' AS label, COUNT(*) AS hours_recorded,
       ROUND(MIN(temp),2) AS min_temp_K, ROUND(MAX(temp),2) AS max_temp_K,
       ROUND(AVG(temp),2) AS avg_temp_K, ROUND(AVG(traffic_volume),1) AS avg_traffic_volume
FROM hourly_weather WHERE date(date_time) = '2015-01-01'
UNION ALL
SELECT 'New Years Day 2016 (Jan 1)', COUNT(*), ROUND(MIN(temp),2), ROUND(MAX(temp),2), ROUND(AVG(temp),2), ROUND(AVG(traffic_volume),1)
FROM hourly_weather WHERE date(date_time) = '2016-01-01'
UNION ALL
SELECT 'New Years Day 2017 (Jan 1)', COUNT(*), ROUND(MIN(temp),2), ROUND(MAX(temp),2), ROUND(AVG(temp),2), ROUND(AVG(traffic_volume),1)
FROM hourly_weather WHERE date(date_time) = '2017-01-01'
UNION ALL
SELECT 'Labor Day 2015 (Sep 7)', COUNT(*), ROUND(MIN(temp),2), ROUND(MAX(temp),2), ROUND(AVG(temp),2), ROUND(AVG(traffic_volume),1)
FROM hourly_weather WHERE date(date_time) = '2015-09-07'
UNION ALL
SELECT 'Labor Day 2016 (Sep 5)', COUNT(*), ROUND(MIN(temp),2), ROUND(MAX(temp),2), ROUND(AVG(temp),2), ROUND(AVG(traffic_volume),1)
FROM hourly_weather WHERE date(date_time) = '2016-09-05'
UNION ALL
SELECT 'Labor Day 2017 (Sep 4)', COUNT(*), ROUND(MIN(temp),2), ROUND(MAX(temp),2), ROUND(AVG(temp),2), ROUND(AVG(traffic_volume),1)
FROM hourly_weather WHERE date(date_time) = '2017-09-04';