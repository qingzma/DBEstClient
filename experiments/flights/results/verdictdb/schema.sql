CREATE TABLE flights (
YEAR_DATE int,
UNIQUE_CARRIER char(100),
ORIGIN char(100),
ORIGIN_STATE_ABR char(2),
DEST char(100),
DEST_STATE_ABR char(2),
DEP_DELAY double,
TAXI_OUT double,
TAXI_IN double,
ARR_DELAY double,
AIR_TIME double,
DISTANCE double)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/data/flights/'
tblproperties ("skip.header.line.count"="1");


CREATE TABLE IF NOT EXISTS fly.flights ( YEAR_DATE int, UNIQUE_CARRIER char(100), ORIGIN char(100), ORIGIN_STATE_ABR char(2), DEST char(100), DEST_STATE_ABR char(2), DEP_DELAY double, TAXI_OUT double, TAXI_IN double, ARR_DELAY double, AIR_TIME double, DISTANCE double) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LOCATION '/data/flights/' tblproperties ("skip.header.line.count"="1")