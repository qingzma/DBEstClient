  
!--SELECT AVG(dep_delay) FROM flights WHERE origin='ATL';
!--SELECT AVG(distance) FROM flights WHERE unique_carrier='TW';
!--SELECT unique_carrier, COUNT(*) FROM flights WHERE origin_state_abr='LA' GROUP BY unique_carrier;
!--SELECT unique_carrier, COUNT(*) FROM flights WHERE origin_state_abr='LA' AND  dest_state_abr='CA' GROUP BY unique_carrier;
!--SELECT year_date, COUNT(*) FROM flights WHERE origin_state_abr='LA' AND dest='JFK' GROUP BY year_date;
!--SELECT year_date, SUM(distance) FROM flights WHERE unique_carrier='9E' GROUP BY year_date;
!--SELECT origin_state_abr, SUM(air_time) FROM flights WHERE dest='HPN' GROUP BY origin_state_abr;
!--SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE year_date=2005 AND origin='PHX' GROUP BY unique_carrier;
!--SELECT dest_state_abr, COUNT(*) FROM flights WHERE distance>2500 GROUP BY dest_state_abr;
!--SELECT unique_carrier, COUNT(*) FROM flights WHERE air_time>1000 AND dep_delay>1500 GROUP BY unique_carrier;
!--SELECT year_date, SUM(arr_delay*dep_delay) FROM flights WHERE origin_state_abr = 'CA' AND dest_state_abr = 'HI' GROUP BY year_date;
!--SELECT dest_state_abr, SUM(taxi_out)-SUM(taxi_in) FROM flights WHERE unique_carrier = 'UA' AND origin = 'ATL' GROUP BY dest_state_abr;


!-- query 1
SELECT dest_state_abr, COUNT(*) FROM flights WHERE 1500<=distance<=2500 GROUP BY dest_state_abr;
!-- query 2
SELECT unique_carrier, COUNT(*) FROM flights WHERE   50<=air_time<=200  GROUP BY unique_carrier;
!-- query 3
SELECT unique_carrier, COUNT(*) FROM flights WHERE 1000<=dep_delay<=1200 AND origin_state_abr='LA'  GROUP BY unique_carrier; 
!-- SELECT unique_carrier, COUNT(*) FROM flights WHERE  origin_state_abr='LA'  GROUP BY unique_carrier;
!-- query 4
SELECT unique_carrier, COUNT(*) FROM flights WHERE 1000<=dep_delay<=1200 AND origin_state_abr='LA'  AND  dest_state_abr='CA' GROUP BY unique_carrier; 
!-- query 5
SELECT dest_state_abr, SUM(taxi_out) FROM flights WHERE 1500<=distance<=2500 unique_carrier = 'UA'  GROUP BY dest_state_abr; 
!-- query 6
SELECT dest_state_abr, SUM(taxi_out) FROM flights WHERE 1500<=distance<=2500 unique_carrier = 'UA'  AND origin = 'ATL' GROUP BY dest_state_abr; 


!-- query 7
SELECT unique_carrier, COUNT(dep_delay) FROM flights WHERE 300<=distance<=1000 GROUP BY unique_carrier;
!--SELECT unique_carrier, COUNT(dep_delay) FROM flights WHERE distance >= 300 and distance <=1000 GROUP BY unique_carrier;
!-- query 8
SELECT unique_carrier, COUNT(dep_delay) FROM flights WHERE 1000<=distance<=1500 GROUP BY unique_carrier;
!-- query 9
SELECT unique_carrier, COUNT(dep_delay) FROM flights WHERE 1500<=distance<=2000 GROUP BY unique_carrier;

!-- query 10
SELECT unique_carrier, SUM(dep_delay) FROM flights WHERE 300<=distance<=1000 GROUP BY unique_carrier;
!-- query 11
SELECT unique_carrier, SUM(dep_delay) FROM flights WHERE 1000<=distance<=1500 GROUP BY unique_carrier;
!-- query 12
SELECT unique_carrier, SUM(dep_delay) FROM flights WHERE 1500<=distance<=2000 GROUP BY unique_carrier;

!-- query 13
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE 300<=distance<=1000 GROUP BY unique_carrier;
!-- query 14
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE 1000<=distance<=1500 GROUP BY unique_carrier;
!-- query 15
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE 1500<=distance<=2000 GROUP BY unique_carrier;


!-- query 16-20
SELECT year_date, SUM(distance) FROM flights WHERE unique_carrier='9E' GROUP BY year_date;
SELECT year_date, SUM(distance) FROM flights WHERE unique_carrier='AA' GROUP BY year_date;
SELECT year_date, SUM(distance) FROM flights WHERE unique_carrier='B6' GROUP BY year_date;
SELECT year_date, SUM(distance) FROM flights WHERE unique_carrier='HA' GROUP BY year_date;
SELECT year_date, SUM(distance) FROM flights WHERE unique_carrier='VX' GROUP BY year_date;
!-- query 21-25
SELECT origin_state_abr, SUM(air_time) FROM flights WHERE dest='ABE' GROUP BY origin_state_abr;
SELECT origin_state_abr, SUM(air_time) FROM flights WHERE dest='ACK' GROUP BY origin_state_abr;
SELECT origin_state_abr, SUM(air_time) FROM flights WHERE dest='BLI' GROUP BY origin_state_abr;
SELECT origin_state_abr, SUM(air_time) FROM flights WHERE dest='CHO' GROUP BY origin_state_abr;
SELECT origin_state_abr, SUM(air_time) FROM flights WHERE dest='FAT' GROUP BY origin_state_abr;
!-- query 26-30
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE year_date=2005 AND origin='ATW' GROUP BY unique_carrier;
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE year_date=2006 AND origin='BQK' GROUP BY unique_carrier;
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE year_date=2007 AND origin='DEN' GROUP BY unique_carrier;
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE year_date=2008 AND origin='MKE' GROUP BY unique_carrier;
SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE year_date=2009 AND origin='PHX' GROUP BY unique_carrier;
!-- query 31-35
SELECT dest_state_abr, AVG(taxi_out) FROM flights WHERE 1500<=distance<=2500 unique_carrier = 'AQ'  GROUP BY dest_state_abr; 
SELECT dest_state_abr, AVG(taxi_out) FROM flights WHERE 2000<=distance<=3000 unique_carrier = 'F9'  GROUP BY dest_state_abr; 
SELECT dest_state_abr, AVG(taxi_out) FROM flights WHERE 1500<=distance<=2500 unique_carrier = 'NW'  GROUP BY dest_state_abr; 
SELECT dest_state_abr, AVG(taxi_out) FROM flights WHERE 2000<=distance<=3000 unique_carrier = 'TZ'  GROUP BY dest_state_abr; 
SELECT dest_state_abr, AVG(taxi_out) FROM flights WHERE 1500<=distance<=2500 unique_carrier = 'UA'  GROUP BY dest_state_abr; 