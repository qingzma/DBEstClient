CREATE EXTERNAL TABLE ci_campusclient_clientstat_5m (
  `ts` bigint, 
  `apmac` string, 
  `acctype` int, 
  `radioid` int, 
  `band` int, 
  `ssid` string, 
  `usermac` string, 
  `downspeed` int, 
  `rssi` int, 
  `uplinkspeed` bigint, 
  `downlinkspeed` bigint, 
  `txdiscardratio` int, 
  `latency` int, 
  `downbytes` bigint, 
  `upbytes` bigint, 
  `kpicount` int, 
  `authtimeouttimes` int, 
  `assofailtimes` int, 
  `authfailtimes` int, 
  `dhcpfailtimes` int, 
  `assosucctimes` int, 
  `authsucctimes` int, 
  `dhcpsucctimes` int, 
  `dot1xsucctimes` int, 
  `dot1xfailtimes` int, 
  `onlinesucctimes` int, 
  `txdiscardframes` int, 
  `txframes` int, 
  `tenantid` string, 
  `siteid` string, 
  `sitename` string, 
  `directregion` string, 
  `regionlevelone` string, 
  `regionleveltwo` string, 
  `regionlevelthree` string, 
  `regionlevelfour` string, 
  `regionlevelfive` string, 
  `regionlevelsix` string, 
  `regionlevelseven` string, 
  `regionleveleight` string, 
  `parentresid` string, 
  `acname` string, 
  `resid` string, 
  `apname` string, 
  `publicarea` int, 
  `vendor` string, 
  `duration` int, 
  `badcount` int, 
  `badtime` int, 
  `lowrssicount` int, 
  `lowrssidur` int, 
  `highlatencycount` int, 
  `highlatencydur` int, 
  `highdiscardcount` int, 
  `highdiscarddur` int, 
  `nonfivegcount` int, 
  `nonfivegdur` int, 
  `exception_flag` int, 
  `last_acc_rst` bigint, 
  `linkquality` float, 
  `portal_succ_times` int, 
  `portal_fail_times` int, 
  `roam_succ_times` int, 
  `roam_fail_times` int)
PARTITIONED BY (folder STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
TBLPROPERTIES ('skip.header.line.count' = '1');

ALTER TABLE ci_campusclient_clientstat_5m ADD PARTITION (folder = '1') LOCATION 'hdfs://master:9000/data/huawei/ci_campusclient_clientstat_5m/data_export_1583337600000_1583424000000_78a5f6f1-ade4-4f9b-a182-c6e31d30ca69';
ALTER TABLE ci_campusclient_clientstat_5m ADD PARTITION (folder = '2') LOCATION 'hdfs://master:9000/data/huawei/ci_campusclient_clientstat_5m/data_export_1583424000000_1583467200000_703ece7f-20ef-4459-a845-c402374746c9';
ALTER TABLE ci_campusclient_clientstat_5m ADD PARTITION (folder = '3') LOCATION 'hdfs://master:9000/data/huawei/ci_campusclient_clientstat_5m/data_export_1583467200000_1583510400000_6e2fecc1-df84-4f7c-8d86-e0609066eb31';
ALTER TABLE ci_campusclient_clientstat_5m ADD PARTITION (folder = '4') LOCATION 'hdfs://master:9000/data/huawei/ci_campusclient_clientstat_5m/data_export_1583510400000_1583596800000_2c5511c7-83b4-4f30-be58-c758e344d43a';
ALTER TABLE ci_campusclient_clientstat_5m ADD PARTITION (folder = '5') LOCATION 'hdfs://master:9000/data/huawei/ci_campusclient_clientstat_5m/data_export_1583683200000_1583726400000_213b14f6-5c8c-4f05-b4fc-51fde91ce343';
ALTER TABLE ci_campusclient_clientstat_5m ADD PARTITION (folder = '6') LOCATION 'hdfs://master:9000/data/huawei/ci_campusclient_clientstat_5m/data_export_1583726400000_1583769600000_a4ecdbf6-d5ff-4dc3-b7cd-9b50c192e70d';
ALTER TABLE ci_campusclient_clientstat_5m ADD PARTITION (folder = '7') LOCATION 'hdfs://master:9000/data/huawei/ci_campusclient_clientstat_5m/data_export_1583769600000_1583856000000_c6613e14-e3a6-4139-95d1-9f37decda0a9';


data_export_1583337600000_1583424000000_78a5f6f1-ade4-4f9b-a182-c6e31d30ca69
data_export_1583424000000_1583467200000_703ece7f-20ef-4459-a845-c402374746c9
data_export_1583467200000_1583510400000_6e2fecc1-df84-4f7c-8d86-e0609066eb31
data_export_1583510400000_1583596800000_2c5511c7-83b4-4f30-be58-c758e344d43a
data_export_1583683200000_1583726400000_213b14f6-5c8c-4f05-b4fc-51fde91ce343
data_export_1583726400000_1583769600000_a4ecdbf6-d5ff-4dc3-b7cd-9b50c192e70d
data_export_1583769600000_1583856000000_c6613e14-e3a6-4139-95d1-9f37decda0a9



hdfs dfs -mkdir /data/huawei/ci_campusclient_clientstat_5m
hdfs dfs -mkdir /data/huawei/ci_campusnetwork_radiokpi_1m

hdfs dfs -put /data/huawei/ci_campusclient_clientstat_5m/* /data/huawei/ci_campusclient_clientstat_5m
hdfs dfs -put /data/huawei/ci_campusnetwork_radiokpi_1m/* /data/huawei/ci_campusnetwork_radiokpi_1m



SELECT ts, COUNT(DISTINCT usermac) 
FROM ci_campusclient_clientstat_5m
WHERE 
ts >= unix_timestamp('2019-03-28T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
AND ts <= unix_timestamp('2020-03-29T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
AND tenantId = 'default-organization-id' 
AND kpiCount >= 0 
AND ssid = 'Apple' 
AND regionLevelEight = '9f642594-20c2-4ccb-8f5d-97d5f59a1e18'
GROUP BY ts;


select ts,regionLevelEight,ssid, COUNT(DISTINCT usermac) 
FROM ci_campusclient_clientstat_5m
WHERE 
ts >= unix_timestamp('2020-03-09T02:10:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
AND ts <= unix_timestamp('2020-03-10T02:10:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
AND tenantId = 'default-organization-id' 
AND kpiCount > 0 
GROUP BY ts,regionLevelEight,ssid;

AND ssid = 'Apple' 
Apple, Huawei, Tencent

select distinct ts FROM ci_campusclient_clientstat_5m;
select distinct regionLevelEight FROM ci_campusclient_clientstat_5m;

select kpiCount,count(kpiCount) from ci_campusclient_clientstat_5m group by kpiCount;

select tenantId,count(tenantId) from ci_campusclient_clientstat_5m group by tenantId;
select ssid,count(ssid) from ci_campusclient_clientstat_5m group by ssid;


select tenantId,kpiCount,regionLevelEight,ssid, count(*) 
from ci_campusclient_clientstat_5m 
WHERE 
ts >= unix_timestamp('2020-03-09T02:10:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
AND ts <= unix_timestamp('2020-03-10T02:10:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
group by tenantId,kpiCount,regionLevelEight,ssid;





SELECT ts, COUNT(DISTINCT usermac)  FROM ci_campusclient_clientstat_5m WHERE  ts >= unix_timestamp('2019-03-28T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000 AND ts <= unix_timestamp('2020-03-29T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000 AND tenantId = 'default-organization-id' AND kpiCount >= 0 AND ssid = 'Apple' AND regionLevelEight = '9f642594-20c2-4ccb-8f5d-97d5f59a1e18'GROUP BY ts;



# to support the SQL where GROUP BY and X is the same attriubte,  I remove the X attribute from the X columns, and only use X in the GB columns, then add a filter to choose the propriate groups from all predictions.



select ts, count( usermac) from hw_sample 
where ts between unix_timestamp('2020-01-28T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000 and  unix_timestamp('2020-04-28T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
AND tenantId = 'default-organization-id' 
AND ssid = 'Tencent' 
GROUP BY ts;
                        # "AND regionLevelEight = '9f642594-20c2-4ccb-8f5d-97d5f59a1e18' "
                        # "AND kpiCount = 0 "