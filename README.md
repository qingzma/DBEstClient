# Welcome to DBEst++
This repository extends the work of  [DBEst: Revisiting Approximate Query Processing Engines with Machine Learning Models . Proceedings of the 2019 International Conference on Management of Data. ACM, 2019](https://dl.acm.org/citation.cfm?id=3324958) by applying MDN. This solves the problem of model updating.

## How to install
```pip install dbestclient```

If you have problems installing the code, you could go to the root directory of DBEst, and install the latest version by 
```pip install -e```.
## How to start
After the installation of DBEstClient, simply type **dbestclient** in the terminal.
```>>> dbestclient```
If nothing goes wrong, you will get:
```
Configuration file config.json does not exist! use default values
warehouse does not exists, so initialize one.
Welcome to DBEst: a model-based AQP engine! Type exit to exit!
dbestclient>
```
Then you can input your SQL queries.

## Dependencies
- python>=3.6
- [numpy](https://github.com/numpy/numpy)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [pandas](https://github.com/pandas-dev/pandas)
- [qreg](https://github.com/qingzma/qreg), etc.

## Features
- provide approximate answers for SQL queries.
- currenly support COUNT, SUM, AVG

## Syntax
Supported SQL queries include model creation and query answering:

To create a model to answer a DML SQL as the following format:
```
SELECT [gb1 [, gb2,...]], COUNT([DISTINCT] y) 
FROM tbl
WHERE 0<=x1<=99
AND x2='15'
AND x3=unix_timestamp('2020-03-05T12:00:00.000Z')
[GROUP BY gb1 [, gb2,... ]];
```
you need to create a model:
- **model creation**
	```
	CREATE TABLE tbl_mdl(
		y REAL|CATEGORICAL [DISTINCT], 
		x1 REAL, 
		x2 CATEGORICAL, 
		x3 CATEGORICAL)  
	FROM '/data/sample.csv'  
	[GROUP BY gb1 [, gb2,... ]]  
	[SIZE 10000|0.1]  
	[METHOD UNIFROM|HASH]
	[SCALE FILE|DATA (file_name)]
	```
	<!-- [ENCODING ONEHOT|BINARY] -->

- **query answering** 
	```
	SELECT [gb1 [, gb2,... ],] AF(y)  
	FROM tbl_mdl  
	WHERE 0<=x1<=99
	AND x2='15'
	AND x3=unix_timestamp('2020-03-05T12:00:00.000Z'
	[GROUP BY gb1 [, gb2,... ]]
	```
## Example
Currently, there is no backend server, and DBEst handles csv files with headers.
- After starting DBEst, you should notice a directory called **dbestwarehouse**  and a configuration file called **config.json** in your current working directory.
- Simply copy the csv file in the directory, and you could create a model for it.

### Beijing PM2.5 example
- download the file ``` wget -O pm25.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv```
-  copy the file to the warehouse directory **dbestwarehouse**
- start dbestclient and create a model between pm2.5 and PRES, with sample size of 1000 by 
```create table mdl(pm25 real, PRES real) from pm25.csv  method uniform size 1000```
 (Note, you need to open the file and rename the header from pm2.5 to pm25)
- Then get result from model only!
``` select count(pm25 real) from mdl where PRES between 1000 and 1020;```
	```
	OK
	578.380307583211
	time cost: 0.014005
	------------------------
	```
### Huawei Dataset
The entry point for the code is experiments/huawei/groupby_huawei_test.py, you could run ```python experiments/huawei/groupby_huawei_test.py``` instead of using the SQL interface.

The sample data is used for testing purposes.

To create a model to support the following simplied query (Q1), 
```
SELECT ts, COUNT( DISTINCT usermac) 
FROM hw_sample 
WHERE ts BETWEEN unix_timestamp('2020-01-28T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000 AND  unix_timestamp('2020-04-28T16:00:00.000Z',"yyyy-MM-dd'T'HH:mm:ss.SSSX")*1000
AND tenantId = 'default-organization-id' 
AND ssid = 'Tencent' 
GROUP BY ts;
```
We need to  train a model based on following SQL format:
```
create table huawei_test(usermac CATEGORICAL DISTINCT, ts REAL, tenantId CATEGORICAL, ssid  CATEGORICAL)  
FROM '/data/huawei/sample.csv' 
GROUP BY ts 
METHOD UNIFORM 
SIZE 118567 
SCALE data;
```
Here, 
- The first attribute in huawei_test(*) is usermac, which is the dependent variable, and the following attributes are the independent variables.
- CATEGORICAL means usermac should be treated as a categorical attribute instead of a real value.
- DISTINCT means this model is used for answer query involving DISTINCT on usermac.
- ts is REAL, which means a density function will be trained on attribute ts, and the SQL is expected to have a range selector on ts, like ``` WHERE ts BETWEEN * AND * ```.
-  tenantId and ssid are of type CATEGORICAL, which means there will be a equal clause in the WHERE clause, like ```AND ssid = 'Tencent'  ```
- SIZE: if ```SIZE > 1```, it is the sample size you want the sample to be, 118567 is the data size, so it is a 100% sample. if ```0 < SIZE < 1```, the data you provided is treated as a sample with the sampling rate provided.

To execute a SQL query from the model, you could use the following SQL format:
```
SELECT ts, COUNT(DISTINCT usermac) FROM huawei_test 
WHERE ts between unix_timestamp('2020-01-28T16:00:00.000Z') and unix_timestamp('2020-04-28T16:00:00.000Z') 
AND tenantId = 'default-organization-id' 
AND ssid = 'Tencent' 
GROUP BY ts;
```
Where,
- unix_timestamp() is the function to convert the date strings into timestamps.
- the condition on ```kpiCount>0``` is currently not supported, only ```=``` is supported at this moment. It will be soon supported.

## Documentation

## TODO 
- ~~SQL size cluase to include num_count.csv~~
- ~~drop model~~.
- embedding. (Not feasible)
- one mdn model instead of many.
- ~~fix GoGs implementation.~~
- query check
- distinct and not distinct using one model only.
- show models.
