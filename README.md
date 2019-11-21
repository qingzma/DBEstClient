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

- **model creation**
	```
	CREATE TABLE t_m(y real, x real)  
	FROM tbl  
	[GROUP BY z]  
	[SIZE 10000]  
	[METHOD UNIFROM|HASH]
	```

- **query answering** 
	```
	SELECT AF(y)  
	FROM t_m  
	[WHERE x BETWEEN a AND b]  
	[GROUP BY z]
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
``` select count(pm25 real) from mdl1 where PRES between 1000 and 1020;```
	```
	OK
	578.380307583211
	time cost: 0.014005
	------------------------
	```

## Documentation