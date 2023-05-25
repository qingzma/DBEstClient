# Welcome to DBEst++: A LightWeight Approximate Query Processing (AQP) Engine
This repository implements the idea for the following papers.

<a id="1">[1]</a> 
Qingzhi Ma, Peter Triantafillou. 
DBEst: Revisiting Approximate Query Processing Engines with Machine Learning Models. 
Proceedings of the 2019 International Conference on Management of Data. ACM, 2019.

<a id="2">[2]</a> 
Qingzhi Ma, Ali M Shanghooshabad, Mehrdad Almasi, Meghdad Kurmanji, Peter Triantafillou. 
Learned Approximate Query Processing: Make it Light, Accurate and Fast. 
The Conference on Innovative Data Systems Research (CIDR) . 2021.

## How to install
DBEst is developed on Linux OS, and is tested on Mac OS as well.
Please use Anaconda to manage the DBEst Python environment.

there is a file called environment.yml, which could be used to create a separate Python environment.

```conda env create -n torch --file environment.yml```

```conda activate torch```

After that, use

```pip install -e path/to/dbest/folder```

to install DBEst locally.
Hopefully, you will get DBEst installed correctly. 
also, please follow instructions of branch 2.0 instead the instruction in this branch(demo).
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
- **CREATE A MODEL**
	```
	CREATE TABLE tbl_mdl(
		y REAL|CATEGORICAL [DISTINCT], 
		x1 REAL, 
		x2 CATEGORICAL, 
		x3 CATEGORICAL)  
	FROM '/data/sample.csv'  
	[GROUP BY gb1 [, gb2,... ]]  
	[SIZE 10000|0.1|'path/to/file']  
	[METHOD UNIFROM|HASH]
	```
	<!-- [ENCODING ONEHOT|BINARY] -->
	Note,
	- The first attribute ```y``` must be the dependent variable (the aggregate attribute.)
	- ```SIZE``` less than 1 means the given data file is treated as a sample with sampling ratio provided. ```SIZE``` greater than 1 means the given data file is treated as the whole data, and a sample of size n will be made. if ```SIZE``` is a path to a file, it means the given data file is treated as a sample, and the frequency of each group is provided in the file. 

To get the query result,
- **ANSWER A QUERY** 
	```
	SELECT [gb1 [, gb2,... ],] AF(y)  
	FROM tbl_mdl  
	WHERE 0<=x1<=99
	AND x2='15'
	AND x3=unix_timestamp('2020-03-05T12:00:00.000Z'
	[GROUP BY gb1 [, gb2,... ]]
	```

To show available models,
- **SHOW MODELS**
	```
	SHOW TABLES;
	```

To drop a model,
- **DROP MODEL**
	```
	DROP TABLE model_name;
	```

To set a parameter,
- **SET A PARAMETER**
	```
	SET name=[']value['];
	```
	The parameters controls the behaviors of DBEst. For example, 
	- ```SET device='gpu'``` enables GPU.
	- ```SET n_jobs=2```     enables parallel inferencing using 2 processes.
	- ```SET v='true'```     enables verbose.
	- ```SET b_print_to_screen='true'``` will print the query results to the screen.
	- ```SET result2file='/path/to/result.file'``` will save the query result to the given file.
	- ```SET b_grid_search='true'``` enables grid search during model training.
	- ```SET table_header='col1,col2,...col_n'``` provides headers for the give data file, if the file header is not provided.
	- ```SET encoder='binary'``` sets the encoding method to be ```binary```. ```one-hot``` is also supported.
	- ```SET n_epoch=20``` sets the training epoch to be 20.


## Example
Currently, there is no backend server, and DBEst handles csv files with headers.
- After starting DBEst, you should notice a directory called **dbestwarehouse**  in your current working directory.
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


## Cluster Mode
Cluster mode could be enabled to enhance the inference performance.

- On the slave nodes, open terminal 

	```dbestslave <host>:<IP>``` to enable the slave serving.

- On the master node,
	Simply place a file called ```slaves``` in dbestwarehouse. A sample salves file looks like 

	```
	127.0.0.1:65432
	127.0.0.1:65433
	127.0.0.2:65432
	```
	Then you are ready to go.

## Run tests
you need to install tox to run the tests.
assuming you are using Anaconda, simply call
```
conda install -c conda-forge tox
```
after that, call
```
tox
```
to run all tests.

## Documentation

## TODO 

