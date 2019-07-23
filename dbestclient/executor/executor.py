# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.parser.parser import DBEstParser
from dbestclient.io import getxy
from dbestclient.ml.regression import DBEstReg
from dbestclient.ml.density import DBEstDensity
from dbestclient.executor.queryengine import QueryEngine

class SqlExecutor:
    """
    This is the executor for the SQL query.
    """
    def __init__(self, config):
        self.parser = None
        self.config = config

    def execute(self, sql):
        # prepare the parser
        if type(sql) == str:
            self.parser = DBEstParser()
            self.parser.parse(sql)
        elif type(sql) == DBEstParser:
            self.parser = sql
        else:
            print("Unrecognized SQL! Please check it!")
            exit(-1)

        # execute the query
        if self.parser.if_nested_query():
            print("Nested query is currently not supported!")
        else:
            if self.parser.if_ddl():
                # DDL, create the model as requested
                file = self.config['warehousedir'] + "/" + self.parser.get_from_name()
                yheader = self.parser.get_y()[0]
                xheader = self.parser.get_x()[0]
                # print(file)
                # print(yheader)
                # print(xheader)

                if not self.parser.if_contain_groupby(): # if group by is not involved
                    getter = getxy.GetXY(backend=self.config['backend_server'])
                    y, x = getter.read(file, yheader, xheader)
                    reg = DBEstReg().fit(x,y)
                    density = DBEstDensity().fit(x)
                    queryengine = QueryEngine(reg,density,100, 1000, 1, 100, self.config)
                    queryengine.approx_count(1000, 10000)
                    queryengine.approx_sum(1000, 10000)
                    queryengine.approx_avg(1000, 10000)
                else: # if group by is involved in the query
                    print("group by is currently not supported.")
            else:
                # DML, provide the prediction using models
                pass


if __name__ == "__main__":
    config = {
        'warehousedir': 'dbestwarehouse',
        'verbose': 'True',
        'b_show_latency': 'True',
        'backend_server': 'None',
    }
    sqlExecutor = SqlExecutor(config)
    sqlExecutor.execute("create table mdl(pm25 real, PRES real) from pm25.csv group by z method uniform size 0.1")
    print(sqlExecutor.parser.parsed)


