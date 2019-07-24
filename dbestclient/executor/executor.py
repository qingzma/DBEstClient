# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import sys
import pickle
from dbestclient.io.sampling import DBEstSampling
from dbestclient.ml.modeltrainer import SimpleModelTrainer, GroupByModelTrainer
from dbestclient.parser.parser import DBEstParser
from dbestclient.io import getxy
from dbestclient.ml.regression import DBEstReg
from dbestclient.ml.density import DBEstDensity
from dbestclient.executor.queryengine import QueryEngine
from dbestclient.ml.modelwraper import SimpleModelWrapper, get_pickle_file_name
from dbestclient.catalog.catalog import DBEstModelCatalog
from dbestclient.tools.dftools import convert_df_to_yx, get_group_count_from_df, get_group_count_from_file
import numpy as np

class SqlExecutor:
    """
    This is the executor for the SQL query.
    """

    def __init__(self, config):
        self.parser = None
        self.config = config

        self.model_catalog = DBEstModelCatalog()
        self.init_model_catalog()

    def init_model_catalog(self):
        # search the warehouse, and add all available models.
        # >>>>>>>>>>>>>>>>>>> implement this please!!! <<<<<<<<<<<<<<<<<<
        pass
        # >>>>>>>>>>>>>>>>>>> implement this please!!! <<<<<<<<<<<<<<<<<<

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
                mdl = self.parser.get_ddl_model_name()
                tbl = self.parser.get_from_name()
                original_data_file = self.config['warehousedir'] + "/" + tbl
                yheader = self.parser.get_y()[0]
                xheader = self.parser.get_x()[0]
                ratio = self.parser.get_sampling_ratio()
                method = self.parser.get_sampling_method()

                sampler = DBEstSampling()
                sampler.make_sample(original_data_file, ratio, method, split_char=self.config['csv_split_char'])

                if not self.parser.if_contain_groupby():  # if group by is not involved
                    n_total_point = sampler.n_total_point
                    xys = sampler.getyx(yheader, xheader)
                    simple_model_wrapper = SimpleModelTrainer(mdl,tbl, xheader, yheader,
                                                              n_total_point,ratio).fit_from_df(xys)
                    # reg = DBEstReg().fit(x, y)
                    # density = DBEstDensity().fit(x)
                    # simpleWrapper = SimpleModelWrapper(mdl, tbl, xheader, y=yheader,n_total_point=n_total_point,
                    #                                    n_sample_point=ratio)
                    # simpleWrapper.load_model(density, reg)

                    simple_model_wrapper.serialize2warehouse(self.config['warehousedir'])
                    self.model_catalog.add_model_wrapper(simple_model_wrapper)

                else:  # if group by is involved in the query
                    groupby_attribute = self.parser.get_groupby_value()
                    xys = sampler.getyx(yheader,xheader)
                    # print(xys[groupby_attribute])
                    print(get_group_count_from_file(original_data_file,groupby_attribute,sep=self.config['csv_split_char']))
                    print(get_group_count_from_df(xys,groupby_attribute))
                    # groupby_model_wrapper = GroupByModelTrainer(mdl, tbl, xheader, yheader, groupby_attribute, n_total_point, n_sample_point,
                    #                                             x_min_value=-np.inf, x_max_value=np.inf)
            else:
                # DML, provide the prediction using models
                mdl = self.parser.get_from_name()
                func, yheader = self.parser.get_aggregate_function_and_variable()
                if self.parser.if_where_exists():
                    xheader, x_lb, x_ub = self.parser.get_where_name_and_range()
                    x_lb = float(x_lb)
                    x_ub = float(x_ub)
                    simple_model_wrapper = self.model_catalog.model_catalog[get_pickle_file_name(mdl)]
                    reg = simple_model_wrapper.reg
                    density = simple_model_wrapper.density
                    n_sample_point = int(simple_model_wrapper.n_sample_point)
                    n_total_point = int(simple_model_wrapper.n_total_point)
                    x_min_value = float(simple_model_wrapper.x_min_value)
                    x_max_value = float(simple_model_wrapper.x_max_value)
                if not self.parser.if_contain_groupby():  # if group by is not involved in the query
                    queryengine = QueryEngine(reg, density, n_sample_point, n_total_point, x_min_value, x_max_value,
                                              self.config)
                    if func.lower() == "count":
                        queryengine.approx_count(x_lb, x_ub)
                    elif func.lower() == "sum":
                        queryengine.approx_sum(x_lb, x_ub)
                    elif func.lower() == "avg":
                        queryengine.approx_avg(x_lb, x_ub)
                    else:
                        print("Aggregate function "+ func + " is not implemented yet!")

                else:  # if group by is involved in the query
                    print("group by is currently not supported.")


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
