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
from dbestclient.ml.modelwraper import SimpleModelWrapper, get_pickle_file_name, GroupByModelWrapper
from dbestclient.catalog.catalog import DBEstModelCatalog
from dbestclient.tools.dftools import convert_df_to_yx, get_group_count_from_df, get_group_count_from_file
import numpy as np
from datetime import datetime
import os


class SqlExecutor:
    """
    This is the executor for the SQL query.
    """

    def __init__(self, config):
        self.parser = None
        self.config = config

        self.model_catalog = DBEstModelCatalog()
        self.init_model_catalog()
        # exit()

    def init_model_catalog(self):
        # search the warehouse, and add all available models.
        # >>>>>>>>>>>>>>>>>>> implement this please!!! <<<<<<<<<<<<<<<<<<
        n_model = 0
        for file_name in os.listdir(self.config['warehousedir']):

            # load simple models
            if file_name.endswith(".pkl"):
                if n_model == 0:
                    print("start loading pre-existing models.")

                with open(self.config['warehousedir'] + "/" + file_name, 'rb') as f:
                    model = pickle.load(f)
                self.model_catalog.model_catalog[model.init_pickle_file_name(
                )] = model
                n_model += 1

            # load group by models
            if os.path.isdir(self.config['warehousedir'] + "/" + file_name):
                n_models_in_groupby = 0
                if n_model == 0:
                    print("start loading pre-existing models.")

                for model_name in os.listdir(self.config['warehousedir'] + "/" + file_name):
                    if model_name.endswith(".pkl"):
                        with open(self.config['warehousedir'] + "/" + file_name + "/"+model_name, 'rb') as f:
                            model = pickle.load(f)
                            n_models_in_groupby += 1

                        if n_models_in_groupby == 1:
                            groupby_model_wrapper = GroupByModelWrapper(model.mdl, model.tbl, model.x, model.y,
                                                                        model.groupby_attribute,
                                                                        x_min_value=model.x_min_value,
                                                                        x_max_value=model.x_max_value)
                        groupby_model_wrapper.add_simple_model(model)

                self.model_catalog.model_catalog[file_name] = groupby_model_wrapper.models
                n_model += 1

        if n_model > 0:
            print("Loaded " + str(n_model) + " models.")
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
                # print(self.config)
                sampler.make_sample(
                    original_data_file, ratio, method, split_char=self.config['csv_split_char'])

                if not self.parser.if_contain_groupby():  # if group by is not involved
                    # check whether this model exists, if so, skip training
                    if os.path.exists(self.config['warehousedir'] + "/" + mdl + '.pkl'):
                        print(
                            "Model {0} exists in the warehouse, please use another model name to train it.".format(mdl))
                        return

                    n_total_point = sampler.n_total_point
                    xys = sampler.getyx(yheader, xheader)
                    # print(xys)
                    simple_model_wrapper = SimpleModelTrainer(mdl, tbl, xheader, yheader,
                                                              n_total_point, ratio).fit_from_df(xys)
                    # reg = DBEstReg().fit(x, y)
                    # density = DBEstDensity().fit(x)
                    # simpleWrapper = SimpleModelWrapper(mdl, tbl, xheader, y=yheader,n_total_point=n_total_point,
                    #                                    n_sample_point=ratio)
                    # simpleWrapper.load_model(density, reg)

                    simple_model_wrapper.serialize2warehouse(
                        self.config['warehousedir'])
                    self.model_catalog.add_model_wrapper(simple_model_wrapper)

                else:  # if group by is involved in the query
                    groupby_attribute = self.parser.get_groupby_value()
                    # check whether this model exists, if so, skip training
                    if os.path.exists(self.config['warehousedir'] + "/" + mdl+"_groupby_" + groupby_attribute):
                        print(
                            "Model {0} exists in the warehouse, please use another model name to train it.".format(mdl))
                        return

                    xys = sampler.getyx(yheader, xheader)
                    # print(xys[groupby_attribute])
                    n_total_point = get_group_count_from_file(
                        original_data_file, groupby_attribute, sep=self.config['csv_split_char'])
                    n_sample_point = get_group_count_from_df(
                        xys, groupby_attribute)
                    groupby_model_wrapper = GroupByModelTrainer(mdl, tbl, xheader, yheader, groupby_attribute, n_total_point, n_sample_point,
                                                                x_min_value=-np.inf, x_max_value=np.inf).fit_from_df(xys)
                    groupby_model_wrapper.serialize2warehouse(
                        self.config['warehousedir']+"/"+groupby_model_wrapper.dir)
                    self.model_catalog.model_catalog[groupby_model_wrapper.dir] = groupby_model_wrapper.models

            else:
                # DML, provide the prediction using models
                mdl = self.parser.get_from_name()
                func, yheader = self.parser.get_aggregate_function_and_variable()
                if self.parser.if_where_exists():
                    xheader, x_lb, x_ub = self.parser.get_where_name_and_range()
                    x_lb = float(x_lb)
                    x_ub = float(x_ub)

                else:
                    print(
                        "support for query without where clause is not implemented yet! abort!")

                if not self.parser.if_contain_groupby():  # if group by is not involved in the query
                    simple_model_wrapper = self.model_catalog.model_catalog[get_pickle_file_name(
                        mdl)]
                    reg = simple_model_wrapper.reg
                    density = simple_model_wrapper.density
                    n_sample_point = int(simple_model_wrapper.n_sample_point)
                    n_total_point = int(simple_model_wrapper.n_total_point)
                    x_min_value = float(simple_model_wrapper.x_min_value)
                    x_max_value = float(simple_model_wrapper.x_max_value)
                    query_engine = QueryEngine(reg, density, n_sample_point, n_total_point, x_min_value, x_max_value,
                                               self.config)
                    p, t = query_engine.predict(func, x_lb=x_lb, x_ub=x_ub)
                    print("OK")
                    print(p)
                    if self.config['verbose']:
                        print("time cost: " + str(t))
                    print("------------------------")

                else:  # if group by is involved in the query
                    start = datetime.now()
                    predictions = {}
                    groupby_attribute = self.parser.get_groupby_value()
                    groupby_key = mdl + "_groupby_"+groupby_attribute

                    for group_value, model_wrapper in self.model_catalog.model_catalog[groupby_key].items():
                        reg = model_wrapper.reg
                        density = model_wrapper.density
                        n_sample_point = int(model_wrapper.n_sample_point)
                        n_total_point = int(model_wrapper.n_total_point)
                        x_min_value = float(model_wrapper.x_min_value)
                        x_max_value = float(model_wrapper.x_max_value)
                        query_engine = QueryEngine(reg, density, n_sample_point, n_total_point, x_min_value,
                                                   x_max_value,
                                                   self.config)
                        predictions[model_wrapper.groupby_value] = query_engine.predict(
                            func, x_lb=x_lb, x_ub=x_ub)[0]

                    print("OK")
                    for key, item in predictions.items():
                        print(key, item)

                    if self.config['verbose']:
                        end = datetime.now()
                        time_cost = (end - start).total_seconds()
                        print("Time cost: %.4fs." % time_cost)
                    print("------------------------")


if __name__ == "__main__":
    config = {
        'warehousedir': '/home/u1796377/Programs/dbestwarehouse',
        'verbose': 'True',
        'b_show_latency': 'True',
        'backend_server': 'None',
        'csv_split_char': ',',
        "epsabs": 10.0,
        "epsrel": 0.1,
        "mesh_grid_num": 20,
        "limit": 30,
    }
    sqlExecutor = SqlExecutor(config)
    # sqlExecutor.execute("create table mdl(pm25 real, PRES real) from pm25.csv group by z method uniform size 0.1")
    sqlExecutor.execute("create table mdl(pm25 real, PRES real) from pm25.csv method uniform size 2000")
    sqlExecutor.execute(
        "select avg(pm25)  from mdl where PRES between 1000 and 1010")
    sqlExecutor.execute(
        "select avg(pm25)  from mdl1 where PRES between 1000 and 1010")
    print(sqlExecutor.parser.parsed)
