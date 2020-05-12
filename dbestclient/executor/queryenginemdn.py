# Created by Qingzhi Ma at 29/01/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

import math
from collections import Counter
from datetime import datetime
from operator import itemgetter

import dill
import numpy as np
import pandas as pd
from scipy import integrate
from torch.multiprocessing import Pool, set_start_method

# from dbestclient.io.sampling import DBEstSampling
from dbestclient.ml.integral import (approx_avg, approx_count,
                                     approx_integrate, approx_sum,
                                     prepare_reg_density_data)
from dbestclient.ml.modeltrainer import KdeModelTrainer

# from dbestclient.tools.dftools import get_group_count_from_summary_file

# try:
#     set_start_method('spawn')
# except RuntimeError:
#     print("Fail to set start method as spawn for pytorch multiprocessing, " +
#           "use default in advance. (see queryenginemdn "
#           "for more info.)")


class MdnQueryEngine:
    def __init__(self, kdeModelWrapper, config=None, b_use_integral=False):
        # self.n_training_point = kdeModelWrapper.n_sample_point
        self.n_total_point = kdeModelWrapper.n_total_point
        self.reg = kdeModelWrapper.reg
        self.kde = kdeModelWrapper.density
        self.x_min = kdeModelWrapper.x_min_value
        self.x_max = kdeModelWrapper.x_max_value
        self.groupby_attribute = kdeModelWrapper.groupby_attribute
        self.groupby_values = kdeModelWrapper.groupby_values
        self.density_column = kdeModelWrapper.x

        # if config is None:
        #     self.config = config = {
        #         'warehousedir': 'dbestwarehouse',
        #         'verbose': 'True',
        #         'b_show_latency': 'True',
        #         'backend_server': 'None',
        #         'epsabs': 10.0,
        #         'epsrel': 0.1,
        #         'mesh_grid_num': 20,
        #         'limit': 30,
        #         'csv_split_char': '|',
        #         'num_epoch': 400,
        #         "reg_type": "mdn",
        #     }
        # else:
        self.config = config
        self.b_use_integral = config.get_config()["b_use_integral"]

    def approx_avg(self, x_min, x_max, groupby_value):
        start = datetime.now()

        def f_pRx(*args):
            # print(self.cregression.predict(x))
            return self.kde.predict([[groupby_value]], args[0], b_plot=False) \
                * self.reg.predict(np.array([[args[0], groupby_value]]))[0]

        def f_p(*args):
            return self.kde.predict([[groupby_value]], args[0], b_plot=False)

        if self.b_use_integral:
            a = integrate.quad(f_pRx, x_min, x_max,
                               epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[0]
            b = integrate.quad(f_p, x_min, x_max,
                               epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[0]
        else:
            a = approx_integrate(f_pRx, x_min, x_max)
            b = approx_integrate(f_p, x_min, x_max)

        if b:
            result = a / b
        else:
            result = None

        if self.config['verbose']:
            end = datetime.now()
            time_cost = (end - start).total_seconds()
            # print("Time spent for approximate AVG: %.4fs." % time_cost)
        return result, time_cost

    def approx_sum(self, x_min, x_max, groupby_value):
        start = datetime.now()

        def f_pRx(*args):
            return self.kde.predict([[groupby_value]], args[0], b_plot=True) \
                * self.reg.predict(np.array([[args[0], groupby_value]]))[0]
            # * self.reg.predict(np.array(args))

        # print(integrate.quad(f_pRx, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[0])
        if self.b_use_integral:
            result = integrate.quad(f_pRx, x_min, x_max, epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[
                0] * float(self.n_total_point[str(int(groupby_value))])
        else:
            result = approx_integrate(
                f_pRx, x_min, x_max) * float(self.n_total_point[str(int(groupby_value))])
        # return result

        # result = result / float(self.n_training_point) * float(self.n_total_point)

        # print("Approximate SUM: %.4f." % result)

        if self.config['verbose'] and result != None:
            end = datetime.now()
            time_cost = (end - start).total_seconds()
            # print("Time spent for approximate SUM: %.4fs." % time_cost)
        return result, time_cost

    def approx_count(self, x_min, x_max, groupby_value):
        start = datetime.now()

        def f_p(*args):
            return self.kde.predict([[groupby_value]], args[0], b_plot=False)
            # return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

        if self.b_use_integral:
            result = integrate.quad(
                f_p, x_min, x_max, epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[0]
        else:
            result = approx_integrate(f_p, x_min, x_max)
        result = result * float(self.n_total_point[str(int(groupby_value))])

        # print("Approximate COUNT: %.4f." % result)
        if self.config['verbose'] and result != None:
            end = datetime.now()
            time_cost = (end - start).total_seconds()
            # print("Time spent for approximate COUNT: %.4fs." % time_cost)
        return result, time_cost

    def predict(self, func, x_lb, x_ub, groupby_value):
        if func.lower() == "count":
            p, t = self.approx_count(x_lb, x_ub, groupby_value)
        elif func.lower() == "sum":
            p, t = self.approx_sum(x_lb, x_ub, groupby_value)
        elif func.lower() == "avg":
            p, t = self.approx_avg(x_lb, x_ub, groupby_value)
        else:
            print("Aggregate function " + func + " is not implemented yet!")
        return p, t

    def predicts(self, func, x_lb, x_ub, b_parallel=True, n_jobs=4, ):  # result2file=None
        result2file = self.config.get_config()["result2file"]
        predictions = {}
        times = {}
        if not b_parallel:  # single process implementation
            for groupby_value in self.groupby_values:
                if groupby_value == "":
                    continue
                pre, t = self.predict(func, x_lb, x_ub, groupby_value)
                predictions[groupby_value] = pre
                times[groupby_value] = t
                # print(groupby_value, pre)
        else:  # multiple threads implementation

            width = int(len(self.groupby_values) / n_jobs)
            subgroups = [self.groupby_values[inde:inde + width]
                         for inde in range(0, len(self.groupby_values), width)]
            if len(self.groupby_values) % n_jobs != 0:
                subgroups[n_jobs - 1] = subgroups[n_jobs - 1] + \
                    subgroups[n_jobs]
                del subgroups[-1]
            # index_in_groups = [[self.groupby_values.index(sgname) for sgname in sgnames] for sgnames in subgroups]

            instances = []

            with Pool(processes=n_jobs) as pool:
                for subgroup in subgroups:
                    i = pool.apply_async(
                        query_partial_group, (self, subgroup, func, x_lb, x_ub))
                    instances.append(i)
                    # print(i.get())
                    # print(instances[0].get(timeout=1))
                for i in instances:
                    result = i.get()
                    pred = result[0]
                    t = result[1]
                    predictions.update(pred)
                    times.update(t)
                    # predictions += pred
                    # times += t
        if result2file is not None:
            with open(result2file, 'w') as f:
                for key in predictions:
                    f.write(key + "," + str(predictions[key]))
        return predictions, times

    def predict_one_pass(self, func: str, x_lb: float, x_ub: float, groups: list = None,  n_jobs: int = 1, filter_dbest=None):
        # b_print_to_screen=True, n_division: int = 20, result2file: str = None,

        b_print_to_screen = self.config.get_config()["b_print_to_screen"]
        n_division = self.config.get_config()["n_division"]
        result2file = self.config.get_config()["result2file"]
        # result2file = self.config.get_config()["result2file"]

        if func.lower() not in ("count", "sum", "avg"):
            raise ValueError("function not supported: "+func)
        if groups is None:  # provide predictions for all groups.
            groups = self.groupby_values

        if self.config.get_config()["accept_filter"]:
            results = self.n_total_point
            # print("results,", results)
            # print("filter_dbest", filter_dbest)
            results = {key: results[key] for key in results if float(
                key) >= filter_dbest[0] and float(key) <= filter_dbest[1]}

            # scale up the result
            scaling_factor = self.config.get_config()["scaling_factor"]
            # print("scaling_factor", scaling_factor)
            results = {key: results[key]*scaling_factor for key in results}

            # print("self.n_total_point", self.n_total_point)
            if b_print_to_screen:
                for key in results:
                    print(",".join(key.split("-")) + "," + str(results[key]))

            if result2file is not None:
                with open(result2file, 'w') as f:
                    for key in results:
                        f.write(str(key) + "," + str(results[key]) + "\n")
            return results

        if n_jobs == 1:
            # print(groups)
            # print(self.n_total_point)
            # print(groups[0], "*******************")
            # print(",".join(groups[0]))
            # for key in groups:
            # print("key is ", key, end="----- ")
            # print(",".join(key))
            if len(groups[0].split(",")) == 1:  # 1d group by
                # print(groups[0].split(","))
                # print("1d")
                scaling_factor = np.array([self.n_total_point[key]
                                           for key in groups])
            else:
                # print(groups[0].split(","))
                # print("n-d")
                scaling_factor = np.array([self.n_total_point[key]
                                           for key in groups])
            print("self.n_total_point", self.n_total_point)
            pre_density, pre_reg, step = prepare_reg_density_data(
                self.kde, x_lb, x_ub, groups=groups, reg=self.reg, n_division=n_division)

            if func.lower() == "count":
                preds = approx_count(pre_density, step)
                preds = np.multiply(preds, scaling_factor)
            elif func.lower() == "sum":
                preds = approx_sum(pre_density, pre_reg, step)
                preds = np.multiply(preds, scaling_factor)
            else:  # avg
                preds = approx_avg(pre_density, pre_reg, step)
            results = dict(zip(groups, preds))

        else:
            instances = []
            results = {}
            n_per_chunk = math.ceil(len(groups)/n_jobs)
            group_chunks = [groups[i:i+n_per_chunk]
                            for i in range(0, len(groups), n_per_chunk)]
            with Pool(processes=n_jobs) as pool:
                # print(self.group_keys_chunk)
                for sub_group in group_chunks:
                    # print(sub_group)
                    # engine = self.enginesContainer[index]
                    # print(sub_group)
                    i = pool.apply_async(
                        self.predict_one_pass, (func, x_lb, x_ub, sub_group, False, n_division))
                    instances.append(i)

                for i in instances:
                    result = i.get()
                    results.update(result)
        if b_print_to_screen:
            for key in results:
                print(",".join(key.split("-")) + "," + str(results[key]))

        if result2file is not None:
            with open(result2file, 'w') as f:
                for key in results:
                    f.write(str(key) + "," + str(results[key]) + "\n")
        return results

    # def set_parameter(self, key, value):
    #     self.config.get_config()[key] = value


def query_partial_group(mdnQueryEngine, group, func, x_lb, x_ub):
    mdnQueryEngine.groupby_values = group
    return mdnQueryEngine.predicts(func, x_lb, x_ub, b_parallel=False, n_jobs=1)


class MdnQueryEngineBundle():
    def __init__(self, config, device):
        self.enginesContainer = {}
        self.config = config
        self.n_total_point = None
        self.group_keys_chunk = None
        self.group_keys = None
        self.pickle_file_name = None
        self.device = device
        self.density_column = None

    def fit(self, df: pd.DataFrame, groupby_attribute: str, n_total_point: dict,
            mdl: str, tbl: str, xheader: str, yheader: str, ):  # n_per_group: int = 10, n_mdn_layer_node=10, encoding = "onehot", b_grid_search = True
        # configuration-related parameters.
        n_per_group = self.config.get_config()["n_per_group"]
        n_mdn_layer_node = self.config.get_config()["n_mdn_layer_node"]
        encoding = self.config.get_config()["encoding"]
        b_grid_search = self.config.get_config()["b_grid_search"]

        self.density_column = xheader

        self.pickle_file_name = mdl
        # print(groupby_attribute)
        # print(df)

        grouped = df.groupby(groupby_attribute)

        # print("grouped", grouped)

        self.group_keys = list(grouped.groups.keys())

        # print("self.group_keys", self.group_keys[:20])
        # print(self.group_keys[0], type(self.group_keys[0]))

        # print("group_keys", self.group_keys)

        # sort the group key by value
        # fakeKey = []
        # for key in self.group_keys:
        #     if key == "":
        #         k = 0.0
        #     else:
        #         try:
        #             k = float(key)
        #         except ValueError:
        #             raise ValueError(
        #                 "ValueError: could not convert string to float in " + __file__)
        #     fakeKey.append(k)

        # self.group_keys = [k for _, k in sorted(zip(fakeKey, self.group_keys))]
        if isinstance(self.group_keys[0], tuple):  # n-d group by  #tuple
            group_keys_float = []
            for idx, group_key in enumerate(self.group_keys):
                group_key_float = [idx]
                for group in group_key:
                    if group == "":
                        g = 0.0
                    else:
                        try:
                            g = float(group)
                        except ValueError:
                            raise ValueError(
                                "ValueError: could not convert string to float in " + __file__)
                    group_key_float.append(g)
                group_keys_float.append(group_key_float)
            group_keys_float = sorted(
                group_keys_float, key=lambda k: [k[1], k[2]])
            # print(group_keys_float[:200])
            sorted_keys = [i[0] for i in group_keys_float]  # [:200]
            # print(sorted_keys)

            self.group_keys = [",".join(i) for i in self.group_keys]

            sorted_group_keys = list(itemgetter(*sorted_keys)(self.group_keys))
            # print("sorted_group_keys", sorted_group_keys)
            self.group_keys_chunk = [sorted_group_keys[i:i + n_per_group] for i in
                                     range(0, len(sorted_group_keys), n_per_group)]
            # print("*"*100)
            # print("self.group_keys_chunk", self.group_keys_chunk)

            groups_chunk = [pd.concat([grouped.get_group(tuple(
                grp.split(","))) for grp in sub_group]) for sub_group in self.group_keys_chunk]
            # groups_chunk = []
            # # groups_chunk = [pd.concat([grouped.get_group(grp.split(",")) for grp in sub_group])
            # #                 for sub_group in sub.split(",") for sub in self.group_keys_chunk]
            # for sub_group in self.group_keys_chunk:
            #     for grp in sub_group:
            #         print("*"*100)
            #         print(grp)
            #         print(grp.split(","))
            #         print(grouped.groups.keys())
            #         print(grouped.get_group(tuple(grp.split(","))))
            #         raise Exception
            # print("groups_chunk", groups_chunk)

            # print("self.group_keys_chunk", self.group_keys_chunk)
            # print("sorted_group_keys", sorted_group_keys[:20])

            # print(groups_chunk)

        else:  # 1d group by
            group_keys_float = []
            for key in self.group_keys:
                if key == "":
                    k = 0.0
                else:
                    try:
                        k = float(key)
                    except ValueError:
                        raise ValueError(
                            "ValueError: could not convert string to float in " + __file__)
                group_keys_float.append(k)

            # print("group_keys_float", group_keys_float[:20])
            self.group_keys = [k for _, k in sorted(
                zip(group_keys_float, self.group_keys))]
            self.group_keys_chunk = [self.group_keys[i:i + n_per_group] for i in
                                     range(0, len(self.group_keys), n_per_group)]
            groups_chunk = [pd.concat([grouped.get_group(grp) for grp in sub_group])
                            for sub_group in self.group_keys_chunk]
            # print("self.group_keys_chunk", self.group_keys_chunk)
            # print("groups_chunk")
            # print(groups_chunk)

        # raise Exception

        # self.group_keys_chunk = [self.group_keys[i:i + n_per_group] for i in
        #                          range(0, len(self.group_keys), n_per_group)]

        # print(self.group_keys_chunk)

        # groups_chunk = [pd.concat([grouped.get_group(grp) for grp in sub_group])
        #                 for sub_group in self.group_keys_chunk]
        # print("groups_chunk")
        # print(groups_chunk)

        # print(n_total_point)
        for index, [chunk_key, chunk_group] in enumerate(zip(self.group_keys_chunk, groups_chunk)):
            # print(index, chunk_key)
            # print("n_total_point", n_total_point)
            # print(index, chunk_group)

            n_total_point_chunk = {k: n_total_point[k]
                                   for k in n_total_point if k in chunk_key}
            # print(n_total_point_chunk)#, chunk_group,chunk_group.dtypes)
            # print("n_total_point_chunk", n_total_point_chunk)
            # raise Exception()
            print("Training network " + str(index) +
                  " for group " + str(chunk_key))
            # print("n_total_point_chunk", n_total_point_chunk)
            kdeModelWrapper = KdeModelTrainer(mdl, tbl, xheader, yheader, groupby_attribute=groupby_attribute,
                                              groupby_values=chunk_key,
                                              n_total_point=n_total_point_chunk,
                                              x_min_value=-np.inf, x_max_value=np.inf, config=self.config, device=self.device).fit_from_df(
                chunk_group, network_size="small", n_mdn_layer_node=n_mdn_layer_node,
                encoding=encoding, b_grid_search=b_grid_search)

            engine = MdnQueryEngine(kdeModelWrapper, config=self.config.copy())
            self.enginesContainer[index] = engine
        return self

    def predicts(self, func, x_lb, x_ub, n_jobs=4, ):
        # result2file=None, n_division=20, b_print_to_screen=True
        result2file = self.config.get_config()["result2file"]
        n_division = self.config.get_config()["n_division"]
        b_print_to_screen = self.config.get_config()["b_print_to_screen"]
        instances = []
        predictions = {}
        # times = {}
        with Pool(processes=n_jobs) as pool:
            # print(self.group_keys_chunk)
            for index, sub_group in enumerate(self.group_keys_chunk):
                # print(sub_group)
                engine = self.enginesContainer[index]
                i = pool.apply_async(
                    engine.predict_one_pass, (func, x_lb, x_ub, sub_group, False, n_division))
                instances.append(i)

            for i in instances:
                result = i.get()
                # pred = result[0]
                # t = result[1]
                predictions.update(result)
                # times.update(t)
        if b_print_to_screen:
            for key in predictions:
                print(key + "," + str(predictions[key]))
        if result2file is not None:
            # print(predictions)
            with open(result2file, 'w') as f:
                for key in predictions:
                    f.write(str(key) + "," + str(predictions[key]) + "\n")
        return predictions

    def init_pickle_file_name(self):
        # self.pickle_file_name = self.pickle_file_name
        return self.pickle_file_name + ".pkl"

    def serialize2warehouse(self, warehouse):
        if self.pickle_file_name is None:
            self.init_pickle_file_name()

        with open(warehouse + '/' + self.init_pickle_file_name(), 'wb') as f:
            dill.dump(self, f)


class MdnQueryEngineXCategorical:
    """ This class is the query engine for x with categorical attributes
    """

    def __init__(self, config):
        self.models = {}
        self.config = config
        self.mdl_name = None
        self.n_total_points = None
        self.x_categorical_columns = None
        self.categorical_distinct_values = None
        self.group_by_columns = None
        self.density_column = None

    # device: str, encoding="binary", b_grid_search=False

    def fit(self, mdl_name: str, origin_table_name: str, data: dict, total_points: dict, usecols: dict):
        if not total_points["if_contain_x_categorical"]:
            raise ValueError("The data provided is not a dict.")

        print("fit MdnQueryEngineXCategorical...")
        self.density_column = usecols["x_continous"][0]
        self.mdl_name = mdl_name
        self.n_total_points = total_points
        total_points.pop("if_contain_x_categorical")
        self.x_categorical_columns = total_points.pop("x_categorical_columns")
        self.categorical_distinct_values = total_points.pop(
            "categorical_distinct_values")
        # print("self.x_categorical_columns",
        #       self.x_categorical_columns)
        # del self.x_categorical_columns[self.density_column]
        self.group_by_columns = usecols['gb']

        # print("x_categorical_columns", self.x_categorical_columns)

        # configuration-related parameters.
        device = self.config.get_config()["device"]
        encoding = self.config.get_config()["encoding"]
        b_grid_search = self.config.get_config()["b_grid_search"]
        # print("total_points", total_points)
        idx = 0
        for categorical_attributes in total_points:
            print("start training  sub_model " +
                  str(idx) + " for "+mdl_name+"...")
            idx += 1
            kdeModelWrapper = KdeModelTrainer(
                mdl_name, origin_table_name, usecols["x_continous"][0], usecols["y"],
                groupby_attribute=usecols["gb"],
                groupby_values=list(
                    total_points[categorical_attributes].keys()),
                n_total_point=total_points[categorical_attributes],
                x_min_value=-np.inf, x_max_value=np.inf,
                config=self.config, device=device).fit_from_df(
                data[categorical_attributes], encoding=encoding, network_size="large", b_grid_search=b_grid_search, )

            qe_mdn = MdnQueryEngine(kdeModelWrapper, self.config.copy())
            self.models[categorical_attributes] = qe_mdn

        # kdeModelWrapper.serialize2warehouse(
        #     self.config['warehousedir'])
        # self.model_catalog.add_model_wrapper(
        #     kdeModelWrapper)

    # result2file=False,n_division=20
    def predicts(self, func, x_lb, x_ub, x_categorical_conditions,  n_jobs=1, filter_dbest=None):
        # print(self.models.keys())
        # print(x_categorical_conditions)
        x_categorical_conditions[2].pop(self.density_column)
        # configuration-related parameters.
        n_jobs = self.config.get_config()["n_jobs"]

        # check the condition when only one model is involved.
        cols = [item.lower() for item in x_categorical_conditions[0]]
        keys = [item for item in x_categorical_conditions[1]]
        if not x_categorical_conditions[2]:
            # prepare the key of the model.

            # print(self.x_categorical_columns)
            # print(keys)
            # print(cols)
            sorted_keys = [keys[cols.index(col)].replace("'", "")
                           for col in self.x_categorical_columns]
            key = ",".join(sorted_keys)

            self.models[key].config.set_parameter(
                "b_print_to_screen", False)

            # make the predictions
            predictions = self.models[key].predict_one_pass(func, x_lb=x_lb, x_ub=x_ub,
                                                            n_jobs=n_jobs, filter_dbest=filter_dbest)

        else:
            # prepare the keys of the models to be called.
            keys_list = []
            predictions = Counter({})
            for col in x_categorical_conditions[2]:
                # print(col, x_categorical_conditions[2][col])
                distinct_values = self.categorical_distinct_values[col.lower()]
                for value in distinct_values:
                    if meet_condition(value, x_categorical_conditions[2][col]):
                        key = []
                        for key_item in self.x_categorical_columns:
                            # print(key_item)
                            if key_item in cols:
                                key.append(
                                    keys[cols.index(key_item)].replace("'", ""))
                            else:
                                key.append(value)
                        # print(key)
                        key = ",".join(key)

                        # make the predictions
                        self.models[key].config.set_parameter(
                            "b_print_to_screen", False)
                        pred = self.models[key].predict_one_pass(func, x_lb=x_lb, x_ub=x_ub,
                                                                 n_jobs=n_jobs, filter_dbest=filter_dbest)
                        predictions = predictions + Counter(pred)
                        keys_list.append(key)

        # print("preditions,", predictions)

        if self.config.get_config()["b_print_to_screen"]:
            headers = list(self.group_by_columns)
            headers.append("value")
            print(" ".join(headers))

            for pred in predictions:
                print(pred, predictions[pred])

            # print(keys_list)
            # print(predictions)

            # print("need to get predictions from multiple models.")

    def serialize2warehouse(self, warehouse):
        with open(warehouse + '/' + self.mdl_name + '.pkl', 'wb') as f:
            dill.dump(self, f)

    def init_pickle_file_name(self):
        return self.mdl_name+".pkl"


def meet_condition(value: str, condition):
    value = float(value)
    # check x greater than
    if condition[0] is None:
        b1 = True
    else:
        if condition[2]:
            b1 = True if value >= float(condition[0]) else False
        else:
            b1 = True if value > float(condition[0]) else False
    # check x less than
    if condition[1] is None:
        b2 = True
    else:
        if condition[3]:
            b2 = True if value <= float(condition[1]) else False
        else:
            b2 = True if value < float(condition[1]) else False

    return b1 and b2


# if __name__ == "__main__":
#     print(meet_condition("2", [0.0, 1, False, False]))
    #     config = {
    #         'warehousedir': '/home/u1796377/Programs/dbestwarehouse',
    #         'verbose': 'True',
    #         'b_show_latency': 'True',
    #         'backend_server': 'None',
    #         'csv_split_char': ',',
    #         "epsabs": 10.0,
    #         "epsrel": 0.1,
    #         "mesh_grid_num": 20,
    #         "limit": 30,
    #         # "b_reg_mean":'True',
    #         "num_epoch": 400,
    #         "reg_type": "mdn",
    #         "density_type": "density_type",
    #         "num_gaussians": 4,
    #     }

    #     headers = ["ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk", "ss_cdemo_sk", "ss_hdemo_sk",
    #                "ss_addr_sk", "ss_store_sk", "ss_promo_sk", "ss_ticket_number", "ss_quantity", "ss_wholesale_cost",
    #                "ss_list_price", "ss_sales_price", "ss_ext_discount_amt", "ss_ext_sales_price",
    #                "ss_ext_wholesale_cost", "ss_ext_list_price", "ss_ext_tax", "ss_coupon_amt", "ss_net_paid",
    #                "ss_net_paid_inc_tax", "ss_net_profit", "none"]
    #     groupby_attribute = "ss_store_sk"
    #     xheader = "ss_wholesale_cost"
    #     yheader = "ss_list_price"

    #     sampler = DBEstSampling(headers=headers, usecols=[
    #         xheader, yheader, groupby_attribute])
    #     total_count = {'total': 2879987999}
    #     original_data_file = "/data/tpcds/40G/ss_600k_headers.csv"

    #     sampler.make_sample(original_data_file, 60000, "uniform", split_char="|",
    #                         num_total_records=total_count)
    #     xyzs = sampler.getyx(yheader, xheader, groupby=groupby_attribute)
    #     n_total_point = get_group_count_from_summary_file(
    #         config['warehousedir'] + "/num_of_points57.txt", sep=',')

    #     bundles = MdnQueryEngineBundle(config=config, device="cpu")
    #     bundles.fit(xyzs, groupby_attribute, n_total_point, "mdl", "tbl",
    #                 xheader, yheader, n_per_group=30, b_grid_search=False,)

    #     bundles.predicts("count", 2451119, 2451483)
