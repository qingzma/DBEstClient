# Created by Qingzhi Ma at 29/01/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

import math
from collections import Counter
from datetime import datetime
from multiprocessing import Pool as PoolCPU
from multiprocessing.pool import ThreadPool
from operator import itemgetter

import dill
import numpy as np
import pandas as pd
from scipy import integrate
from torch.multiprocessing import Pool as PoolGPU

# from dbestclient.io.sampling import DBEstSampling
from dbestclient.ml.integral import (approx_avg, approx_count,
                                     approx_integrate, approx_sum,
                                     prepare_reg_density_data, prepare_var)
from dbestclient.ml.mdn import KdeMdn, RegMdnGroupBy
from dbestclient.ml.modeltrainer import KdeModelTrainer
from dbestclient.socket import app_client
from dbestclient.tools.running_parameters import shrink_runtime_config

# from torch.multiprocessing import set_start_method


# from torch.multiprocessing import Pool, set_start_method


# from dbestclient.tools.dftools import get_group_count_from_summary_file

# try:
#     set_start_method('spawn')
# except RuntimeError:
#     print("Fail to set start method as spawn for pytorch multiprocessing, " +
#           "use default in advance. (see queryenginemdn "
#           "for more info.)")


class GenericQueryEngine:
    def __init__(self):
        self.mdl_name = None

    def serialize2warehouse(self, warehouse, runtime_config):
        with open(warehouse + '/' + self.mdl_name + runtime_config["model_suffix"], 'wb') as f:
            dill.dump(self, f)

    def init_pickle_file_name(self, runtime_config):
        return self.mdl_name+runtime_config["model_suffix"]

    def fit(self, mdl_name: str, origin_table_name: str, data: dict, total_points: dict, usecols: dict, runtime_config: dict):
        pass

    def predicts(self, func: str, x_lb: float, x_ub: float, x_categorical_conditions, runtime_config, groups: list = None, filter_dbest=None):
        pass


class MdnQueryEngine(GenericQueryEngine):
    t_before_multiple_processing = None

    def __init__(self, kdeModelWrapper, config):
        super().__init__()
        # self.n_training_point = kdeModelWrapper.n_sample_point
        self.n_total_point = kdeModelWrapper.n_total_point
        self.reg = kdeModelWrapper.reg
        self.kde = kdeModelWrapper.density
        self.x_min = kdeModelWrapper.x_min_value
        self.x_max = kdeModelWrapper.x_max_value
        self.groupby_attribute = kdeModelWrapper.groupby_attribute
        self.groupby_values = kdeModelWrapper.groupby_values
        self.density_column = kdeModelWrapper.x
        self.mdl_name = kdeModelWrapper.mdl

        self.config = config
        # self.b_use_integral = runtime_config["b_use_integral"]

    def approx_avg(self, x_min, x_max, groupby_value, runtime_config):
        start = datetime.now()

        def f_pRx(*args):
            return self.kde.predict([[groupby_value]], args[0], b_plot=False) \
                * self.reg.predict(np.array([[args[0], groupby_value]]))[0]

        def f_p(*args):
            return self.kde.predict([[groupby_value]], args[0], b_plot=False)

        if runtime_config["b_use_integral"]:
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

        if runtime_config['b_show_latency']:
            end = datetime.now()
            time_cost = (end - start).total_seconds()
        return result, time_cost

    def approx_sum(self, x_min, x_max, groupby_value, runtime_config):
        start = datetime.now()

        def f_pRx(*args):
            return self.kde.predict([[groupby_value]], args[0], b_plot=True) \
                * self.reg.predict(np.array([[args[0], groupby_value]]))[0]

        if runtime_config["b_use_integral"]:
            result = integrate.quad(f_pRx, x_min, x_max, epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[
                0] * float(self.n_total_point[str(int(groupby_value))])
        else:
            result = approx_integrate(
                f_pRx, x_min, x_max) * float(self.n_total_point[str(int(groupby_value))])

        if runtime_config['b_show_latency'] and result != None:
            end = datetime.now()
            time_cost = (end - start).total_seconds()
        return result, time_cost

    def approx_count(self, x_min, x_max, groupby_value, runtime_config):
        start = datetime.now()

        def f_p(*args):
            return self.kde.predict([[groupby_value]], args[0], b_plot=False)
            # return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

        if runtime_config["b_use_integral"]:
            result = integrate.quad(
                f_p, x_min, x_max, epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[0]
        else:
            result = approx_integrate(f_p, x_min, x_max)
        result = result * float(self.n_total_point[str(int(groupby_value))])

        if runtime_config['b_show_latency'] and result != None:
            end = datetime.now()
            time_cost = (end - start).total_seconds()

        return result, time_cost
    
    def approx_var(self,runtime_config):
        start = datetime.now()
        print("within approx_var!")

        result = 9999999.9999
        if runtime_config['b_show_latency'] and result != None:
            end = datetime.now()
            time_cost = (end - start).total_seconds()

        return result, time_cost

    def predict(self, func, x_lb=None, x_ub=None, groupby_value=None, runtime_config=None):
        if func.lower() == "count":
            p, t = self.approx_count(x_lb, x_ub, groupby_value, runtime_config)
        elif func.lower() == "sum":
            p, t = self.approx_sum(x_lb, x_ub, groupby_value, runtime_config)
        elif func.lower() == "avg":
            p, t = self.approx_avg(x_lb, x_ub, groupby_value, runtime_config)
        elif func.lower() == "var":
            p, t = self.approx_var(runtime_config=runtime_config)
        else:
            print("Aggregate function " + func + " is not implemented yet!")
        return p, t
    
    

    # def predicts(self, func, x_lb, x_ub, b_parallel=True, n_jobs=4, filter_dbest=None):  # result2file=None
    #     result2file = self.config.get_config()["result2file"]
    #     predictions = {}
    #     times = {}
    #     if not b_parallel:  # single process implementation
    #         for groupby_value in self.groupby_values:
    #             if groupby_value == "":
    #                 continue
    #             print("func, x_lb, x_ub, groupby_value",
    #                   func, x_lb, x_ub, groupby_value)
    #             pre, t = self.predict(func, x_lb, x_ub, groupby_value)
    #             predictions[groupby_value] = pre
    #             times[groupby_value] = t
    #             # print(groupby_value, pre)
    #     else:  # multiple threads implementation

    #         width = int(len(self.groupby_values) / n_jobs)
    #         subgroups = [self.groupby_values[inde:inde + width]
    #                      for inde in range(0, len(self.groupby_values), width)]
    #         if len(self.groupby_values) % n_jobs != 0:
    #             subgroups[n_jobs - 1] = subgroups[n_jobs - 1] + \
    #                 subgroups[n_jobs]
    #             del subgroups[-1]
    #         # index_in_groups = [[self.groupby_values.index(sgname) for sgname in sgnames] for sgnames in subgroups]

    #         instances = []

    #         with Pool(processes=n_jobs) as pool:
    #             for subgroup in subgroups:
    #                 i = pool.apply_async(
    #                     query_partial_group, (self, subgroup, func, x_lb, x_ub))
    #                 instances.append(i)
    #                 # print(i.get())
    #                 # print(instances[0].get(timeout=1))
    #             for i in instances:
    #                 result = i.get()
    #                 pred = result[0]
    #                 t = result[1]
    #                 predictions.update(pred)
    #                 times.update(t)
    #                 # predictions += pred
    #                 # times += t
    #     if result2file is not None:
    #         with open(result2file, 'w') as f:
    #             for key in predictions:
    #                 f.write(key + "," + str(predictions[key]))
    #     return predictions, times

    def predicts(self, func: str, x_lb: float=None, x_ub: float=None,  x_categorical_conditions=None, runtime_config=None, groups: list = None, filter_dbest=None, time2exclude_from_multiprocessing=None):
        if time2exclude_from_multiprocessing is not None:
            t_after_multiple_processing = datetime.now()
            print("should reduce time {} from the query response time.".format(
                (t_after_multiple_processing - time2exclude_from_multiprocessing).total_seconds()))
        b_print_to_screen = runtime_config["b_print_to_screen"]
        # n_division = runtime_config["n_division"]
        result2file = runtime_config["result2file"]
        if "slaves" in runtime_config:
            if runtime_config["slaves"].size() == 0:
                n_jobs = runtime_config["n_jobs"]
            else:
                n_jobs = runtime_config["slaves"].size()
        else:
            n_jobs = runtime_config["n_jobs"]
        # result2file = self.config.get_config()["result2file"]

        if func.lower() not in ("count", "sum", "avg","var"):
            raise ValueError("function not supported: "+func)
        if groups is None:  # provide predictions for all groups.
            groups = self.groupby_values

        if self.config.get_config()["accept_filter"]:
            results = self.n_total_point
            # print("results,", results)
            # print("filter_dbest", filter_dbest)
            results = {key: results[key] for key in results if float(
                key) >= filter_dbest[0] and float(key) <= filter_dbest[1]}

            # if func.lower() not in ("count", "sum"):
            #     # scale up the result
            #     scaling_factor = self.config.get_config()["scaling_factor"]
            #     # print("scaling_factor", scaling_factor)
            #     results = {key: results[key]*scaling_factor for key in results}

            # print("self.n_total_point", self.n_total_point)
            if b_print_to_screen:
                for key in results:
                    print(",".join(key.split("-")) +
                          "," + str(results[key]))

            if result2file is not None:
                with open(result2file, 'w') as f:
                    for key in results:
                        f.write(str(key) + "," + str(results[key]) + "\n")
            return results

        if func.lower() in ["count", "sum", "avg"]:
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
                # print("self.n_total_point", self.n_total_point)
                pre_density, pre_reg, step = prepare_reg_density_data(
                    self.kde, x_lb, x_ub, groups=groups, reg=self.reg, runtime_config=runtime_config)
                # print("pre_density, pre_reg",pre_density,)
                # print(pre_reg)

                if func.lower() == "count":
                    preds = approx_count(pre_density, step)
                    preds = np.multiply(preds, scaling_factor)
                elif func.lower() == "sum":
                    preds = approx_sum(pre_density, pre_reg, step)
                    preds = np.multiply(preds, scaling_factor)
                elif func.lower() == "avg":  # avg
                    preds = approx_avg(pre_density, pre_reg, step)
                else:
                    raise TypeError("wrong aggregate!")
                results = dict(zip(groups, preds))

            else:
                runtime_config_process = shrink_runtime_config(
                    runtime_config)
                # use multi-processing to achieve parallel
                if runtime_config["slaves"].is_empty():
                    instances = []
                    results = {}
                    # print("n_jobs,", n_jobs)
                    n_per_chunk = math.ceil(len(groups)/n_jobs)
                    group_chunks = [groups[i:i+n_per_chunk]
                                    for i in range(0, len(groups), n_per_chunk)]
                    # print("create Pool...", datetime.now())
                    if runtime_config["device"] == "cpu":
                        pool = PoolCPU(processes=n_jobs)
                    else:
                        pool = PoolGPU(processes=n_jobs)
                        # from torch.multiprocessing import Pool, set_start_method
                        # try:
                        #     set_start_method('spawn')
                        # except RuntimeError:
                        #     print("Fail to set start method as spawn for pytorch multiprocessing, " +
                        #           "use default in advance. (see queryenginemdn "
                        #           "for more info.)")

                    # with Pool(processes=n_jobs) as pool:
                    # print(self.group_keys_chunk)

                    time2exclude_from_multiprocessing = datetime.now()

                    for sub_group in group_chunks:

                        i = pool.apply_async(
                            self.predicts, (func, x_lb, x_ub, x_categorical_conditions, runtime_config_process, sub_group, filter_dbest, time2exclude_from_multiprocessing))
                        instances.append(i)

                    for i in instances:
                        result = i.get()
                        results.update(result)
                else:  # slaves are used
                    slaves = runtime_config["slaves"]
                    instances = []
                    results = {}
                    n_jobs = slaves.size()
                    n_per_chunk = math.ceil(len(groups)/n_jobs)
                    group_chunks = [groups[i:i+n_per_chunk]
                                    for i in range(0, len(groups), n_per_chunk)]

                    pool = ThreadPool(processes=n_jobs)
                    hosts = slaves.get()
                    for sub_group, host in zip(group_chunks, hosts):
                        # print("host", host)
                        query = dict(func=func, x_lb=x_lb, x_ub=x_ub, x_categorical_conditions=x_categorical_conditions, runtime_config=runtime_config_process,
                                    sub_group=sub_group, filter_dbest=filter_dbest, mdl_name=self.mdl_name+runtime_config["model_suffix"])
                        i = pool.apply_async(
                            app_client.run, (hosts[host].host, hosts[host].port, "select", query))
                        instances.append(i)

                    for i in instances:
                        result = i.get()
                        results.update(result)
                        # result = app_client.run(
                        #     host, slaves.get()[host], "select", query)
        
        elif func.lower() == "var":
            print("predict var")

            results= prepare_var(self.kde, groups=groups, runtime_config=runtime_config)#{"group":999.99}
        else:
            raise TypeError("unexpected aggregated.")
        runtime_config["b_print_to_screen"] = b_print_to_screen
        if runtime_config["b_print_to_screen"]:
            for key in results:
                print(",".join(key.split("-")) +
                      "," + str(results[key]))

        if result2file is not None:
            with open(result2file, 'w') as f:
                for key in results:
                    f.write(str(key) + "," + str(results[key]) + "\n")
        return results

    def serialize2warehouse(self, warehouse, runtime_config):
        with open(warehouse + '/' + self.mdl_name + runtime_config["model_suffix"], 'wb') as f:
            dill.dump(self, f)

    def init_pickle_file_name(self, runtime_config):
        return self.mdl_name+runtime_config["model_suffix"]


def query_partial_group(mdnQueryEngine, group, func, x_lb, x_ub):
    mdnQueryEngine.groupby_values = group
    return mdnQueryEngine.predicts(func, x_lb, x_ub, b_parallel=False, n_jobs=1)


class MdnQueryEngineGoGs():
    # __init__(self, config, device):
    def __init__(self, config):
        self.enginesContainer = {}
        self.config = config
        self.n_total_point = None
        self.group_keys_chunk = None
        self.group_keys = None
        self.pickle_file_name = None
        self.density_column = None

    def fit(self, df: pd.DataFrame, groupby_attribute: str, n_total_point: dict,
            mdl: str, tbl: str, xheader: str, yheader: str, runtime_config: dict):  # n_per_group: int = 10, n_mdn_layer_node=10, encoding = "onehot", b_grid_search = True
        # configuration-related parameters.
        n_per_group = self.config.get_config()["n_per_gg"]
        n_mdn_layer_node = self.config.get_config()["n_mdn_layer_node_reg"]
        encoding = self.config.get_config()["encoder"]
        b_grid_search = self.config.get_config()["b_grid_search"]

        self.density_column = xheader

        self.pickle_file_name = mdl
        # print(groupby_attribute)
        # print(df)

        # df = df["data"]
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
                                              x_min_value=-np.inf, x_max_value=np.inf, config=self.config).fit_from_df(
                chunk_group, network_size="small", runtime_config=runtime_config)

            engine = MdnQueryEngine(
                kdeModelWrapper, config=self.config.copy())
            self.enginesContainer[index] = engine
        return self

    def predicts(self, func, x_lb, x_ub, x_categorical_conditions, runtime_config, groups: list = None, filter_dbest=None):
        # result2file=None, n_division=20, b_print_to_screen=True
        result2file = runtime_config["result2file"]
        n_division = runtime_config["n_division"]
        b_print_to_screen = runtime_config["b_print_to_screen"]
        n_jobs = runtime_config["n_jobs"]
        instances = []
        predictions = {}
        # times = {}
        if runtime_config["device"] == "cpu":
            pool = PoolCPU(processes=n_jobs)
        else:
            pool = PoolGPU(processes=n_jobs)
        # with Pool(processes=n_jobs) as pool:
        # print(self.group_keys_chunk)
        for index, sub_group in enumerate(self.group_keys_chunk):
            # print(sub_group)
            engine = self.enginesContainer[index]
            runtime_config["b_print_to_screen"] = False
            i = pool.apply_async(
                engine.predicts, (func, x_lb, x_ub, x_categorical_conditions, runtime_config, sub_group, filter_dbest))
            instances.append(i)

        for i in instances:
            result = i.get()
            # pred = result[0]
            # t = result[1]
            predictions.update(result)
            # times.update(t)
        runtime_config["b_print_to_screen"] = b_print_to_screen
        if b_print_to_screen:
            for key in predictions:
                print(key + "," + str(predictions[key]))
        if result2file is not None:
            # print(predictions)
            with open(result2file, 'w') as f:
                for key in predictions:
                    f.write(str(key) + "," + str(predictions[key]) + "\n")
        return predictions

    def init_pickle_file_name(self, runtime_config):
        # self.pickle_file_name = self.pickle_file_name
        return self.pickle_file_name + runtime_config["model_suffix"]

    def serialize2warehouse(self, warehouse, runtime_config):
        if self.pickle_file_name is None:
            self.init_pickle_file_name(runtime_config)

        with open(warehouse + '/' + self.init_pickle_file_name(runtime_config), 'wb') as f:
            dill.dump(self, f)


class MdnQueryEngineXCategorical(GenericQueryEngine):
    """ This class is the query engine for x with categorical attributes
    """

    def __init__(self, config):
        super().__init__()
        self.models = {}
        self.config = config
        self.mdl_name = None
        self.n_total_points = None
        self.x_categorical_columns = None
        self.categorical_distinct_values = None
        self.group_by_columns = None
        self.density_column = None

    # device: str, encoding="binary", b_grid_search=False

    def fit(self, mdl_name: str, origin_table_name: str, data: dict, total_points: dict, usecols: dict, runtime_config: dict):
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
        device = runtime_config["device"]
        encoding = self.config.get_config()["encoder"]
        b_grid_search = self.config.get_config()["b_grid_search"]
        # print("total_points", total_points)
        idx = 0
        for categorical_attributes in total_points:
            print("start training  sub_model " +
                  str(idx) + " for "+mdl_name+"...")
            idx += 1
            # print("total_points", total_points)
            # print(list(total_points[categorical_attributes].keys()))
            # GoG is not used, use kdeModelTrainer instead.
            if not self.config.get_config()["b_use_gg"]:
                kdeModelWrapper = KdeModelTrainer(
                    mdl_name, origin_table_name, usecols["x_continous"][0], usecols["y"],
                    groupby_attribute=usecols["gb"],
                    groupby_values=list(
                        total_points[categorical_attributes].keys()),
                    n_total_point=total_points[categorical_attributes],
                    x_min_value=-np.inf, x_max_value=np.inf,
                    config=self.config).fit_from_df(
                    # data[categorical_attributes]["data"], runtime_config=runtime_config, network_size="large",)
                    data[categorical_attributes], runtime_config=runtime_config, network_size="large",)

                qe_mdn = MdnQueryEngine(
                    kdeModelWrapper, self.config.copy())
            else:  # use GoGs
                qe_mdn = MdnQueryEngineGoGs(
                    config=self.config.copy()).fit(data[categorical_attributes], usecols["gb"],
                                                   total_points[categorical_attributes], mdl_name, origin_table_name,
                                                   usecols["x_continous"][0], usecols["y"],
                                                   runtime_config)

            self.models[categorical_attributes] = qe_mdn

        # kdeModelWrapper.serialize2warehouse(
        #     self.config['warehousedir'])
        # self.model_catalog.add_model_wrapper(
        #     kdeModelWrapper)

    # result2file=False,n_division=20
    def predicts(self, func, x_lb, x_ub, x_categorical_conditions, runtime_config, groups: list = None, filter_dbest=None,):
        # print(self.models.keys())
        # print(x_categorical_conditions)
        b_print_to_screen = runtime_config["b_print_to_screen"]
        x_categorical_conditions[2].pop(self.density_column)
        # configuration-related parameters.
        # n_jobs = runtime_config["n_jobs"]

        # check the condition when only one model is involved.
        cols = [item.lower() for item in x_categorical_conditions[0]]
        keys = [item for item in x_categorical_conditions[1]]
        if not x_categorical_conditions[2]:
            print("enter prediction X 1 model...", datetime.now())
            # prepare the key of the model.

            # print(self.x_categorical_columns)
            # print(keys)
            # print(cols)
            sorted_keys = [keys[cols.index(col)].replace("'", "")
                           for col in self.x_categorical_columns]
            key = ",".join(sorted_keys)

            runtime_config["b_print_to_screen"] = False

            # make the predictions
            predictions = self.models[key].predicts(
                func, x_lb=x_lb, x_ub=x_ub, x_categorical_conditions=x_categorical_conditions, runtime_config=runtime_config, filter_dbest=filter_dbest)

        else:
            # prepare the keys of the models to be called.
            print("enter prediction X multiple models...", datetime.now())
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
                        runtime_config["b_print_to_screen"] = False
                        print("start entering multi-processing", datetime.now())
                        pred = self.models[key].predicts(func, x_lb=x_lb, x_ub=x_ub, x_categorical_conditions=x_categorical_conditions,
                                                         runtime_config=runtime_config, filter_dbest=filter_dbest)
                        predictions = predictions + Counter(pred)
                        keys_list.append(key)
            predictions = dict(predictions)
        # print("preditions,", predictions)
        # restore b_print_to_screen
        runtime_config["b_print_to_screen"] = b_print_to_screen
        if b_print_to_screen:
            headers = list(self.group_by_columns)
            headers.append("value")
            print(" ".join(headers))

            for pred in predictions:
                print(pred, predictions[pred])

            # print(keys_list)
            # print(predictions)

            # print("need to get predictions from multiple models.")

        return predictions

    def serialize2warehouse(self, warehouse, runtime_config):
        with open(warehouse + '/' + self.mdl_name + runtime_config["model_suffix"], 'wb') as f:
            dill.dump(self, f)

    def init_pickle_file_name(self, runtime_config):
        return self.mdl_name+runtime_config["model_suffix"]


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


class MdnQueryEngineXCategoricalOneModel(GenericQueryEngine):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_total_points = None
        self.group_by_columns = None
        self.density_column = None

    def serialize2warehouse(self, warehouse, runtime_config):
        with open(warehouse + '/' + self.mdl_name + runtime_config["model_suffix"], 'wb') as f:
            dill.dump(self, f)

    def init_pickle_file_name(self, runtime_config):
        return self.mdl_name+runtime_config["model_suffix"]

    def fit(self, mdl_name: str, origin_table_name: str, gbs, xs, ys, total_points: dict, usecols: dict, runtime_config: dict):
        # if not total_points["if_contain_x_categorical"]:
        #     raise ValueError("The data provided is not a dict.")
        if runtime_config['v']:
            print("fit MdnQueryEngineXCategoricalOneModel...")

        self.density_column = usecols["x_continous"][0]
        self.mdl_name = mdl_name
        self.n_total_points = total_points
        # total_points.pop("if_contain_x_categorical")
        self.x_categorical_columns = usecols["x_categorical"]
        # self.categorical_distinct_values = total_points.pop(
        #     "categorical_distinct_values")
        self.group_by_columns = usecols['gb']

        # configuration-related parameters.
        device = runtime_config["device"]
        encoding = self.config.get_config()["encoder"]
        b_grid_search = self.config.get_config()["b_grid_search"]

        if self.config.config['b_use_gg']:
            raise ValueError("Method not implemented.")
        else:
            config = self.config.copy()

            if runtime_config['v']:
                print("training density...")
            # print("usecols", usecols)
            self.density = KdeMdn(config, b_store_training_data=False).fit(
                gbs, xs, runtime_config)

            if runtime_config['v']:
                print("training regression...")
            self.reg = RegMdnGroupBy(config, b_store_training_data=False).fit(
                gbs, xs, ys, runtime_config)
            # kdeModelWrapper = KdeModelTrainer(
            #     mdl_name, origin_table_name, usecols["x_continous"][0], usecols["y"],
            #     groupby_attribute=usecols["gb"],
            #     groupby_values=list(
            #         total_points[categorical_attributes].keys()),
            #     n_total_point=total_points[categorical_attributes],
            #     x_min_value=-np.inf, x_max_value=np.inf,
            #     config=self.config).fit_from_df(
            #         data[categorical_attributes], runtime_config=runtime_config)

            # qe_mdn = MdnQueryEngine(
            #     kdeModelWrapper, self.config.copy())

    def predicts(self, func: str, x_lb: float, x_ub: float, x_categorical_conditions, runtime_config, groups: list = None, filter_dbest=None):
        if "slaves" in runtime_config:
            if runtime_config["slaves"].size() == 0:
                n_jobs = runtime_config["n_jobs"]
            else:
                n_jobs = runtime_config["slaves"].size()
        else:
            n_jobs = runtime_config["n_jobs"]
        # result2file = self.config.get_config()["result2file"]
        b_print_to_screen = runtime_config["b_print_to_screen"]
        result2file = runtime_config["result2file"]

        if func.lower() not in ("count", "sum", "avg"):
            raise ValueError("function not supported: "+func)

        # print("x_categorical_conditions", x_categorical_conditions)

        if len(x_categorical_conditions[1]) > 1:
            key = ",".join(x_categorical_conditions[1]).replace("'", "")
        else:
            key = x_categorical_conditions[1][0].replace("'", "")
        # print("key", key)
        # print("self.n_total_points", self.n_total_points)

        groups_no_categorical = list(self.n_total_points[key].keys())

        groups = [[item]+x_categorical_conditions[1]
                  for item in groups_no_categorical]
        groups = [','.join(g).replace("'", "") for g in groups]
        # print("groups", groups)

        if n_jobs == 1:
            # print(groups)
            # print(self.n_total_point)
            # print(groups[0], "*******************")
            # print(",".join(groups[0]))
            # for key in groups:
            # print("key is ", key, end="----- ")
            # print(",".join(key))

            # print(self.n_total_points)

            if len(groups[0].split(",")) == 1:  # 1d group by
                # print(groups[0].split(","))
                # print("1d")
                scaling_factor = np.array([self.n_total_points[key][k]
                                           for k in groups_no_categorical])
            else:
                # print(groups[0].split(","))
                # print("n-d")
                scaling_factor = np.array([self.n_total_points[key][k]
                                           for k in groups_no_categorical])
            # print("SF",self.n_total_points[key])
            # for item in self.n_total_points[key]:
            #     print(item, self.n_total_points[key][item])
            # print("scaling_factor",scaling_factor)
            # print("self.n_total_point", self.n_total_point)
            pre_density, pre_reg, step = prepare_reg_density_data(
                self.density, x_lb, x_ub, groups=groups, reg=self.reg, runtime_config=runtime_config)
            # print("pre_density, pre_reg",pre_density,)
            # print(pre_reg)

            if func.lower() == "count":
                preds = approx_count(pre_density, step)
                preds = np.multiply(preds, scaling_factor)
            elif func.lower() == "sum":
                preds = approx_sum(pre_density, pre_reg, step)
                preds = np.multiply(preds, scaling_factor)
            else:  # avg
                preds = approx_avg(pre_density, pre_reg, step)
            # print("groups-------------", groups)
            results = dict(zip(groups_no_categorical, preds))

        else:
            runtime_config_process = shrink_runtime_config(
                runtime_config)
            # use multi-processing to achieve parallel
            if runtime_config["slaves"].is_empty():
                instances = []
                results = {}
                # print("n_jobs,", n_jobs)
                n_per_chunk = math.ceil(len(groups)/n_jobs)
                group_chunks = [groups[i:i+n_per_chunk]
                                for i in range(0, len(groups), n_per_chunk)]
                # print("create Pool...", datetime.now())
                if runtime_config["device"] == "cpu":
                    pool = PoolCPU(processes=n_jobs)
                else:
                    pool = PoolGPU(processes=n_jobs)
                    # from torch.multiprocessing import Pool, set_start_method
                    # try:
                    #     set_start_method('spawn')
                    # except RuntimeError:
                    #     print("Fail to set start method as spawn for pytorch multiprocessing, " +
                    #           "use default in advance. (see queryenginemdn "
                    #           "for more info.)")

                # with Pool(processes=n_jobs) as pool:
                # print(self.group_keys_chunk)

                time2exclude_from_multiprocessing = datetime.now()

                for sub_group in group_chunks:

                    i = pool.apply_async(
                        self.predicts, (func, x_lb, x_ub, x_categorical_conditions, runtime_config_process, sub_group, filter_dbest, time2exclude_from_multiprocessing))
                    instances.append(i)

                for i in instances:
                    result = i.get()
                    results.update(result)
            else:  # slaves are used
                slaves = runtime_config["slaves"]
                instances = []
                results = {}
                n_jobs = slaves.size()
                n_per_chunk = math.ceil(len(groups)/n_jobs)
                group_chunks = [groups[i:i+n_per_chunk]
                                for i in range(0, len(groups), n_per_chunk)]

                pool = ThreadPool(processes=n_jobs)
                hosts = slaves.get()
                for sub_group, host in zip(group_chunks, hosts):
                    # print("host", host)
                    query = dict(func=func, x_lb=x_lb, x_ub=x_ub, x_categorical_conditions=x_categorical_conditions, runtime_config=runtime_config_process,
                                 sub_group=sub_group, filter_dbest=filter_dbest, mdl_name=self.mdl_name+runtime_config["model_suffix"])
                    i = pool.apply_async(
                        app_client.run, (hosts[host].host, hosts[host].port, "select", query))
                    instances.append(i)

                for i in instances:
                    result = i.get()
                    results.update(result)
                    # result = app_client.run(
                    #     host, slaves.get()[host], "select", query)
        runtime_config["b_print_to_screen"] = b_print_to_screen
        if runtime_config["b_print_to_screen"]:
            for key in results:
                print(",".join(key.split("-")) +
                      "," + str(results[key]))

        if result2file is not None:
            with open(result2file, 'w') as f:
                for key in results:
                    f.write(str(key) + "," + str(results[key]) + "\n")
        return results
