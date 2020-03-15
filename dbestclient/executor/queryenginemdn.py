# Created by Qingzhi Ma at 29/01/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

from datetime import datetime

import dill
import numpy as np
import pandas as pd
from scipy import integrate
from torch.multiprocessing import Pool, set_start_method

from dbestclient.io.sampling import DBEstSampling
from dbestclient.ml.integral import approx_integrate
from dbestclient.ml.modeltrainer import KdeModelTrainer
from dbestclient.tools.dftools import get_group_count_from_summary_file

try:
    set_start_method('spawn')
except RuntimeError:
    print("Fail to set start method as spawn for pytorch multiprocessing, " +
          "use default in advance. (see queryenginemdn "
          "for more info.)")


class MdnQueryEngine:
    def __init__(self, kdeModelWrapper, config=None, b_use_integral=False):
        self.n_training_point = kdeModelWrapper.n_sample_point
        self.n_total_point = kdeModelWrapper.n_total_point
        self.reg = kdeModelWrapper.reg
        self.kde = kdeModelWrapper.density
        self.x_min = kdeModelWrapper.x_min_value
        self.x_max = kdeModelWrapper.x_max_value
        self.groupby_attribute = kdeModelWrapper.groupby_attribute
        self.groupby_values = kdeModelWrapper.groupby_values
        if config is None:
            self.config = config = {
                'warehousedir': 'dbestwarehouse',
                'verbose': 'True',
                'b_show_latency': 'True',
                'backend_server': 'None',
                'epsabs': 10.0,
                'epsrel': 0.1,
                'mesh_grid_num': 20,
                'limit': 30,
                'csv_split_char': '|',
                'num_epoch': 400,
                "reg_type": "mdn",
            }
        else:
            self.config = config
        self.b_use_integral = b_use_integral

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

    def predicts(self, func, x_lb, x_ub, b_parallel=True, n_jobs=4, result2file=None):
        predictions = {}
        times = {}
        if not b_parallel:  # single process implementation
            for groupby_value in self.groupby_values:
                if groupby_value == "":
                    continue
                pre, t = self.predict(func, x_lb, x_ub, groupby_value)
                predictions[groupby_value] = pre
                times[groupby_value] = t
                print(groupby_value, pre)
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


def query_partial_group(mdnQueryEngine, group, func, x_lb, x_ub):
    mdnQueryEngine.groupby_values = group
    return mdnQueryEngine.predicts(func, x_lb, x_ub, b_parallel=False, n_jobs=1)


class MdnQueryEngineBundle():
    def __init__(self, config: dict):
        self.enginesContainer = {}
        self.config = config
        self.n_total_point = None
        self.group_keys_chunk = None
        self.group_keys = None
        self.pickle_file_name = None

    def fit(self, df: pd.DataFrame, groupby_attribute: str, n_total_point: dict,
            mdl: str, tbl: str, xheader: str, yheader: str, n_per_group: int = 10, n_mdn_layer_node=10,
            b_one_hot_encoding=True, b_grid_search=True):

        self.pickle_file_name = mdl
        grouped = df.groupby(groupby_attribute)

        self.group_keys = list(grouped.groups.keys())

        # sort the group key by value
        fakeKey = []
        for key in self.group_keys:
            if key == "":
                k = 0.0
            else:
                try:
                    k = float(key)
                except ValueError:
                    raise ValueError(
                        "ValueError: could not convert string to float in " + __file__)
            fakeKey.append(k)

        self.group_keys = [k for _, k in sorted(zip(fakeKey, self.group_keys))]
        # print(self.group_keys)

        self.group_keys_chunk = [self.group_keys[i:i + n_per_group] for i in
                                 range(0, len(self.group_keys), n_per_group)]
        # print(self.group_keys_chunk)

        groups_chunk = [pd.concat([grouped.get_group(grp) for grp in sub_group])
                        for sub_group in self.group_keys_chunk]

        # print(n_total_point)
        for index, [chunk_key, chunk_group] in enumerate(zip(self.group_keys_chunk, groups_chunk)):
            # print(index,chunk_key)
            n_total_point_chunk = {k: n_total_point[k]
                                   for k in n_total_point if k in chunk_key}
            # print(n_total_point_chunk)#, chunk_group,chunk_group.dtypes)
            # raise Exception()
            print("Training network " + str(index) +
                  " for group " + str(chunk_key))

            kdeModelWrapper = KdeModelTrainer(mdl, tbl, xheader, yheader, groupby_attribute=groupby_attribute,
                                              groupby_values=chunk_key,
                                              n_total_point=n_total_point_chunk, n_sample_point={},
                                              x_min_value=-np.inf, x_max_value=np.inf, config=self.config).fit_from_df(
                chunk_group, network_size="small", n_mdn_layer_node=n_mdn_layer_node,
                b_one_hot_encoding=b_one_hot_encoding, b_grid_search=b_grid_search)

            engine = MdnQueryEngine(kdeModelWrapper, config=self.config)
            self.enginesContainer[index] = engine
        return self

    def predicts(self, func, x_lb, x_ub, n_jobs=4, result2file=None):
        instances = []
        predictions = {}
        times = {}
        with Pool(processes=n_jobs) as pool:
            # print(self.group_keys_chunk)
            for index, sub_group in enumerate(self.group_keys_chunk):
                # print(sub_group)
                engine = self.enginesContainer[index]
                i = pool.apply_async(query_partial_group,
                                     (engine, sub_group, func, x_lb, x_ub))
                instances.append(i)
                # print(i.get())
                # print(instances[0].get(timeout=1))
            for i in instances:
                result = i.get()
                pred = result[0]
                t = result[1]
                predictions.update(pred)
                times.update(t)
        if result2file is not None:
            # print(predictions)
            with open(result2file, 'w') as f:
                for key in predictions:
                    f.write(str(key) + "," + str(predictions[key]) + "\n")
        return predictions, times

    def init_pickle_file_name(self):
        # self.pickle_file_name = self.pickle_file_name
        return self.pickle_file_name + ".pkl"

    def serialize2warehouse(self, warehouse):
        if self.pickle_file_name is None:
            self.init_pickle_file_name()

        with open(warehouse + '/' + self.init_pickle_file_name(), 'wb') as f:
            dill.dump(self, f)


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
        # "b_reg_mean":'True',
        "num_epoch": 400,
        "reg_type": "mdn",
        "density_type": "density_type",
        "num_gaussians": 4,
    }

    headers = ["ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk", "ss_cdemo_sk", "ss_hdemo_sk",
               "ss_addr_sk", "ss_store_sk", "ss_promo_sk", "ss_ticket_number", "ss_quantity", "ss_wholesale_cost",
               "ss_list_price", "ss_sales_price", "ss_ext_discount_amt", "ss_ext_sales_price",
               "ss_ext_wholesale_cost", "ss_ext_list_price", "ss_ext_tax", "ss_coupon_amt", "ss_net_paid",
               "ss_net_paid_inc_tax", "ss_net_profit", "none"]
    groupby_attribute = "ss_store_sk"
    xheader = "ss_wholesale_cost"
    yheader = "ss_list_price"

    sampler = DBEstSampling(headers=headers, usecols=[
                            xheader, yheader, groupby_attribute])
    total_count = {'total': 2879987999}
    original_data_file = "/data/tpcds/40G/ss_600k_headers.csv"

    sampler.make_sample(original_data_file, 60000, "uniform", split_char="|",
                        num_total_records=total_count)
    xyzs = sampler.getyx(yheader, xheader, groupby=groupby_attribute)
    n_total_point = get_group_count_from_summary_file(
        config['warehousedir'] + "/num_of_points57.txt", sep=',')

    bundles = MdnQueryEngineBundle(config=config)
    bundles.fit(xyzs, groupby_attribute, n_total_point, "mdl", "tbl",
                xheader, yheader, n_per_group=30, b_grid_search=False,)

    bundles.predicts("count", 2451119, 2451483)
