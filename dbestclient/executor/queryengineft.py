# Created by Qingzhi Ma at 14/02/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

# This is the query engine for frequence table-based AQP processing.
from datetime import datetime

import numpy as np
from scipy import integrate


class QueryEngineFt:
    def __init__(self, reg, ft, n_training_point, n_total_point, x_min, x_max, config=None):
        self.n_training_point = n_training_point
        self.n_total_point = n_total_point
        self.reg = reg
        self.ft = ft
        self.x_min = x_min
        self.x_max = x_max
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

    def approx_avg(self, x_min, x_max):
        start = datetime.now()

        def f_pRx(*args):
            # print(self.cregression.predict(x))
            return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1))) * self.reg.predict(
                [[args[0]]])[0]

        def f_p(*args):
            return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

        a = integrate.quad(f_pRx, x_min, x_max,
                           epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[0]
        b = integrate.quad(f_p, x_min, x_max,
                           epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[0]

        if b:
            result = a / b
        else:
            result = None
        # if result != None:
        #      print("Approximate AVG: %.4f." % result)
        # else:
        #     print("Nan")

        if self.config['verbose']:
            end = datetime.now()
            time_cost = (end - start).total_seconds()
            # print("Time spent for approximate AVG: %.4fs." % time_cost)
        return result, time_cost

    def approx_sum(self, x_min, x_max):
        start = datetime.now()

        def f_pRx(*args):
            return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1))) \
                   * self.reg.predict([[args[0]]])[0]
            # * self.reg.predict(np.array(args))

        # print(integrate.quad(f_pRx, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[0])
        result = integrate.quad(f_pRx, x_min, x_max, epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[
                     0] * float(self.n_total_point)
        # return result

        # result = result / float(self.n_training_point) * float(self.n_total_point)

        # print("Approximate SUM: %.4f." % result)

        if self.config['verbose'] and result != None:
            end = datetime.now()
            time_cost = (end - start).total_seconds()
            # print("Time spent for approximate SUM: %.4fs." % time_cost)
        return result, time_cost

    def approx_count(self, x_min, x_max):
        start = datetime.now()

        def f_p(*args):
            return np.exp(self.kde.score_samples(np.array(args).reshape(1, -1)))

        result = integrate.quad(f_p, x_min, x_max, epsabs=self.config['epsabs'], epsrel=self.config['epsrel'])[0]
        result = result * float(self.n_total_point)

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

    def predicts(self, func, x_lb, x_ub):
        predictions = []
        times = []
        for groupby_value in self.groupby_values:
            if groupby_value == "":
                continue
            pre, t = self.predict(func, x_lb, x_ub, float(groupby_value))
            predictions.append(pre)
            times.append(t)
            print(groupby_value, pre)
        return predictions, times


if __name__ == "__main__":
    pass