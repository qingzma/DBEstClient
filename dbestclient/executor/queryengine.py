# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from datetime import  datetime
from scipy import integrate
import numpy as np




class QueryEngine:
    def __init__(self, reg, kde, n_training_point, n_total_point, x_min, x_max, config=None):
        self.n_training_point = n_training_point
        self.n_total_point = n_total_point
        self.reg = reg
        self.kde = kde
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
                'csv_split_char': '|'
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

    def predict(self,func, x_lb, x_ub):
        if func.lower() == "count":
            p,t = self.approx_count(x_lb, x_ub)
        elif func.lower() == "sum":
            p,t = self.approx_sum(x_lb, x_ub)
        elif func.lower() == "avg":
            p,t = self.approx_avg(x_lb, x_ub)
        else:
            print("Aggregate function " + func + " is not implemented yet!")
        return p,t

if __name__ == "__main__":
    pass


