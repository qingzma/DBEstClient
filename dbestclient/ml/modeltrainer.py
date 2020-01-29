# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.ml.density import DBEstDensity
from dbestclient.ml.modelwraper import SimpleModelWrapper, GroupByModelWrapper, KdeModelWrapper
from dbestclient.ml.regression import DBEstReg
from dbestclient.tools.dftools import convert_df_to_yx
import numpy as np
from dbestclient.ml.mdn import RegMdn
import pandas as pd


class SimpleModelTrainer:

    def __init__(self, mdl, tbl, xheader, yheader, n_total_point, n_sample_point, groupby_attribute=None,
                 groupby_value=None, config=None):
        self.xheader = xheader
        self.yheader = yheader
        self.simpe_model_wrapper = SimpleModelWrapper(mdl, tbl, xheader, y=yheader, n_total_point=n_total_point,
                                                      n_sample_point=n_sample_point,
                                                      groupby_attribute=groupby_attribute, groupby_value=groupby_value)
        self.config = config

    def fit(self, x, y):
        # print(x,y)
        # if x_reg is None:
        #     reg = DBEstReg(config=self.config).fit(x_kde, y_kde)
        # else:
        reg = DBEstReg(config=self.config).fit(x, y)
        density = DBEstDensity(config=self.config).fit(x)
        # print("in modeltrainer",reg.predict([[1000], [1005],[1010], [1015],[1020], [1025],[1030], [1035]]))
        self.simpe_model_wrapper.load_model(density, reg)
        return self.simpe_model_wrapper

    def fit_from_df(self, df):
        # if df_reg is None:
        #     y_kde, x_kde = convert_df_to_yx(df_kde, self.xheader, self.yheader)
        #     return self.fit(x_kde, y_kde)
        # else:
        y, x = convert_df_to_yx(df, self.xheader, self.yheader)
        # y_kde, x_kde = convert_df_to_yx(df_kde, self.xheader, self.yheader)
        return self.fit(x, y)


class GroupByModelTrainer:
    def __init__(self, mdl, tbl, xheader, yheader, groupby_attribute, n_total_point, n_sample_point,
                 x_min_value=-np.inf, x_max_value=np.inf, config=None):
        self.groupby_model_wrapper = GroupByModelWrapper(mdl, tbl, xheader, yheader, groupby_attribute,
                                                         x_min_value=x_min_value, x_max_value=x_max_value)
        self.groupby_attribute = groupby_attribute
        self.mdl = mdl
        self.tbl = tbl
        self.xheader = xheader
        self.yheader = yheader
        self.n_total_point = n_total_point
        self.n_sample_point = n_sample_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.config = config

    def fit_from_df(self, df):
        sample_grouped = df.groupby(by=self.groupby_attribute)
        for name, group in sample_grouped:
            print("training " + name)
            simple_model_wrapper = SimpleModelTrainer(self.mdl, self.tbl, self.xheader, self.yheader,
                                                      self.n_total_point[name], self.n_sample_point[name],
                                                      groupby_attribute=self.groupby_attribute, groupby_value=name,
                                                      config=self.config).fit_from_df(group)
            self.groupby_model_wrapper.add_simple_model(simple_model_wrapper)
        # print(self.groupby_model_wrapper)
        return self.groupby_model_wrapper


class KdeModelTrainer:
    def __init__(self, mdl, tbl, xheader, yheader, groupby_attribute,groupby_values, n_total_point, n_sample_point,
                 x_min_value=-np.inf, x_max_value=np.inf, config=None):
        self.kde_model_wrapper = KdeModelWrapper(mdl, tbl, xheader, yheader, n_total_point, n_sample_point,
                                                 x_min_value=x_min_value, x_max_value=x_max_value,
                                                 groupby_values=groupby_values)
        self.groupby_attribute = groupby_attribute
        self.groupby_values = groupby_values
        self.mdl = mdl
        self.tbl = tbl
        self.xheader = xheader
        self.yheader = yheader
        self.n_total_point = n_total_point
        self.n_sample_point = n_sample_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.config = config

    def fit_from_df(self, df):
        print("Starting training kde models for model "+ self.mdl)

        x = df[self.xheader].values#.reshape(-1,1)
        y = df[self.yheader].values
        groupby = df[self.groupby_attribute].values



        xzs_train = np.concatenate(
            (x[:, np.newaxis], groupby[:, np.newaxis]), axis=1)
        print("training regression...")
        reg = RegMdn(dim_input=2).fit(xzs_train, y,num_epoch=5)

        print("training density...")
        density = RegMdn(dim_input=1)
        density.fit(groupby[:,np.newaxis], x, num_epoch=5, b_show_plot=False, num_gaussians=5)

        # density = DBEstDensity(config=self.config).fit(x)
        self.kde_model_wrapper.load_model(density, reg)

        return self.kde_model_wrapper

        # sample_grouped = df.groupby(by=self.groupby_attribute)
        # for name, group in sample_grouped:
        #     print("training " + name)
        #     simple_model_wrapper = SimpleModelTrainer(self.mdl, self.tbl, self.xheader, self.yheader,
        #                                               self.n_total_point[name], self.n_sample_point[name],
        #                                               groupby_attribute=self.groupby_attribute, groupby_value=name,
        #                                               config=self.config).fit_from_df(group)
        #     self.groupby_model_wrapper.add_simple_model(simple_model_wrapper)
        # # print(self.groupby_model_wrapper)
        # return self.groupby_model_wrapper
