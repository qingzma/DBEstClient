# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.ml.density import DBEstDensity
from dbestclient.ml.modelwraper import SimpleModelWrapper, GroupByModelWrapper
from dbestclient.ml.regression import DBEstReg
from dbestclient.tools.dftools import convert_df_to_yx
import numpy as np
from dbestclient.ml.mdn import RegMdn
import pandas as pd


class SimpleModelTrainer:

    def __init__(self, mdl, tbl, xheader, yheader, n_total_point, n_sample_point,groupby_attribute=None, groupby_value=None):
        self.xheader = xheader
        self.yheader = yheader
        self.simpe_model_wrapper = SimpleModelWrapper(mdl, tbl, xheader, y=yheader, n_total_point=n_total_point,
                                                      n_sample_point=n_sample_point, groupby_attribute=groupby_attribute, groupby_value=groupby_value)

    def fit(self, x_kde, y_kde, x_reg=None, y_reg=None):
        # print(x,y)
        if x_reg is None:
            reg = DBEstReg().fit(x_kde, y_kde, type='torch')
        else:
            reg = DBEstReg().fit(x_reg, y_reg, type='torch')
        density = DBEstDensity().fit(x_kde)
        # print("in modeltrainer",reg.predict([[1000], [1005],[1010], [1015],[1020], [1025],[1030], [1035]]))
        self.simpe_model_wrapper.load_model(density, reg)
        return self.simpe_model_wrapper

    def fit_from_df(self, df_reg, df_kde):
        if df_reg is None:
            y_kde, x_kde = convert_df_to_yx(df_kde, self.xheader, self.yheader)
            return self.fit(x_kde, y_kde)
        else:
            y_reg, x_reg = convert_df_to_yx(df_reg, self.xheader, self.yheader)
            y_kde, x_kde = convert_df_to_yx(df_kde, self.xheader, self.yheader)
            return self.fit(x_kde, y_kde, x_reg=x_reg, y_reg=y_reg)


class GroupByModelTrainer:
    def __init__(self, mdl, tbl, xheader, yheader, groupby_attribute, n_total_point, n_sample_point,
                 x_min_value=-np.inf, x_max_value=np.inf):
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


    def fit_from_df(self,df):
        sample_grouped = df.groupby(by=self.groupby_attribute)
        for name, group in sample_grouped:
            print("training " +name )
            simple_model_wrapper = SimpleModelTrainer(self.mdl, self.tbl, self.xheader, self.yheader,
                                                      self.n_total_point[name], self.n_sample_point[name],
                                                      groupby_attribute=self.groupby_attribute, groupby_value=name).fit_from_df(group)
            self.groupby_model_wrapper.add_simple_model(simple_model_wrapper)
        # print(self.groupby_model_wrapper)
        return self.groupby_model_wrapper

