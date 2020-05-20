# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk


import numpy as np

from dbestclient.ml.density import DBEstDensity
from dbestclient.ml.mdn import KdeMdn, RegMdnGroupBy  # RegMdn
from dbestclient.ml.modelwraper import (GroupByModelWrapper, KdeModelWrapper,
                                        SimpleModelWrapper)
from dbestclient.ml.regression import DBEstReg
from dbestclient.tools.dftools import convert_df_to_yx


class SimpleModelTrainer:

    def __init__(self, mdl, tbl, xheader, yheader, n_total_point, n_sample_point, groupby_attribute=None,
                 groupby_value=None, config=None):
        self.xheader = xheader
        self.yheader = yheader
        self.simpe_model_wrapper = SimpleModelWrapper(mdl, tbl, xheader, y=yheader, n_total_point=n_total_point,
                                                      n_sample_point=n_sample_point,
                                                      groupby_attribute=groupby_attribute, groupby_value=groupby_value)
        self.config = config

    def fit(self, x, y, runtime_config):
        # print(x,y)
        # if x_reg is None:
        #     reg = DBEstReg(config=self.config).fit(x_kde, y_kde)
        # else:
        reg = DBEstReg(config=self.config).fit(x, y, runtime_config)
        density = DBEstDensity(config=self.config).fit(x, None, runtime_config)
        # print("in modeltrainer",reg.predict([[1000], [1005],[1010], [1015],[1020], [1025],[1030], [1035]]))
        self.simpe_model_wrapper.load_model(density, reg)
        return self.simpe_model_wrapper

    def fit_from_df(self, df, runtime_config):
        # if df_reg is None:
        #     y_kde, x_kde = convert_df_to_yx(df_kde, self.xheader, self.yheader)
        #     return self.fit(x_kde, y_kde)
        # else:
        y, x = convert_df_to_yx(df, self.xheader[0], self.yheader[0])
        # y_kde, x_kde = convert_df_to_yx(df_kde, self.xheader, self.yheader)
        return self.fit(x, y, runtime_config)


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

    def fit_from_df(self, df, runtime_config):
        sample_grouped = df.groupby(by=self.groupby_attribute)
        for name, group in sample_grouped:
            print("training " + name)
            simple_model_wrapper = SimpleModelTrainer(self.mdl, self.tbl, self.xheader, self.yheader,
                                                      self.n_total_point[name], self.n_sample_point[name],
                                                      groupby_attribute=self.groupby_attribute, groupby_value=name,
                                                      config=self.config).fit_from_df(group, runtime_config)
            self.groupby_model_wrapper.add_simple_model(simple_model_wrapper)
        # print(self.groupby_model_wrapper)
        return self.groupby_model_wrapper


class KdeModelTrainer:
    def __init__(self, mdl, tbl, xheader, yheader, groupby_attribute, groupby_values, n_total_point,
                 x_min_value=-np.inf, x_max_value=np.inf, config=None):
        self.kde_model_wrapper = KdeModelWrapper(mdl, tbl, xheader, yheader, n_total_point,
                                                 x_min_value=x_min_value, x_max_value=x_max_value,
                                                 groupby_values=groupby_values)
        self.groupby_attribute = groupby_attribute
        self.groupby_values = groupby_values
        self.mdl = mdl
        self.tbl = tbl
        self.xheader = xheader
        self.yheader = yheader
        self.n_total_point = n_total_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.config = config
        self.enc = None

        # if device.lower() not in ("cpu", "gpu"):
        #     raise ValueError("unexpected device type.")
        # if device.lower() == "cpu":
        #     # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     device = torch.device("cpu")
        #     print("CPU is used"+'-'*20)
        # else:
        #     if torch.cuda.is_available():
        #         print("GPU available, use GPU.")
        #         device = torch.device("cuda:0")
        #     else:
        #         print("No GPU available, use CPU instead.")
        #         print("CPU is used"+'-'*20)
        #         device = torch.device("cpu")
        # self.device = device

    def fit_from_df(self, df, runtime_config, network_size=None,  b_shuffle_data=True):
        # init parameters
        device = runtime_config["device"]

        print("Starting training kde models for model " + self.mdl)

        # print(df)
        # shuffle the order in the data
        if b_shuffle_data:
            df = df.sample(frac=1).reset_index(drop=True)

        x = df[self.xheader].values  # .reshape(-1,1)
        if self.yheader[1] == "categorical":
            b_skip_reg_training = True
            b_skip_density_training = True
        else:
            b_skip_reg_training = False
            b_skip_density_training = False
            # print(df)
            # print(self.yheader[0])
            y = df[self.yheader[0]].values
        groupby = df[self.groupby_attribute].values

        xzs_train = np.concatenate(
            (x[:, np.newaxis], groupby), axis=1)

        if network_size is None:
            if b_skip_reg_training:
                reg = None
            else:
                print("training regression...")
                print("*"*80)
                config = self.config.copy()
                reg = RegMdnGroupBy(config, b_store_training_data=False).fit(
                    groupby, x, y, runtime_config)

            if b_skip_density_training:
                density = None
            else:
                print("training density...")
                print("*"*80)
                # density = RegMdn(dim_input=1,n_mdn_layer_node=20)
                config = self.config.copy()
                density = KdeMdn(config,
                                 b_store_training_data=False).fit(groupby, x, runtime_config)
        else:
            if network_size.lower() == "small":
                if b_skip_reg_training:
                    reg = None
                else:
                    print("training regression...")
                    print("*"*80)
                    config = self.config.copy()
                    # config.config["n_epoch"] = 10
                    config.config["n_gaussians_reg"] = 3
                    # config.config["n_mdn_layer_node"] = 10
                    # config.config["b_grid_search"] = False
                    reg = RegMdnGroupBy(config, b_store_training_data=False).fit(
                        groupby, x, y, runtime_config)

                if b_skip_density_training:
                    density = None
                else:
                    print("training density...")
                    print("*"*80)
                    # density = RegMdn(dim_input=1,n_mdn_layer_node=20)
                    config = self.config.copy()
                    # config.config["n_epoch"] = 20
                    config.config["n_gaussions_density"] = 10
                    # config.config["n_mdn_layer_node"] = 10
                    # config.config["b_grid_search"] = False

                    density = KdeMdn(config,
                                     b_store_training_data=False).fit(groupby, x, runtime_config)

            elif network_size.lower() == "large":
                if b_skip_reg_training:
                    reg = None
                else:
                    print("training regression...")
                    print("*"*80)
                    config = self.config.copy()
                    # config.config["n_epoch"] = 20
                    config.config["n_gaussians_reg"] = 5
                    # config.config["n_mdn_layer_node"] = 20
                    # config.config["b_grid_search"] = False
                    reg = RegMdnGroupBy(config, b_store_training_data=False,).fit(
                        groupby, x, y, runtime_config)

                if b_skip_density_training:
                    density = None
                else:
                    print("training density...")
                    print("*"*80)
                    config = self.config.copy()
                    # config.config["n_epoch"] = 20
                    config.config["n_gaussians_density"] = 20
                    # config.config["n_mdn_layer_node"] = 20
                    # config.config["b_grid_search"] = False
                    # density = RegMdn(dim_input=1,n_mdn_layer_node=20)
                    density = KdeMdn(config, b_store_training_data=False).fit(
                        groupby, x, runtime_config)

            elif network_size.lower() == "testing":
                if b_skip_reg_training:
                    reg = None
                else:
                    print("training regression...")
                    print("*"*80)
                    config = self.config.copy()
                    config.config["n_epoch"] = 1
                    config.config["n_gaussians_reg"] = 2
                    config.config["n_mdn_layer_node"] = 20
                    config.config["b_grid_search"] = False
                    reg = RegMdnGroupBy(config, b_store_training_data=False,).fit(
                        groupby, x, y, runtime_config)
                if b_skip_density_training:
                    density = None
                else:
                    print("training density...")
                    print("*"*80)
                    # density = RegMdn(dim_input=1,n_mdn_layer_node=20)
                    config = self.config.copy()
                    config.config["n_epoch"] = 2
                    config.config["n_gaussians_density"] = 8
                    config.config["n_mdn_layer_node"] = 10
                    config.config["b_grid_search"] = False
                    density = KdeMdn(config, b_store_training_data=False).fit(
                        groupby, x, runtime_config)

            else:
                raise ValueError("unexpected network_size passed in "+__file__)

        # density.plot_density_per_group()

        # density = DBEstDensity(config=self.config).fit(x)
        self.kde_model_wrapper.load_model(self.mdl, density, reg)

        return self.kde_model_wrapper
