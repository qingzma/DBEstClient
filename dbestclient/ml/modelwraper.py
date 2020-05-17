# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
# import pickle
import os

import dill
import numpy as np


def deserialize_model_wrapper(file):
    return dill.load(file)


def get_pickle_file_name(mdl, runtime_config):
    return mdl + runtime_config["model_suffix"]
    # return mdl+"_yx_"+y+"_"+x+".pkl"


class SimpleModelWrapper:
    def __init__(self, mdl, tbl, x, y=None, n_total_point=1.0,
                 n_sample_point=1.0, x_min_value=-np.inf, x_max_value=np.inf, groupby_attribute=None, groupby_value=None):
        self.mdl = mdl
        self.tbl = tbl
        self.x = x
        self.y = y
        self.n_total_point = n_total_point
        self.n_sample_point = n_sample_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.groupby_attribute = groupby_attribute
        self.groupby_value = groupby_value

        self.reg = None
        self.density = None

        # generate the pickle file name
        self.pickle_file_name = None
        # self.init_pickle_file_name(runtime_config)
        # self.pickle_string = None

    def load_model(self, density, reg=None):
        self.reg = reg
        self.density = density

    def init_pickle_file_name(self, runtime_config):
        # self.pickle_file_name = self.mdl #+ "_" + self.tbl
        # if self.y is not None:
        #     self.pickle_file_name += "_yx_" + self.y + "_"
        # else:
        #     self.pickle_file_name += "_x_"
        # self.pickle_file_name += self.x
        # if self.groupby_attribute is not None:
        #     self.pickle_file_name += "_groupby_" + self.groupby_attribute + "_" + self.groupby_value
        # self.pickle_file_name += ".pkl"
        # return self.pickle_file_name
        self.pickle_file_name = self.mdl
        if self.groupby_value is not None:
            self.pickle_file_name += "_groupby_" + self.groupby_value
        self.pickle_file_name += runtime_config["model_suffix"]
        return self.pickle_file_name

    # def get_groupby_catalog_prefix(self):
    #     # groupby_catalog_prefix = self.mdl #+ "_" + self.tbl
    #     # if self.y is not None:
    #     #     groupby_catalog_prefix += "_yx_" + self.y + "_"
    #     # else:
    #     #     groupby_catalog_prefix += "_x_"
    #     # groupby_catalog_prefix += self.x
    #     # if self.groupby_attribute is not None:
    #     #     groupby_catalog_prefix += "_groupby_" + self.groupby_attribute
    #     # return groupby_catalog_prefix
    #     return self.mdl + "_groupby_" + self.groupby_attribute

    def serialize(self):
        return dill.dumps(self)

    def serialize2warehouse(self, warehouse, runtime_config):
        if self.pickle_file_name is None:
            self.init_pickle_file_name(runtime_config)
        with open(warehouse + '/' + self.pickle_file_name, 'wb') as f:
            dill.dump(self, f)


class GroupByModelWrapper:
    def __init__(self, mdl, tbl, x, y, groupby_attribute,  x_min_value=-np.inf, x_max_value=np.inf):
        self.mdl = mdl
        self.tbl = tbl
        self.x = x
        self.y = y
        self.n_total_point = {}  # n_total_point
        self.n_sample_point = {}  # n_sample_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.groupby_attribute = groupby_attribute

        self.models = {}
        # generate the pickle file names
        self.dir = self.mdl + "_groupby_" + self.groupby_attribute

    def add_simple_model(self, simple_model):
        self.models[simple_model.init_pickle_file_name()] = simple_model
        self.n_total_point[simple_model.groupby_value] = simple_model.n_total_point
        self.n_sample_point[simple_model.groupby_value] = simple_model.n_sample_point

    def serialize2warehouse(self, warehouse):
        if os.path.exists(warehouse):
            print("warehouse for the group by exists! abort!")
        else:
            os.mkdir(warehouse)
            for group, model_wrapper in self.models.items():
                model_wrapper.serialize2warehouse(warehouse)


class KdeModelWrapper:
    def __init__(self, mdl, tbl, x, y=None, n_total_point={},
                 x_min_value=-np.inf, x_max_value=np.inf, groupby_attribute=None, groupby_values={}):
        self.mdl = mdl
        self.tbl = tbl
        self.x = x
        self.y = y
        self.n_total_point = n_total_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.groupby_attribute = groupby_attribute
        self.groupby_values = groupby_values

        self.reg = None
        self.density = None

        # generate the pickle file name
        self.pickle_file_name = None
        # self.init_pickle_file_name()
        # self.pickle_string = None

    def load_model(self, mdl_name, density, reg=None):
        self.reg = reg
        self.density = density
        self.mdl = mdl_name

    def init_pickle_file_name(self, runtime_config):
        self.pickle_file_name = self.mdl
        # if self.groupby_attribute is not None:
        #     self.pickle_file_name += "_groupby_" + self.groupby_attribute
        self.pickle_file_name += runtime_config["model_suffix"]
        return self.pickle_file_name

    def serialize(self):
        return dill.dumps(self)

    def serialize2warehouse(self, warehouse, runtime_config):
        if self.pickle_file_name is None:
            self.init_pickle_file_name(runtime_config)
        self.x = None
        self.y = None
        with open(warehouse + '/' + self.pickle_file_name, 'wb') as f:
            dill.dump(self, f)


if __name__ == "__main__":
    # simpleWraper = SimpleModelWrapper("mdl", "tbl", "x", y="y", groupby_attribute="z", groupby_value="10")
    # print(simpleWraper.init_pickle_file_name())
    # print(simpleWraper.get_groupby_catalog_prefix())
    # print(get_pickle_file_name("mdl"))

    kdeModelWraper = KdeModelWrapper(
        "mdl", "tbl", "x", y="y", groupby_attribute="z", groupby_values={"1": 10, "2": 20})
    print(kdeModelWraper.init_pickle_file_name({}))
    # print(kdeModelWraper.get_groupby_catalog_prefix())
    print(get_pickle_file_name("mdl", {}))
