# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import pickle
import numpy as np
import os

def deserialize_model_wrapper(file):
    return pickle.load(file)


def get_pickle_file_name(mdl):
    return mdl + ".pkl"
    # return mdl+"_yx_"+y+"_"+x+".pkl"


class SimpleModelWrapper:
    def __init__(self, mdl, tbl, x, y=None, n_total_point=1.0,
                 n_sample_point=1.0, x_min_value=-np.inf, x_max_value=np.inf, groupby_tag=None):
        self.mdl = mdl
        self.tbl = tbl
        self.x = x
        self.y = y
        self.n_total_point = n_total_point
        self.n_sample_point = n_sample_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.groupby_tag=groupby_tag

        self.reg = None
        self.density = None

        # generate the pickle file name
        self.pickle_file_name = None
        self.init_pickle_file_name()
        # self.pickle_string = None

    def load_model(self, density, reg=None):
        self.reg = reg
        self.density = density

    def init_pickle_file_name(self):
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
        if self.groupby_tag is not None:
            self.pickle_file_name += self.groupby_tag
        self.pickle_file_name += ".pkl"
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
        return pickle.dumps(self)

    def serialize2warehouse(self, warehouse):
        if self.pickle_file_name is None:
            self.init_pickle_file_name()
        with open(warehouse + '/' + self.pickle_file_name, 'wb') as f:
            pickle.dump(self, f)


class GroupByModelWrapper:
    def __init__(self, mdl, tbl, x, y, groupby_attribute, n_total_point={},
                 n_sample_point={}, x_min_value=-np.inf, x_max_value=np.inf):
        self.mdl = mdl
        self.tbl = tbl
        self.x = x
        self.y = y
        self.n_total_point = n_total_point
        self.n_sample_point = n_sample_point
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.groupby_attribute = groupby_attribute

        self.models={}
        # generate the pickle file names
        self.dir = self.mdl + "_groupby_" + self.groupby_attribute


    def add_simple_model(self,simple_model):
        self.models[simple_model.init_pickle_file_name()] = simple_model

    def serialize2warehouse(self, warehouse):
        if os.path.exists(warehouse):
            print("warehouse for the group by exists! abort!")
        else:
            os.mkdir(warehouse)
            for group, model_wrapper in self.models.items():
                model_wrapper.serialize2warehouse(warehouse)





if __name__ == "__main__":
    simpleWraper = SimpleModelWrapper("mdl", "tbl", "x", y="y", groupby_attribute="z", groupby_value="10")
    print(simpleWraper.init_pickle_file_name())
    print(simpleWraper.get_groupby_catalog_prefix())
    print(get_pickle_file_name("mdl"))
