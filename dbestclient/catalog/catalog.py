# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk


class DBEstModelCatalog:
    def __init__(self):
        self.model_catalog = {}

    def add_model_wrapper(self, model_wrapper):
        if model_wrapper.groupby_attribute is None:
            self.model_catalog[model_wrapper.init_pickle_file_name()] = model_wrapper
        else:
            self.model_catalog[model_wrapper.get_groupby_catalog_prefix()]={}
            current_group = self.model_catalog[model_wrapper.get_groupby_catalog_prefix()]


