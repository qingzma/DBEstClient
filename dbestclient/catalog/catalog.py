# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk


class DBEstModelCatalog:
    def __init__(self):
        self.model_catalog = {}

    def add_model_wrapper(self, model_wrapper, runtime_config):
        if hasattr(model_wrapper, 'groupby_value'):        # using regression and kde
            if model_wrapper.groupby_value is None:
                self.model_catalog[model_wrapper.init_pickle_file_name(runtime_config
                                                                       )] = model_wrapper
            else:
                self.model_catalog[model_wrapper.dir] = model_wrapper.models
        else:  # using kde
            self.model_catalog[model_wrapper.init_pickle_file_name(runtime_config)
                               ] = model_wrapper
