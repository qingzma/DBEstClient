# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

import numpy as np
from sklearn.neighbors import KernelDensity

from dbestclient.ml.mdn import KdeMdn


class DBEstDensity:
    def __init__(self, config, kernel=None):
        if kernel is None:
            self.kernel = 'gaussian'
        self.kde = None
        self.config = config

    def fit(self, x, groupby_attribute,  runtime_config):
        density_type = self.config.config["density_type"]

        if density_type == 'kde':
            self.kde = KernelDensity(kernel=self.kernel).fit(x)
        elif density_type == 'mdn':
            print("x", x, type(x))
            groups = np.zeros(x.shape)
            print("groups", groups)
            x = x.reshape(1, -1)[0]
            print("x", x)
            self.kde = KdeMdn(self.config).fit(
                groups, x, runtime_config)
        else:
            raise Exception("unexpected density_type.")
        return self.kde
