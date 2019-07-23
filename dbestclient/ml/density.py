# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

from sklearn.neighbors import KernelDensity


class DBEstDensity:
    def __init__(self, kernel=None):
        if kernel is None:
            self.kernel = 'gaussian'
        self.kde = None

    def fit(self, x):
        self.kde = KernelDensity(kernel=self.kernel).fit(x)
        return self.kde