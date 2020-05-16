# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

# from sklearn.neighbors import KernelDensity

# from dbestclient.ml.mdn import RegMdn


# class DBEstDensity:
#     def __init__(self,config, kernel=None):
#         if kernel is None:
#             self.kernel = 'gaussian'
#         self.kde = None
#         self.config=config

#     def fit(self, x, groupby_attribute=None,density_type=None):
#         if self.config["density_type"]  not in ['mdn','kde']:
#             raise Exception("The density type must be mdn or kde!")
#         if density_type is None:
#             density_type = self.config["density_type"]

#         if density_type == 'kde':
#             self.kde = KernelDensity(kernel=self.kernel).fit(x)
#         if density_type == 'mdn' and groupby_attribute is not None:
#             self.kde = RegMdn(dim_input=2).fit(groupby_attribute, x,num_epoch=self.config["num_epoch"],num_gaussians=self.config["num_gaussians"])
#         return self.kde
