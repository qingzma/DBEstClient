# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from qregpy import qreg
from dbestclient.ml import mdn

class DBEstReg:
    def __init__(self, config):
        self.reg = None
        self.config = config

    def fit(self, x, y, reg_type=None):
        if self.config["reg_type"]  not in ['mdn','qreg']:
            raise Exception("The regression type must be mdn or qreg!")
        if reg_type is None:
            reg_type = self.config["reg_type"]

        if reg_type == 'qreg':
            self.reg = qreg.QReg(base_models=["linear", "polynomial"], verbose=False).fit(x, y)
        if reg_type == 'mdn':
            self.reg = mdn.RegMdn(dim_input=1).fit(x,y,num_epoch=self.config["num_epoch"],num_gaussians=self.config["num_gaussians"])
        return self.reg
