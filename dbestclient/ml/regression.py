# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from qregpy import qreg
from dbestclient.ml import mdn

class DBEstReg:
    def __init__(self):
        self.reg = None

    def fit(self, x, y, type='qreg'):
        if type == 'qreg':
            self.reg = qreg.QReg(base_models=["linear", "polynomial"], verbose=False).fit(x, y)
        if type == 'torch':
            self.reg = mdn.RegMdn(dim_input=1).fit(x,y,num_epoch=100)
        return self.reg
