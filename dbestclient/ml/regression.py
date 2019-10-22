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

    def fit(self, x, y, type='tf'):
        if type == 'qreg':
            self.reg = qreg.QReg(base_models=["linear", "polynomial"], verbose=False).fit(x, y)
        if type == 'tf':
            self.reg = mdn.MDN(n_features=1,n_kernels=20).fit(x,y)
        return self.reg
