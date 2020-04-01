# Created by Qingzhi Ma at 18/11/2019
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from builtins import print

import numpy as np
import torch
import torch.nn.functional as functional
from qregpy import qreg

print(functional.one_hot(torch.tensor([[0,1],[0,2],[0,3]])))

# from dbestclient.ml import mdn

# x = np.linspace(0, 10, 100)
# y = x ** 2 + 1
# x = x[:, np.newaxis]
# # print(x)
# # print(y)
# # plt.plot(x,y)
# # plt.show()

# reg1 = qreg.QReg(base_models=["linear", "polynomial"], verbose=False).fit(x, y)
# preds1 = reg1.predict([[1], [2]])
# print(preds1)

# # x=x.reshape(-1,1)
# reg2 = mdn.RegMdn(dim_input=1).fit(x, y, num_gaussians=4, num_epoch=400)
# # preds2 = reg2.predict([1, 2])
# preds2 = reg2.predict([[1], [2]])
# preds2 = reg2.predict([[1], [2]])
# print("preds2", preds2)
# preds2
