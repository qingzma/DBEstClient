# Created by Qingzhi Ma at 19/02/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import numpy as np

from dbestclient.ml.mdn import KdeMdn
import matplotlib.pyplot as plt


def test_ss_2d_density():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    file = "/data/tpcds/40G/ss_600k_headers.csv"
    df = pd.read_csv(file, sep='|')
    df = df.dropna(subset=['ss_sold_date_sk', 'ss_store_sk', 'ss_sales_price'])
    df_train = df  # .head(1000)
    df_test = df  # .head(1000)
    g_train = df_train.ss_store_sk.values[:, np.newaxis]
    x_train = df_train.ss_sold_date_sk.values  # df_train.pm25.values
    g_test = df_test.ss_store_sk.values
    # temp_test = df_test.PRES.values  #df_test.pm25.values

    # data = df_train.groupby(["ss_store_sk"]).get_group(1)["ss_sold_date_sk"].values
    # plt.hist(data, bins=100)
    # plt.show()
    # raise Exception()
    print(g_train)
    print(x_train)
    raise Exception()

    kdeMdn = KdeMdn(b_store_training_data=True, b_one_hot=True)
    kdeMdn.fit(g_train, x_train, num_epoch=10, num_gaussians=20)

    kdeMdn.plot_density_per_group()

    # regMdn = RegMdn(dim_input=1, b_store_training_data=True)
    # regMdn.fit(g_train, x_train, num_epoch=100, b_show_plot=False, num_gaussians=5)
    #
    # print(kdeMdn.predict([[1]], 2451119, b_plot=True))
    xxs, p = kdeMdn.predict([[1]], 2451119, b_plot=True)
    xxs = [kdeMdn.denormalize(xi, kdeMdn.meanx, kdeMdn.widthx) for xi in xxs]
    print(xxs, p)
    # yys = [regMdn.denormalize(yi, regMdn.meany, regMdn.widthy) for yi in yys]
    plt.plot(xxs, p)
    plt.show()


if __name__=="__main__":
    test_ss_2d_density()