# # Created by Qingzhi Ma
# # All right reserved
# # Department of Computer Science
# # the University of Warwick
# # Q.Ma.2@warwick.ac.uk


# https://www.katnoria.com/mdn/
# https://github.com/sagelywizard/pytorch-mdn
"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
# from torch.utils.data import Dataset
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats



ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    data = data.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)

    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i, idx]).add(mu[i, idx])
    return sample


class RegMdn():
    """ This class implements the regression using mixture density network.
    """

    def __init__(self, dim_input, b_store_training_data=True):
        if b_store_training_data:
            self.xs = None   # query range
            self.ys = None   # aggregate value
            self.zs = None   # group by balue
        self.b_store_training_data = b_store_training_data
        self.meanx = None
        self.widthx = None
        self.meany = None
        self.widthy = None
        self.meanz = None
        self.widthz = None
        self.model = None
        self.is_normalized = False
        self.dim_input = dim_input
        self.is_training_data_denormalized = False

    def fit(self, xs, ys, b_show_plot=False, b_normalize=True, num_epoch=400,num_gaussians=2):
        """ fit a regression y= R(x)"""
        if len(xs.shape) !=2:
            raise Exception("xs should be 2-d, but got unexpected shape.")
        if self.dim_input == 1:
            return self.fit2d(xs, ys, b_show_plot=b_show_plot,
                              b_normalize=b_normalize, num_epoch=num_epoch,num_gaussians=num_gaussians)
        elif self.dim_input == 2:
            return self.fit3d(xs[:, 0], xs[:, 1], ys, b_show_plot=b_show_plot,
                              b_normalize=b_normalize, num_epoch=num_epoch,num_gaussians=num_gaussians)
        else:
            print("dimension mismatch")
            sys.exit(0)

    def predict(self, xs, b_show_plot=False):
        """ make predictions"""
        if self.dim_input == 1:
            # print(self.predict2d(xs, b_show_plot=b_show_plot))
            return self.predict2d(xs, b_show_plot=b_show_plot)
        elif self.dim_input == 2:
            return self.predict3d(xs[:, 0], xs[:, 1], b_show_plot=b_show_plot)
        else:
            print("dimension mismatch")
            sys.exit(0)

    def fit3d(self, xs, zs, ys, b_show_plot=False, b_normalize=True, num_epoch=200,num_gaussians=5):
        """ fit a regression y = R(x,z)

        Args:
            xs ([float]): query range attribute
            zs ([float]): group by attribute
            ys ([float]): aggregate attribute
            b_show_plot (bool, optional): whether to show the plot. Defaults to True.
        """
        if b_normalize:
            self.meanx = np.mean(xs)
            self.widthx = np.max(xs)-np.min(xs)
            self.meany = np.mean(ys)
            self.widthy = np.max(ys)-np.min(ys)
            self.meanz = np.mean(zs)
            self.widthz = np.max(zs)-np.min(zs)

            # s= [(i-meanx)/1 for i in x]
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])
            ys = np.array([self.normalize(i, self.meany, self.widthy)
                           for i in ys])
            zs = np.array([self.normalize(i, self.meanz, self.widthz)
                           for i in zs])
            self.is_normalized = True
        self.xs = xs
        self.ys = ys
        self.zs = zs

        if b_show_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, zs, ys)
            ax.set_xlabel('query range attribute')
            ax.set_ylabel('group by attribute')
            ax.set_zlabel('aggregate attribute')
            plt.show()

        xzs = [[xs[i], zs[i]] for i in range(len(xs))]
        # xy =x[:,np.newaxis]
        ys = ys[:, np.newaxis]
        tensor_xzs = torch.stack([torch.Tensor(i)
                                  for i in xzs])  # transform to torch tensors

        tensor_ys = torch.stack([torch.Tensor(i) for i in ys])

        my_dataset = torch.utils.data.TensorDataset(
            tensor_xzs, tensor_ys)  # create your datset
        # , num_workers=8) # create your dataloader
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset, batch_size=1000, shuffle=False)

        # initialize the model
        self.model = nn.Sequential(
            nn.Linear(self.dim_input ,20),
            nn.Tanh(),
            nn.Dropout(0.01),
            MDN(20, 1, num_gaussians)
        )

        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(num_epoch):
            if epoch % 100 == 0:
                print("< Epoch {}".format(epoch))
            # train the model
            for minibatch, labels in my_dataloader:
                self.model.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = mdn_loss(pi, sigma, mu, labels)
                loss.backward()
                optimizer.step()
        return self

    def fit2d(self, xs, ys, b_show_plot=False, b_normalize=True, num_epoch=200,num_gaussians=5):
        """ fit a regression y = R(x)

        Args:
            xs ([float]): query range attribute
            ys ([float]): aggregate attribute
            b_show_plot (bool, optional): whether to show the plot. Defaults to True.
        """

        if b_normalize:
            self.meanx = np.mean(xs)
            self.widthx = np.max(xs)-np.min(xs)
            self.meany = np.mean(ys)
            self.widthy = np.max(ys)-np.min(ys)

            # s= [(i-meanx)/1 for i in x]
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])
            ys = np.array([self.normalize(i, self.meany, self.widthy)
                           for i in ys])

            self.is_normalized = True
        self.xs = xs
        self.ys = ys

        if b_show_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
            ax.set_xlabel('query range attribute')

            ax.set_ylabel('aggregate attribute')
            plt.show()

        # xzs = [[xs[i], zs[i]] for i in range(len(xs))]
        # xs = xs[:, np.newaxis]
        ys = ys[:, np.newaxis]
        tensor_xs = torch.stack([torch.Tensor(i)
                                 for i in xs])  # transform to torch tensors

        # tensor_x.flatten(-1)
        tensor_ys = torch.stack([torch.Tensor(i) for i in ys])

        my_dataset = torch.utils.data.TensorDataset(
            tensor_xs, tensor_ys)  # create your datset
        # , num_workers=8) # create your dataloader
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset, batch_size=1000, shuffle=False)

        # initialize the model
        self.model = nn.Sequential(
            nn.Linear(self.dim_input, 20),
            nn.Tanh(),
            nn.Dropout(0.01),
            MDN(20, 1, num_gaussians)
        )

        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(num_epoch):
            if epoch % 100 == 0:
                print("< Epoch {}".format(epoch))
            # train the model
            for minibatch, labels in my_dataloader:
                self.model.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = mdn_loss(pi, sigma, mu, labels)
                loss.backward()
                optimizer.step()



        # xxs = np.linspace(np.min(xs), np.max(xs),100)
        # yys = self.predict2d(xxs,b_show_plot=True)
        return self

    def predict3d(self, xs, zs, b_show_plot=True, num_points=10, b_generate_samples=False):
        if self.is_normalized:
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])
            zs = np.array([self.normalize(i, self.meanz, self.widthz)
                           for i in zs])
        xzs = np.array([[xs[i], zs[i]] for i in range(len(xs))])
        tensor_xzs = torch.stack([torch.Tensor(i)
                                  for i in xzs])
        # xzs_data = torch.from_numpy(xzs)

        pi, sigma, mu = self.model(tensor_xzs)
        # print("mu,", mu)
        # print("sigma", sigma)


        if b_generate_samples:
            samples = sample(pi, sigma, mu).data.numpy().reshape(-1)
            for i in range(num_points-1):
                samples = np.vstack(
                    (samples, sample(pi, sigma, mu).data.numpy().reshape(-1)))
            samples = np.mean(samples, axis=0)
        else:
            mu = mu.detach().numpy().reshape(len(xs), -1)
            pi = pi.detach().numpy()  # .reshape(-1,2)
            samples = np.sum(np.multiply(pi, mu), axis=1)

        # print(samples.data.numpy().reshape(-1))

        if self.is_normalized:
            # de-normalize the data
            samples = [self.denormalize(
                i, self.meany, self.widthy) for i in samples]
            xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                           for i in xs])
            zs = np.array([self.denormalize(i, self.meanz, self.widthz)
                           for i in zs])
        # print(x_test)
        if b_show_plot:

            if not self.is_training_data_denormalized:
                self.xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                                    for i in self.xs])
                self.ys = np.array([self.denormalize(i, self.meany, self.widthy)
                                    for i in self.ys])
                self.zs = np.array([self.denormalize(i, self.meanz, self.widthz)
                                    for i in self.zs])
                self.is_training_data_denormalized = True
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.xs, self.zs, self.ys)
            ax.scatter(xs, zs, samples)
            ax.set_xlabel('query range attribute')
            ax.set_ylabel('group by attribute')
            ax.set_zlabel('aggregate attribute')
            plt.show()
        return samples

    def predict2d(self, xs, b_show_plot=False, num_points=10, b_generate_samples=False):
        if self.is_normalized:
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])

        # xs = xs[:, np.newaxis]
        tensor_xs = torch.stack([torch.Tensor(i)
                                 for i in xs])
        # xzs_data = torch.from_numpy(xzs)

        pi, sigma, mu = self.model(tensor_xs)
        if b_generate_samples:
            samples = sample(pi, sigma, mu).data.numpy().reshape(-1)
            for i in range(num_points-1):
                samples = np.vstack(
                    (samples, sample(pi, sigma, mu).data.numpy().reshape(-1)))
            samples = np.mean(samples, axis=0)
        else:
            # print("len",len(xs))
            # mu1 = mu.detach().numpy()
            # pi1 = pi.detach().numpy()
            # print("mu", mu1)
            # print("weight", pi1)

            mu = mu.detach().numpy().reshape(len(xs),-1)
            pi = pi.detach().numpy()#.reshape(-1,2)
            # print("mu",mu)
            # print("weight",pi)
            # xymul=
            # print(xymul)
            samples = np.sum(np.multiply(pi,mu),axis=1)

        # print("small",samples)

        if self.is_normalized:
            # de-normalize the data
            samples = [self.denormalize(
                i, self.meany, self.widthy) for i in samples]
            xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                           for i in xs])
            # print("large",samples)

        if b_show_plot:
            if not self.is_training_data_denormalized:
                self.xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                                    for i in self.xs])
                self.ys = np.array([self.denormalize(i, self.meany, self.widthy)
                                    for i in self.ys])
                self.is_training_data_denormalized = True
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(self.xs,  self.ys)
            ax1.scatter(xs, samples)
            ax1.set_xlabel('query range attribute')
            ax1.set_ylabel('aggregate attribute')
            plt.show()

        samples = list(samples)
        return samples

    def kde_predict(self, z, xs):
        if self.is_normalized:
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])

        # xs = xs[:, np.newaxis]
        tensor_xs = torch.stack([torch.Tensor(i)
                                 for i in xs])
        # xzs_data = torch.from_numpy(xzs)

        pi, sigma, mu = self.model(tensor_xs)
        mu = mu.detach().numpy().reshape(len(xs), -1)[0]
        pi = pi.detach().numpy()[0]  # .reshape(-1,2)
        sigma = sigma.detach().numpy().reshape(len(sigma),-1)[0]
        return gm(pi,mu,sigma,)

    def normalize(self, x, mean, width):
        return (x-mean)/width*2

    def denormalize(self, x, mean, width):
        return 0.5*width*x + mean


def test1():
    x = np.random.uniform(low=1, high=10, size=(1000,))
    # z = np.random.uniform(low=1, high=10, size=(1000,))
    z = np.random.randint(0, 7, size=(1000,))
    noise = np.random.normal(1, 5, 1000)
    y = x**2 - z**2 + noise
    print(min(x), max(x))
    print(min(y), max(y))

    xz = np.concatenate((x[:, np.newaxis], z[:, np.newaxis]), axis=1)

    regMdn = RegMdn(dim_input=2)
    # regMdn.fit(xz, y, num_epoch=200, b_show_plot=False)
    regMdn.fit(xz,  y, num_epoch=400, b_show_plot=False)

    x_test = np.random.uniform(low=1, high=10, size=(500,))
    z_test = np.random.randint(0, 7, size=(500,))
    xz_test = np.concatenate(
        (x_test[:, np.newaxis], z_test[:, np.newaxis]), axis=1)
    regMdn.predict(xz_test, b_show_plot=True)
    # regMdn.predict([1,2],b_show_plot=True)
    # regMdn.predict([3,4], b_show_plot=True)
    # regMdn.predict([5,6],b_show_plot=True)
    # regMdn.predict([7,8],b_show_plot=True)


def test_pm25_2d():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    # file = "/home/u1796377/Programs/dbestwarehouse/pm25_torch_2k.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['pm25', 'PRES'])
    df_train = df.head(1000)
    df_test = df.tail(1000)
    pres_train = df_train.PRES.values[:,np.newaxis]
    pm25_train = df_train.pm25.values
    pres_test = df_test.PRES.values
    pm25_test = df_test.pm25.values

    regMdn = RegMdn(dim_input=1)
    regMdn.fit(pres_train, pm25_train, num_epoch=100, b_show_plot=False)
    print(regMdn.predict([[1000], [1005],[1010], [1015],[1020], [1025],[1030], [1035]], b_show_plot=True))
    print(regMdn.predict([[1000.5], [1005.5], [1010.5], [1015.5], [1020.5], [1025.5], [1030.5], [1035.5]], b_show_plot=True))
    xxs = np.linspace(np.min(pres_train),np.max(pres_train),100)
    # print(regMdn.predict(xxs,b_show_plot=True))



def test_pm25_3d():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['pm25', 'PRES', 'TEMP'])
    df_train = df.head(1000)
    df_test = df.tail(1000)
    pres_train = df_train.PRES.values
    temp_train = df_train.TEMP.values
    pm25_train = df_train.pm25.values
    pres_test = df_test.PRES.values
    pm25_test = df_test.pm25.values
    temp_test = df_test.TEMP.values
    xzs_train = np.concatenate(
        (temp_train[:, np.newaxis], pres_train[:, np.newaxis]), axis=1)
    xzs_test = np.concatenate(
        (temp_test[:, np.newaxis], pres_test[:, np.newaxis]), axis=1)
    regMdn = RegMdn(dim_input=2)
    regMdn.fit(xzs_train, pm25_train, num_epoch=400, b_show_plot=False)
    print(regMdn.predict(xzs_test, b_show_plot=True))
    regMdn.predict(xzs_train, b_show_plot=True)


def test_pm25_2d_density():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['TEMP', 'PRES'])
    df_train = df.head(1000)
    df_test = df.tail(1000)
    pres_train = df_train.PRES.values[:,np.newaxis]
    temp_train = df_train.PRES.values #df_train.pm25.values
    pres_test = df_test.PRES.values
    temp_test = df_test.PRES.values  #df_test.pm25.values

    regMdn = RegMdn(dim_input=1)
    regMdn.fit(pres_train, temp_train, num_epoch=400, b_show_plot=False)
    # regMdn.predict(pres_train, b_show_plot=True)
    # regMdn.predict(pres_test, b_show_plot=True)
    regMdn.kde_predict(1020,[[10]])
    print("finished")


def test_pm25_3d_density():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['pm25', 'PRES', 'TEMP'])
    df_train = df.head(1000)
    df_test = df.tail(1000)
    pres_train = df_train.PRES.values
    temp_train = df_train.TEMP.values
    pm25_train = df_train.PRES.values  # df_train.pm25.values
    pres_test = df_test.PRES.values
    pm25_test = df_test.pm25.values
    temp_test = df_test.PRES.values  # df_test.TEMP.values
    xzs_train = np.concatenate(
        (temp_train[:, np.newaxis], pres_train[:, np.newaxis]), axis=1)
    xzs_test = np.concatenate(
        (temp_test[:, np.newaxis], pres_test[:, np.newaxis]), axis=1)
    regMdn = RegMdn(dim_input=2)
    regMdn.fit(xzs_train, pm25_train, num_epoch=1000, b_show_plot=True)
    regMdn.predict(xzs_test, b_show_plot=True)
    regMdn.predict(xzs_train, b_show_plot=True)

def test_gm():
    from sklearn import mixture
    import random
    kde = mixture.GaussianMixture(n_components=2, covariance_type='spherical')
    kde.fit(np.random.rand(100, 1))
    # x = np.array(np.linspace(-5, 15, 100)).reshape(-1, 1)
    # print(x)
    # y = kde.predict(x)

    kde.weights_=np.array([0.5,0.5])
    kde.means_ = np.array([[1] ,[3]])
    kde.covariances_= np.array([10,20])

    x = np.array(np.linspace(-5,15,1000)).reshape(-1, 1)
    print(x)
    y = kde.score_samples(x)
    y=np.exp(y)
    print(y)

    plt.plot(x,y)
    plt.show()

def gm(weights, mus, vars, x):
    result = 0
    for index in range(len(weights)):
        result +=stats.norm(mus[index], vars[index]).pdf(x) * weights[index]
    return result

def test_gmm():
    weights = [0.5, 0.5]
    mus = [0,10]
    vars = [4, 4]
    xs = np.array(np.linspace(-5, 15, 1000))
    results = [gm(weights,mus,vars,x) for x in xs]
    plt.plot(xs,results)
    plt.show()


if __name__ == "__main__":
    # test_pm25_2d_density()
    test_pm25_2d_density()
