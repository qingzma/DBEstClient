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

    def fit(self, xs, ys, b_show_plot=False, b_normalize=True, num_epoch=400):
        """ fit a regression y= R(x)"""
        if self.dim_input == 1:
            return self.fit2d(xs, ys, b_show_plot=b_show_plot,
                              b_normalize=b_normalize, num_epoch=num_epoch)
        elif self.dim_input == 2:
            return self.fit3d(xs[:, 0], xs[:, 1], ys, b_show_plot=b_show_plot,
                              b_normalize=b_normalize, num_epoch=num_epoch)
        else:
            print("dimension mismatch")
            sys.exit(0)

    def predict(self, xs, b_show_plot=True):
        """ make predictions"""
        if self.dim_input == 1:
            return self.predict2d(xs, b_show_plot=b_show_plot)
        elif self.dim_input == 2:
            return self.predict3d(xs[:, 0], xs[:, 1], b_show_plot=b_show_plot)
        else:
            print("dimension mismatch")
            sys.exit(0)

    def fit3d(self, xs, zs, ys, b_show_plot=False, b_normalize=True, num_epoch=200):
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
            nn.Linear(self.dim_input, 20),
            nn.Tanh(),
            nn.Dropout(0.01),
            MDN(20, 1, 10)
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

    def fit2d(self, xs, ys, b_show_plot=False, b_normalize=True, num_epoch=200):
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
        xs = xs[:, np.newaxis]
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
            MDN(20, 1, 10)
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

    def predict3d(self, xs, zs, b_show_plot=True):
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
        samples = sample(pi, sigma, mu).data.numpy().reshape(-1)

        # print(samples.data.numpy().reshape(-1))

        # print(x_test)
        if b_show_plot:
            if self.is_normalized:
                # de-normalize the data
                samples = [self.denormalize(
                    i, self.meany, self.widthy) for i in samples]
                xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                            for i in xs])
                zs = np.array([self.denormalize(i, self.meanz, self.widthz)
                            for i in zs])
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

    def predict2d(self, xs, b_show_plot=True):
        if self.is_normalized:
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])

        xs = xs[:, np.newaxis]
        tensor_xs = torch.stack([torch.Tensor(i)
                                 for i in xs])
        # xzs_data = torch.from_numpy(xzs)

        pi, sigma, mu = self.model(tensor_xs)
        # print("mu,", mu)
        # print("sigma", sigma)
        samples = sample(pi, sigma, mu).data.numpy().reshape(-1)

        if b_show_plot:
            if self.is_normalized:
                # de-normalize the data
                samples = [self.denormalize(
                    i, self.meany, self.widthy) for i in samples]
                xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                            for i in xs])
            if not self.is_training_data_denormalized:
                self.xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                                    for i in self.xs])
                self.ys = np.array([self.denormalize(i, self.meany, self.widthy)
                                    for i in self.ys])
                self.is_training_data_denormalized=True
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self.xs,  self.ys)
            ax.scatter(xs, samples)
            ax.set_xlabel('query range attribute')
            ax.set_ylabel('aggregate attribute')
            plt.show()
        return samples

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

    regMdn = RegMdn(dim_input=1)
    # regMdn.fit(xz, y, num_epoch=200, b_show_plot=False)
    regMdn.fit(x,  y, num_epoch=400, b_show_plot=False)

    x_test = np.random.uniform(low=1, high=10, size=(500,))
    z_test = np.random.randint(0, 7, size=(500,))
    xz_test = np.concatenate(
        (x_test[:, np.newaxis], z_test[:, np.newaxis]), axis=1)
    # regMdn.predict(xz_test)
    regMdn.predict([1,2],b_show_plot=True)
    regMdn.predict([3,4], b_show_plot=True)
    regMdn.predict([5,6],b_show_plot=True)
    regMdn.predict([7,8],b_show_plot=True)


def test_pm25():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['pm25', 'PRES'])
    df_train = df.head(10000)
    df_test = df.tail(10000)
    pres_train = df_train.PRES.values
    pm25_train = df_train.pm25.values
    pres_test = df_test.PRES.values
    pm25_test = df_test.pm25.values

    regMdn = RegMdn(dim_input=1)
    regMdn.fit(pres_train, pm25_train, num_epoch=400, b_show_plot=False)
    regMdn.predict(pres_test, b_show_plot=True)


if __name__ == "__main__":
    test1()
