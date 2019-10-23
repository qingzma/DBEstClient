'''
 Created by Qingzhi Ma on Tue Oct 22 2019

 Copyright (c) 2019 Department of Computer Science, University of Warwick
 Copyright 2019 Qingzhi Ma

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from __future__ import print_function, division
import pandas as pd
import math
from skimage import io, transform
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

import warnings
# from dbestclient.ml import mdn
# warnings.filterwarnings("ignore")


# plt.rcParams['figure.figsize'] = (8, 8)
# plt.ion()
np.random.seed(42)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians, device=None):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

        if device is not None:
            self.z_h.to(device)
            self.z_pi.to(device)
            self.z_mu.to(device)
            self.z_sigma.to(device)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = 1000
    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(-10, 10, n_samples)
    y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)
    x_test = torch.linspace(-15, 15, n_samples).view(-1, 1)

    # plt.figure(figsize=(8, 8))
    # plt.scatter(x_data, y_data, alpha=0.4)
    # plt.show()
    model = MDN(n_hidden=20, n_gaussians=10)

    optimizer = torch.optim.Adam(model.parameters())

    def mdn_loss_fn(y, mu, sigma, pi):
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = torch.exp(m.log_prob(y))
        loss = torch.sum(loss * pi, dim=1)
        loss = -torch.log(loss)
        return torch.mean(loss)

    for epoch in range(10000):
        pi, mu, sigma = model(x_data)
        loss = mdn_loss_fn(y_data, mu, sigma, pi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(loss.data.tolist())
    pi, mu, sigma = model(x_test)
    # print(pi)
    # print(mu)
    # print(sigma)
    # print(pi.size())
    # print(mu.size())
    # print(sigma.size())

    k = torch.multinomial(pi, 1).view(-1)
    print(k)
    y_pred = torch.normal(mu, sigma)[np.arange(n_samples), k].data

    # print(y_pred)
    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data, alpha=0.4)
    plt.scatter(x_test, y_pred, alpha=0.4, color='red')
    plt.show()

    # generate data
    # n = 2500
    # d = 1
    # t = 1
    # x_train = np.random.uniform(0, 1, (n, d)).astype(np.float32)
    # noise = np.random.uniform(-0.1, 0.1, (n, d)).astype(np.float32)
    # y_train = x_train + 0.3*np.sin(2*np.pi*x_train) + noise
    # x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)

    # mdn = MDN()
    # mdn.fit(x_train, x_test)

    # # plot
    # fig = plt.figure(figsize=(8, 8))
    # plt.plot(x_train, y_train, 'go', alpha=0.5, markerfacecolor='none')
    # plt.show()

    # # define a simple neural net
    # h = 15
    # w1 = Variable(torch.randn(d, h))


# class MDN(nn.Module):
#     """ Mixture Density Network """

#     def __init__(self, d=1, t=1, h=15):
#         super(MDN, self).__init__()
#         self.w1 = Variable(torch.randn(d, h) *
#                            np.sqrt(1/d), requires_grad=True)
#         self.b1 = Variable(torch.zeros(1, h), requires_grad=True)
#         self.w2 = Variable(torch.randn(h, t) *
#                            np.sqrt(1/h), requires_grad=True)
#         self.b2 = Variable(torch.zeros(1, t), requires_grad=True)

#     def forward(self, x):
#         # a relu introduces kinks in the predicted curve
#         self.out = torch.tanh(x.mm(self.w1) + self.b1)
#         self.out = self.out.mm(self.w2) + self.b2
#         return self.out


# class regMDN():
#     def __init__(self):
#         pass

#     def fit(self, x_train, y_train, epochs=1000):
#         # wrap up the data as Variables
#         x = Variable(torch.from_numpy(x_train))
#         y = Variable(torch.from_numpy(y_train))
#         opt = optim.Adam([self.w1, self.b1, self.w2, self.b2], lr=1e-3)

#         for epoch in range(epochs):
#             opt.zero_grad()
#             self.out.forward(x)
#             loss = F.mse_loss(self.out, y)
#             if epoch % 100 == 0:
#                 print(epoch, loss.data[0])
#             loss.backward()
#             opt.step()
def test1():

    n_samples = 1000
    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(-10, 10, n_samples)
    y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)
    x_test = torch.linspace(-15, 15, 1000).view(-1, 1)

    from dbestclient.ml.mdn import MixtureDensityNetwork
    model = MixtureDensityNetwork(1, 1, 20)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(10000):
        optimizer.zero_grad()
        loss = model.loss(x_data, y_data).mean()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(loss.data.tolist())
    y_pred = model.sample(x_test)

    # k = torch.multinomial(pi, 1).view(-1)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data, alpha=0.4)
    plt.scatter(x_test, y_pred, alpha=0.4, color='red')
    plt.show()


if __name__ == "__main__":
    test()
