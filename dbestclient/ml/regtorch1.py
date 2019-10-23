'''
 Created by Qingzhi Ma on Wed Oct 23 2019
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

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


np.random.seed(42)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# def generate_data(b_show_plot=True):
#     n = 2500
#     d = 1
#     t = 1
#     x_train = np.random.uniform(0, 1, (n, d)).astype(np.float32)
#     noise = np.random.uniform(-0.1, 0.1, (n, d)).astype(np.float32)
#     y_train = x_train + 0.3*np.sin(2*np.pi*x_train) + noise
#     x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)

#     x_train_inv = y_train
#     y_train_inv = x_train
#     # new x has a slightly different range
#     x_test = np.linspace(-0.1, 1.1, n).reshape(-1, 1).astype(np.float32)
#     if b_show_plot:
#         fig = plt.figure(figsize=(8, 8))
#         plt.plot(x_train_inv, y_train_inv, 'go',
#                  alpha=0.5, markerfacecolor='none')
#         plt.show()
#     return x_train_inv, y_train_inv, x_test


# if __name__ == "__main__":
n = 2500
d = 1
t = 1
x_train = np.random.uniform(0, 1, (n, d)).astype(np.float32)
noise = np.random.uniform(-0.1, 0.1, (n, d)).astype(np.float32)
y_train = x_train + 0.3*np.sin(2*np.pi*x_train) + noise
x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)


x_train_inv = y_train
y_train_inv = x_train
# new x has a slightly different range
x_test = np.linspace(-0.1, 1.1, n).reshape(-1, 1).astype(np.float32)


# dimensionality of hidden layer
h = 50
# K mixing components (PRML p. 274)
# Can also formulate as a K-dimensional, one-hot
# encoded, latent variable $$z$$, and have the model
# produce values for $$\mu_k = p(z_k = 1)$$, i.e., the
# prob of each possible state of $$z$$. (PRML p. 430)
k = 30  # 3
# We specialize to the case of isotropic covariances (PRML p. 273),
# so the covariance matrix is diagonal with equal diagonal elements,
# i.e., the variances for each dimension of y are equivalent.
# therefore, the MDN outputs pi & sigma scalars for each mixture
# component, and a mu vector for each mixture component containing
# means for each target variable.
# NOTE: we could use the shorthand `d_out = 3*k`, since our target
# variable for this project only has a dimensionality of 1, but
# the following is more general.
# d_out = (t + 2) * k  # t is L from PRML p. 274
# NOTE: actually cleaner to just separate pi, sigma^2, & mu into
# separate functions.
d_pi = k
d_sigmasq = k
d_mu = t * k

w1 = Variable(torch.randn(d, h) * np.sqrt(2/(d+h)), requires_grad=True)
b1 = Variable(torch.zeros(1, h), requires_grad=True)
w_pi = Variable(torch.randn(h, d_pi) * np.sqrt(2/(d+h)), requires_grad=True)
b_pi = Variable(torch.zeros(1, d_pi), requires_grad=True)
w_sigmasq = Variable(torch.randn(h, d_sigmasq) *
                     np.sqrt(2/(d+h)), requires_grad=True)
b_sigmasq = Variable(torch.zeros(1, d_sigmasq), requires_grad=True)
w_mu = Variable(torch.randn(h, d_mu) * np.sqrt(2/(d+h)), requires_grad=True)
b_mu = Variable(torch.zeros(1, d_mu), requires_grad=True)


def forward(x):
    out = torch.tanh(x.mm(w1) + b1)  # shape (n, h)
    # out = F.leaky_relu(x.mm(w1) + b1)  # interesting possibility
    # p(z_k = 1) for all k; K mixing components that sum to 1; shape (n, k)
    pi = F.softmax(out.mm(w_pi) + b_pi, dim=1)
    # K gaussian variances, which must be >= 0; shape (n, k)
    sigmasq = torch.exp(out.mm(w_sigmasq) + b_sigmasq)
    mu = out.mm(w_mu) + b_mu  # K * L gaussian means; shape (n, k*t)
    return pi, sigmasq, mu


def gaussian_pdf(x, mu, sigmasq):
    # NOTE: we could use the new `torch.distributions` package for this now
    return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * torch.norm((x-mu), 2, 1)**2)


def loss_fn(pi, sigmasq, mu, target):
    # PRML eq. 5.153, p. 275
    # compute the likelihood p(y|x) by marginalizing p(z)p(y|x,z)
    # over z. for now, we assume the prior p(w) is equal to 1,
    # although we could also include it here.  to implement this,
    # we average over all examples of the negative log of the sum
    # over all K mixtures of p(z)p(y|x,z), assuming Gaussian
    # distributions.  here, p(z) is the prior over z, and p(y|x,z)
    # is the likelihood conditioned on z and x.
    losses = Variable(torch.zeros(n))  # p(y|x)
    for i in range(k):  # marginalize over z
        likelihood_z_x = gaussian_pdf(
            target, mu[:, i*t:(i+1)*t], sigmasq[:, i])
        prior_z = pi[:, i]
        losses += prior_z * likelihood_z_x
    loss = torch.mean(-torch.log(losses))
    return loss


opt = optim.Adam([w1, b1, w_pi, b_pi, w_sigmasq,
                  b_sigmasq, w_mu, b_mu], lr=0.008)

# wrap up the inverse data as Variables
x = Variable(torch.from_numpy(x_train_inv))
y = Variable(torch.from_numpy(y_train_inv))

for e in range(500):
    opt.zero_grad()
    pi, sigmasq, mu = forward(x)
    loss = loss_fn(pi, sigmasq, mu, y)
    if e % 100 == 0:
        print(loss.item())
    loss.backward()
    opt.step()


def sample_mode(pi, sigmasq, mu):
    # for prediction, could use conditional mode, but it doesn't
    # have an analytical solution (PRML p. 277). alternative is
    # to return the mean vector of the most probable component,
    # which is the approximate conditional mode from the mixture
    # NOTE: this breaks autograd, but that's fine because we
    # won't be computing gradients for this path
    # NOTE: pi, sigmasq, & mu are tensors
    n, k = pi.shape
    _, kt = mu.shape
    t = int(kt / k)
    # mixture w/ largest prob, i.e., argmax_k p(z==1)
    _, max_component = torch.max(pi, 1)
    out = Variable(torch.zeros(n, t))
    for i in range(n):
        for j in range(t):
            out[i, j] = mu[i, max_component.data[i]*t+j]
    return out


def sample_preds(pi, sigmasq, mu, samples=10):
    # rather than sample the single conditional mode at each
    # point, we could sample many points from the GMM produced
    # by the model for each point, yielding a dense set of
    # predictions
    N, K = pi.shape
    _, KT = mu.shape
    T = int(KT / K)
    out = Variable(torch.zeros(N, samples, T))  # s samples per example
    for i in range(N):
        for j in range(samples):
            # pi must sum to 1, thus we can sample from a uniform
            # distribution, then transform that to select the component
            u = np.random.uniform()  # sample from [0, 1)
            # split [0, 1] into k segments: [0, pi[0]), [pi[0], pi[1]), ..., [pi[K-1], pi[K])
            # then determine the segment `u` that falls into and sample from that component
            prob_sum = 0
            for k in range(K):
                prob_sum += pi.data[i, k]
                if u < prob_sum:
                    # sample from the kth component
                    for t in range(T):
                        sample = np.random.normal(
                            mu.data[i, k*T+t], np.sqrt(sigmasq.data[i, k]))
                        out[i, j, t] = sample
                    break
    return out


# sample
pi, sigmasq, mu = forward(Variable(torch.from_numpy(x_test)))
cond_mode = sample_mode(pi, sigmasq, mu)
preds = sample_preds(pi, sigmasq, mu, samples=10)

# plot the conditional mode at each point along x
fig = plt.figure(figsize=(8, 8))
plt.plot(x_train_inv, y_train_inv, 'go', alpha=0.5, markerfacecolor='none')
plt.plot(x_test, cond_mode.data.numpy(), 'r.')
plt.show()

# plot the means at each point along x
fig = plt.figure(figsize=(8, 8))
plt.plot(x_train_inv, y_train_inv, 'go', alpha=0.5, markerfacecolor='none')
plt.plot(x_test, mu.data.numpy(), 'r.')
plt.show()

# plot sampled predictions at each point along x
fig = plt.figure(figsize=(8, 8))
plt.plot(x_train_inv, y_train_inv, 'go', alpha=0.5, markerfacecolor='none')
for i in range(preds.shape[1]):
    plt.plot(x_test, preds.data.numpy()[:, i].reshape(n, 1), 'r.', alpha=0.3)
plt.show()

print(pi)
