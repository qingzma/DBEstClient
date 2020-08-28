# Created by Qingzhi Ma
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk


import itertools as it
import math
import random
import sys
from concurrent import futures
from copy import deepcopy
from os import remove
from os.path import abspath

import category_encoders as ce
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.multiprocessing import Pool

from dbestclient.ml.integral import approx_count, prepare_reg_density_data
from dbestclient.ml.embedding import   columns2sentences,WordEmbedding


# https://www.katnoria.com/mdn/
# https://github.com/sagelywizard/pytorch-mdn
"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""


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

    def __init__(self, in_features, out_features, num_gaussians, device):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

        self.pi = self.pi.to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


# ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


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
    ret = 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 *
                                                   ((data - mu) / sigma) ** 2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target, device):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)

    nll = -torch.log(torch.sum(prob, dim=1)).to(device)
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


def gaussion_predict(weights: list, mus: list, sigmas: list, xs: list, n_jobs=1):
    if n_jobs == 1:
        result = np.array([np.multiply(stats.norm(mus, sigmas).pdf(x),
                                       weights).sum(axis=1).tolist() for x in xs]).transpose()
    else:
        with Pool(processes=n_jobs) as pool:
            instances = []
            results = []
            for x in xs:
                i = pool.apply_async(
                    gaussion_predict, (weights, mus, sigmas, [x], 1))
                instances.append(i)
            for i in instances:
                result = i.get()
                # print("partial result", result)
                results.append(result)
            result = np.concatenate(results, axis=1)

            # with futures.ThreadPoolExecutor() as executor:
            #     for x in xs:
            #         future = executor.submit(
            #             gaussion_predict, weights, mus, sigmas, [x], 1)
            #         results.append(future.result())
            # result = np.concatenate(results, axis=1)
    return result


def gm(weights: list, mus: list, vars: list, x: list, b_plot=False, n_division=100):
    """ given a list of points, calculate the gaussian mixture probability

    Args:
        weights (list): weights
        mus (list): the centroids of gaussions.
        vars (list): the variances.
        x (list): the targeting points.
        b_plot (bool, optional): whether return the value for plotting. Defaults to False.
        n_division (int, optional): number of division, if b_plot=True. Defaults to 100.

    Returns:
        float: the pdf of a gaussian mixture.
    """
    if not b_plot:
        result = [stats.norm(mu_i, vars_i).pdf(
            x)*weights_i for mu_i, vars_i, weights_i in zip(mus, vars, weights)]
        result = sum(result)
        # result = 0
        # for index in range(len(weights)):
        #     result += stats.norm(mus[index], vars[index]
        #                          ).pdf(x) * weights[index]
        # print(result)
        return result
    else:
        xs = np.linspace(-1, 1, n_division)
        # ys = [gm(weights, mus, vars, xi, b_plot=False) for xi in xs]
        ys = gm(weights, mus, vars, xs, b_plot=False)
        return xs, ys
        # plt.plot(xs, ys)
        # plt.show()


def normalize(x_point: float, mean: float, width: float) -> float:
    """normalize the data point

    Args:
        x (float): the data point
        mean (float): the mean value
        width (float): the width

    Returns:
        float: the normalized value
    """
    return (x_point - mean) / width * 2


def denormalize(x_point: float, mean: float, width: float) -> float:
    """de-normalize the data point

    Args:
        x (float): the data point
        mean (float): the mean value
        width (float): the width

    Returns:
        float: the de-normalized value
    """
    return 0.5 * width * x_point + mean


def de_serialize(file: str):
    """de-serialize the model from a file.

    Args:
        file (str): the file path.

    Returns:
        Callable: the model.
    """
    with open(file, 'rb') as f:
        return dill.load(f)


class GenericMdn:
    def __init__(self, config):
        self.meanx = None
        self.widthx = None
        self.config = config

    def fit(self, runtime_config):
        raise NotImplementedError("Method fit() is not implemented.")

    def fit_grid_search(self, runtime_config):
        raise NotImplementedError(
            "Method fit_grid_search() is not implemented.")

    def predict(self, runtime_config):
        raise NotImplementedError("Method predict() is not implemented.")

    def normalize(self, xs: np.array):
        """normalize the data

        Args:
            x (list): the data points to be normalized.
            mean (float): the mean value of x.
            width (float): the range of x.

        Returns:
            list: the normalized data.
        """
        return (xs - self.meanx) / self.widthx * 2

    def denormalize(self, xs):
        """de-normalize the data

        Args:
            x (list): the data points to be de-normalized.
            mean (float): the mean value of x.
            width (float): the range of x.

        Returns:
            list: the de-normalized data.
        """
        return 0.5 * self.widthx * xs + self.meanx


class RegMdnGroupBy():
    """ This class implements the regression using mixture density network for group by queries.
    """

    def __init__(self, config,  b_store_training_data=False,  b_normalize_data=True):
        if b_store_training_data:
            self.x_points = None  # query range
            self.y_points = None  # aggregate value
            self.z_points = None  # group by balue
        self.sample_x = None        # used in the score() function
        self.sample_g = None
        self.sample_average_y = None
        self.b_store_training_data = b_store_training_data
        self.meanx = None
        self.widthx = None
        self.meany = None
        self.widthy = None
        self.model = None
        self.last_xs = None
        self.last_pi = None
        self.last_mu = None
        self.last_sigma = None
        self.config = config
        self.b_normalize_data = b_normalize_data
        self.enc = None

    def fit(self, z_group: list, x_points: list, y_points: list, runtime_config, lr: float = 0.001, n_workers=0):
        """fit the MDN regression model.

        Args:
            z_group (list): group by values
            x_points (list): x points
            y_points (list): y points
            n_epoch (int, optional): number of epochs for training. Defaults to 100.
            n_gaussians (int, optional): the number of gaussions. Defaults to 5.
            n_hidden_layer (int, optional): the number of hidden layers. Defaults to 1.
            n_mdn_layer_node (int, optional): the node number in the hidden layer. Defaults to 10.
            lr (float, optional): the learning rate of the MDN network for training. Defaults to 0.001.

        Raises:
            ValueError: The hidden layer should be 1 or 2.            

        Returns:
            RegMdnGroupBy: The regression model.
        """
        n_epoch = self.config.config["n_epoch"]
        n_gaussians = self.config.config["n_gaussians_reg"]
        n_hidden_layer = self.config.config["n_hidden_layer"]
        n_mdn_layer_node = self.config.config["n_mdn_layer_node_reg"]
        b_grid_search = self.config.config["b_grid_search"]
        encoder = self.config.config["encoder"]
        device = runtime_config["device"]

        if not b_grid_search:
            if encoder == "onehot":
                self.enc = OneHotEncoder(handle_unknown='ignore')
                zs_encoded = z_group
                zs_encoded = self.enc.fit_transform(zs_encoded).toarray()
            elif encoder == "binary":
                # print(z_group)
                # prepare column names for binary encoding
                columns = list(range(len(z_group[0])))
                self.enc = ce.BinaryEncoder(cols=columns)
                zs_encoded = self.enc.fit_transform(z_group).to_numpy()
            elif encoder == "embedding":
                sentences = columns2sentences(z_group, x_points, y_points)
                self.enc = WordEmbedding()
                self.enc.fit(sentences, gbs=["gb"],dim=self.config.config["n_embedding_dim"])
                gbs_data = z_group.reshape(1,-1)[0]
                zs_encoded = self.enc.predicts(gbs_data)
                # raise TypeError("embedding is not supported yet.")

            if self.b_normalize_data:
                self.meanx = (np.max(x_points) + np.min(x_points)) / 2
                self.widthx = np.max(x_points) - np.min(x_points)
                self.meany = (np.max(y_points) + np.min(y_points)) / 2
                self.widthy = np.max(y_points) - np.min(y_points)

                x_points = np.array([normalize(i, self.meanx, self.widthx)
                                     for i in x_points])
                y_points = np.array([normalize(i, self.meany, self.widthy)
                                     for i in y_points])
            if self.b_store_training_data:
                self.x_points = x_points
                self.y_points = y_points
                self.z_points = z_group
            else:
                # delete the previous stored data in grid search, to save space.
                self.x_points = None
                self.y_points = None
                self.z_points = None

            if encoder in ["onehot", "binary", "embedding"]:
                xs_encoded = x_points[:, np.newaxis]
                xzs_encoded = np.concatenate(
                    [xs_encoded, zs_encoded], axis=1).tolist()
                tensor_xzs = torch.stack([torch.Tensor(i)
                                          for i in xzs_encoded])

            else:
                xzs = [[x_point, z_point]
                       for x_point, z_point in zip(x_points, z_group)]
                tensor_xzs = torch.stack([torch.Tensor(i)
                                          for i in xzs])  # transform to torch tensors
            y_points = y_points[:, np.newaxis]
            tensor_ys = torch.stack([torch.Tensor(i) for i in y_points])

            # move variables to cuda
            tensor_xzs = tensor_xzs.to(device)
            tensor_ys = tensor_ys.to(device)

            my_dataset = torch.utils.data.TensorDataset(
                tensor_xzs, tensor_ys)  # create your dataloader
            my_dataloader = torch.utils.data.DataLoader(
                my_dataset, batch_size=self.config.config["batch_size"], shuffle=True, num_workers=n_workers)

            if encoder == "onehot":
                input_dim = sum([len(i) for i in self.enc.categories_]) + 1
            elif encoder == "binary":
                input_dim = len(self.enc.base_n_encoder.feature_names) + 1
            elif encoder == "embedding":
                input_dim = self.enc.dim + 1
            else:
                raise ValueError("Encoding should be binary or onehot")

            # initialize the model
            if n_hidden_layer == 1:
                self.model = nn.Sequential(
                    nn.Linear(input_dim, n_mdn_layer_node),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    MDN(n_mdn_layer_node, 1, n_gaussians, device)
                )
            elif n_hidden_layer == 2:
                self.model = nn.Sequential(
                    nn.Linear(input_dim, n_mdn_layer_node),
                    nn.Tanh(),
                    nn.Linear(n_mdn_layer_node, n_mdn_layer_node),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    MDN(n_mdn_layer_node, 1, n_gaussians, device)
                )
            else:
                raise ValueError(
                    "The hidden layer should be 1 or 2, but you provided "+str(n_hidden_layer))

            self.model = self.model.to(device)

            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            decay_rate = 0.96
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=decay_rate)
            for epoch in range(n_epoch):
                if runtime_config["v"]:
                    if epoch % 1 == 0:
                        print("< Epoch {}".format(epoch))
                # train the model
                for minibatch, labels in my_dataloader:
                    minibatch.to(device)
                    labels.to(device)
                    self.model.zero_grad()
                    pi, sigma, mu = self.model(minibatch)
                    loss = mdn_loss(pi, sigma, mu, labels, device)
                    loss.backward()
                    optimizer.step()
                my_lr_scheduler.step()
            self.model.eval()
            print("Finish regression training.")
            return self
        else:
            return self.fit_grid_search(z_group, x_points, y_points, runtime_config)

    def fit_grid_search(self, z_group: list, x_points: list, y_points: list, runtime_config):
        """use grid search to tune the hyper parameters.

        Args:
            z_group (list): group by values
            x_points (list): independent values
            y_points (list): dependent values

        Returns:
            RegMdnGroupBy: the fitted model
        """
        param_grid = {'epoch': [5], 'lr': [0.001], 'node': [
            5, 10, 20], 'hidden': [1, 2], 'gaussian_reg': [3, 5]}
        # param_grid = {'epoch': [5], 'lr': [0.001], 'node': [
        #     5], 'hidden': [1], 'gaussian': [3]}

        errors = []
        combinations = it.product(*(param_grid[Name] for Name in param_grid))
        combinations = list(combinations)
        combs = []
        for combination in combinations:
            idx = 0
            comb = {}
            # print(combination)
            for key in param_grid:
                comb[key] = combination[idx]
                idx += 1
            combs.append(comb)

        self.b_store_training_data = True
        for para in combs:
            print("Grid search for parameter set :", para)
            config = self.config.copy()
            config.config["n_gaussians_reg"] = para['gaussian_reg']
            # config.config["n_gaussians_density"] = para['gaussian_density']
            config.config["n_epoch"] = para['epoch']
            config.config["n_hidden_layer"] = para['hidden']
            config.config["n_mdn_layer_node_reg"] = para['node']
            config.config["b_grid_search"] = False

            instance = RegMdnGroupBy(config, b_store_training_data=True).fit(z_group, x_points, y_points,
                                                                             runtime_config, lr=para['lr'])
            errors.append(instance.score(runtime_config))

        print("errors for grid search ", errors)
        index = errors.index(min(errors))
        para = combs[index]
        print("Finding the best configuration for the network", para)

        self.b_store_training_data = False
        # release space
        self.x_points = None
        self.y_points = None
        self.z_points = None
        self.sample_x = None
        self.sample_g = None
        self.sample_average_y = None

        config = self.config.copy()
        config.config["n_gaussians_reg"] = para['gaussian_reg']
        # config.config["n_gaussians_density"] = para['gaussian_density']
        # config.config["n_epoch"] = para['epoch']
        config.config["n_hidden_layer"] = para['hidden']
        config.config["n_mdn_layer_node_reg"] = para['node']
        config.config["b_grid_search"] = False

        instance = RegMdnGroupBy(config).fit(z_group, x_points, y_points,
                                             runtime_config, lr=para['lr'])
        print("-"*80)
        return instance

    def predict(self, z_group: list, x_points: list, runtime_config, b_plot=False) -> list:
        """provide predictions for given groups and points.

        Args:
            z_group (list): the group by values
            x_points (list): the corresponding x points
            b_plot (bool, optional): to plot the data or not.. Defaults to False.

        Raises:
            Exception: [description]

        Returns:
            list: the predictions.
        """
        # torch.set_num_threads(4)
        # check input data type, and convert to np.array
        if type(z_group) is list:
            z_group = np.array(z_group)
        if type(x_points) is list:
            x_points = np.array(x_points)
        encoder = self.config.config["encoder"]
        device = runtime_config["device"]

        if encoder == 'no':
            convert2float = True
            if convert2float:
                try:
                    zs_float = []
                    for item in z_group:
                        if item[0] == "":
                            zs_float.append([0.0])
                        else:
                            zs_float.append([(float)(item[0])])
                    z_group = zs_float
                except:
                    raise Exception

        if self.b_normalize_data:
            x_points = normalize(x_points, self.meanx, self.widthx)

        if encoder == "onehot":
            # zs_encoded = z_group  # [:, np.newaxis]
            zs_encoded = self.enc.transform(z_group).toarray()
            x_points = x_points[:, np.newaxis]
            xzs_encoded = np.concatenate(
                [x_points, zs_encoded], axis=1).tolist()
            tensor_xzs = torch.stack([torch.Tensor(i)
                                      for i in xzs_encoded])
        elif encoder == "binary":
            zs_encoded = self.enc.transform(z_group).to_numpy()
            x_points = x_points[:, np.newaxis]
            xzs_encoded = np.concatenate(
                [x_points, zs_encoded], axis=1).tolist()
            tensor_xzs = torch.stack([torch.Tensor(i)
                                      for i in xzs_encoded])
        elif encoder == "embedding":
            zs_transformed =  z_group.reshape(1,-1)[0]
            zs_encoded = self.enc.predicts(zs_transformed)
            x_points = x_points[:, np.newaxis]
            xzs_encoded = np.concatenate(
                [x_points, zs_encoded], axis=1).tolist()
            tensor_xzs = torch.stack([torch.Tensor(i)
                                      for i in xzs_encoded])
            
        else:
            xzs = [[x_point, z_point]
                   for x_point, z_point in zip(x_points, z_group)]
            tensor_xzs = torch.stack([torch.Tensor(i)
                                      for i in xzs])

        tensor_xzs = tensor_xzs.to(device)
        self.model = self.model.to(device)

        pis, sigmas, mus = self.model(tensor_xzs)
        if not b_plot:
            pis = pis.cpu().detach().numpy()  # [0]
            # sigmas = sigmas.detach().numpy().reshape(len(sigmas), -1)[0]
            mus = mus.cpu().detach().numpy().reshape(len(z_group), -1)  # [0]
            predictions = np.sum(np.multiply(pis, mus), axis=1)

            if self.b_normalize_data:
                predictions = [denormalize(pred, self.meany, self.widthy)
                               for pred in predictions]
            return predictions
        else:
            samples = sample(pis, sigmas, mus).data.numpy().reshape(-1)
            if self.b_normalize_data:
                samples = [denormalize(pred, self.meany, self.widthy)
                           for pred in samples]
            # plt.scatter(z_group, x_points, samples)
            # plt.show()
            # return samples

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if len(self.x_points) > 2000:
                idx = np.random.randint(0, len(self.x_points), 2000)
                if self.b_normalize_data:
                    x_samples = [denormalize(i, self.meanx, self.widthx)
                                 for i in self.x_points[idx]]
                    y_samples = [denormalize(i, self.meany, self.widthy)
                                 for i in self.y_points[idx]]
                ax.scatter(x_samples,
                           self.z_points[idx], y_samples)
            else:
                ax.scatter(self.x_points, self.z_points, self.y_points)

            if self.b_normalize_data:
                x_points = denormalize(x_points, self.meanx, self.widthx)
            if len(samples) > 2000:
                idx = np.random.randint(0, len(x_points), 2000)
                ax.scatter(np.array(x_points)[idx], np.array(
                    z_group)[idx], np.array(samples)[idx])
            else:
                ax.scatter(x_points, z_group, samples)
            ax.set_xlabel('query range attribute')
            ax.set_ylabel('group by attribute')
            ax.set_zlabel('aggregate attribute')
            plt.show()
            return samples

    def score(self, runtime_config) -> float:
        """ evaluate the error for this model. currenltly, 
        it is the sum of all absolute errors, for a random sample of points.

        Raises:
            ValueError: b_store_training_data must be set to True to enable the score() function.

        Returns:
            float: the absolute error
        """
        gs = ["g1", "g2", "g3", "g4", "g5"]
        if not self.b_store_training_data:
            raise ValueError(
                "b_store_training_data must be set to True to enable the score() function.")
        else:
            # groups = self.enc.categories_[0]

            if self.sample_x is None:
                # process group by values
                data = {gs[i]: [row[i] for row in self.z_points]
                        for i in range(len(self.z_points[0]))}
                # append x y values
                data['x'] = denormalize(self.x_points, self.meanx, self.widthx)
                data['y'] = denormalize(self.y_points, self.meany, self.widthy)

                df = pd.DataFrame(data)
                columns = list(df.columns.values)
                columns.remove("y")

                # df = pd.DataFrame(
                #     {'g': self.z_points, 'x': denormalize(self.x_points, self.meanx, self.widthx), 'y': denormalize(self.y_points, self.meany, self.widthy)})
                # mean_y = df.groupby(['g', 'x'])['y'].mean()  # .reset_index()
                mean_y = df.groupby(columns)['y'].mean()  # .reset_index()
                # print(df)
                # raise Exception

                # make the same index here
                df = df.set_index(columns)  # df = df.set_index(['g', 'x'])
                df['mean_y'] = mean_y
                # print(df)
                df = df.reset_index()  # to take the hierarchical index off again

                df = df.sample(
                    n=min(1000, len(self.x_points)), random_state=1, replace=False)
                self.sample_x = df["x"].values
                # for g in columns[:-1]:
                #     self.sample_g = df["g"].values
                self.sample_g = df[columns[:-1]].values
                # raise Exception

                self.sample_average_y = df["mean_y"].values

            predictions = self.predict(
                self.sample_g, self.sample_x, runtime_config)
            errors = [abs(pred-tru)
                      for pred, tru in zip(predictions, self.sample_average_y)]
            errors = sum(sorted(errors)[10:-10])
            return errors


class RegMdn():
    """ This class implements the regression using mixture density network.
    """

    # , n_mdn_layer_node=20, b_one_hot=True
    def __init__(self, config, dim_input,  b_store_training_data=False):
        if b_store_training_data:
            self.xs = None  # query range
            self.ys = None  # aggregate value
            self.zs = None  # group by balue
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
        self.last_xs = None
        self.last_pi = None
        self.last_mu = None
        self.last_sigma = None
        self.enc = None
        self.config = config

    # num_epoch=400, num_gaussians=5

    def fit(self, xs, ys, runtime_config, b_show_plot=False, b_normalize=True):
        """ fit a regression y= R(x)"""
        if len(xs.shape) != 2:
            raise Exception("xs should be 2-d, but got unexpected shape.")
        if self.dim_input == 1:
            return self.fit2d(xs, ys, runtime_config, b_show_reg_plot=b_show_plot,
                              b_normalize=b_normalize, )
        elif self.dim_input == 2:
            return self.fit3d(xs[:, 0], xs[:, 1], ys, runtime_config, b_show_plot=b_show_plot,
                              b_normalize=b_normalize, )
        else:
            print("dimension mismatch")
            sys.exit(0)

    def predict(self, xs, runtime_config, b_show_plot=False):
        """ make predictions"""
        if self.dim_input == 1:
            return self.predict2d(xs, runtime_config, b_show_plot=b_show_plot)
        elif self.dim_input == 2:
            return self.predict3d(xs[:, 0], xs[:, 1], runtime_config, b_show_plot=b_show_plot)
        else:
            print("dimension mismatch")
            sys.exit(0)

    def fit3d(self, xs, zs, ys, runtime_config, b_show_plot=False, b_normalize=True,  n_workers=0):
        """ fit a regression y = R(x,z)

        Args:
            xs ([float]): query range attribute
            zs ([float]): group by attribute
            ys ([float]): aggregate attribute
            b_show_plot (bool, optional): whether to show the plot. Defaults to True.
        """
        b_one_hot = True
        device = runtime_config["device"]
        n_mdn_layer_node = self.config.config["n_mdn_layer_node"]
        num_gaussians = self.config.config["n_gaussions"]
        num_epoch = self.config.config["n_epoch"]
        if b_one_hot:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            zs_onehot = zs[:, np.newaxis]
            zs_onehot = self.enc.fit_transform(zs_onehot).toarray()

        if b_normalize:
            self.meanx = (np.max(xs) + np.min(xs)) / 2
            self.widthx = np.max(xs) - np.min(xs)
            self.meany = (np.max(ys) + np.min(ys)) / 2
            self.widthy = np.max(ys) - np.min(ys)
            # self.meanz = np.mean(zs)
            # self.widthz = np.max(zs)-np.min(zs)

            # s= [(i-meanx)/1 for i in x]
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])
            ys = np.array([self.normalize(i, self.meany, self.widthy)
                           for i in ys])
            # zs = np.array([self.normalize(i, self.meanz, self.widthz)
            #                for i in zs])
            self.is_normalized = True

        if self.b_store_training_data:
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

        if b_one_hot:
            xs_onehot = xs[:, np.newaxis]
            xzs_onehot = np.concatenate(
                [xs_onehot, zs_onehot], axis=1).tolist()
            tensor_xzs = torch.stack([torch.Tensor(i)
                                      for i in xzs_onehot])  # transform to torch tensors
        else:
            xzs = [[xs[i], zs[i]] for i in range(len(xs))]
            tensor_xzs = torch.stack([torch.Tensor(i)
                                      for i in xzs])  # transform to torch tensors
        ys = ys[:, np.newaxis]
        tensor_ys = torch.stack([torch.Tensor(i) for i in ys])

        # move variables to cuda
        tensor_xzs = tensor_xzs.to(device)
        tensor_ys = tensor_ys.to(device)

        my_dataset = torch.utils.data.TensorDataset(
            tensor_xzs, tensor_ys)  # create your datset
        # , num_workers=8) # create your dataloader
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset, batch_size=self.config.config["batch_size"], shuffle=False, num_workers=n_workers)

        input_dim = len(self.enc.categories_[0]) + 1
        # initialize the model
        self.model = nn.Sequential(
            nn.Linear(input_dim, n_mdn_layer_node),  # self.dim_input
            nn.Tanh(),
            nn.Dropout(0.01),
            MDN(n_mdn_layer_node, 1, num_gaussians, device)
        )

        self.model = self.model.to(device)

        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(num_epoch):
            if epoch % 100 == 0:
                print("< Epoch {}".format(epoch))
            # train the model
            for minibatch, labels in my_dataloader:
                minibatch.to(device)
                labels.to(device)
                self.model.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = mdn_loss(pi, sigma, mu, labels, device)
                loss.backward()
                optimizer.step()
        return self

    def fit3d_grid_search(self,  xs: list,  zs: list, ys: list, runtime_config, b_normalize=True):
        """ fit the regression, using grid search to find the optimal parameters.

        Args:
            xs (list): x points.
            zs (list): group by attributes
            ys (list): y values.
            b_normalize (bool, optional): whether the values should be normalized
                        for training. Defaults to True.

        Returns:
            RegMdn: the model.
        """

        param_grid = {'epoch': [5], 'lr': [0.001, 0.0001], 'node': [
            5, 10, 20], 'hidden': [1, 2], 'gaussian': [2, 4]}
        # param_grid = {'epoch': [2], 'lr': [0.001], 'node': [4,  12], 'hidden': [1, 2], 'gaussian': [10]}
        errors = []
        combinations = it.product(*(param_grid[Name] for Name in param_grid))
        combinations = list(combinations)
        combs = []
        for combination in combinations:
            idx = 0
            comb = {}
            for key in param_grid:
                comb[key] = combination[idx]
                idx += 1
            combs.append(comb)

        self.b_store_training_data = True
        # for para in combs:
        #     print("Grid search for parameter set :", para)
        #     instance = self.fit(zs, xs, b_normalize=b_normalize, num_gaussians=para['gaussian'], num_epoch=para['epoch'],
        #                         n_mdn_layer_node=para['node'], lr=para['lr'], hidden=para['hidden'], b_grid_search=False)
        #     errors.append(instance.score())

        # index = errors.index(min(errors))
        # para = combs[index]
        # print("Finding the best configuration for the network", para)

        # self.b_store_training_data = False
        # instance = self.fit(zs, xs, b_normalize=True, num_gaussians=para['gaussian'], num_epoch=20,
        #                     n_mdn_layer_node=para['node'], lr=para['lr'], hidden=para['hidden'], b_grid_search=False)
        # return instance

    def fit2d(self, xs, ys, runtime_config, b_show_reg_plot=False, b_normalize=True,
              b_show_density_plot=False, n_workers=0):
        """ fit a regression y = R(x)

        Args:
            xs([float]): query range attribute
            ys([float]): aggregate attribute
            b_show_plot(bool, optional): whether to show the plot. Defaults to True.
        """
        n_mdn_layer_node = self.config.config["n_mdn_layer_node"]
        num_epoch = self.config.config["n_epoch"]

        if b_normalize:
            self.meanx = (np.max(xs) + np.min(xs)) / 2
            self.widthx = np.max(xs) - np.min(xs)
            self.meany = (np.max(ys) + np.min(ys)) / 2
            self.widthy = np.max(ys) - np.min(ys)

            # s= [(i-meanx)/1 for i in x]
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])
            ys = np.array([self.normalize(i, self.meany, self.widthy)
                           for i in ys])

            self.is_normalized = True

        if self.b_store_training_data:
            self.xs = xs
            self.ys = ys

        if b_show_reg_plot:
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
            my_dataset, batch_size=self.config.config["batch_size"], shuffle=False, num_workers=n_workers)

        # initialize the model
        self.model = nn.Sequential(
            nn.Linear(self.dim_input, n_mdn_layer_node),
            nn.Tanh(),
            nn.Dropout(0.01),
            MDN(n_mdn_layer_node, 1,
                self.config.config["n_gaussians_reg"], runtime_config['device'])
        )

        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(num_epoch):
            if epoch % 5 == 0:
                print("< Epoch {}".format(epoch))
            # train the model
            for minibatch, labels in my_dataloader:
                self.model.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = mdn_loss(pi, sigma, mu, labels,
                                runtime_config['device'])
                loss.backward()
                optimizer.step()

        return self

    def predict3d(self, xs, zs, runtime_config, b_show_plot=True, b_generate_samples=False):
        if self.is_normalized:
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])
            # zs = np.array([self.normalize(i, self.meanz, self.widthz)
            #                for i in zs])

        zs_onehot = zs[:, np.newaxis]
        zs_onehot = self.enc.transform(zs_onehot).toarray()

        xs_onehot = xs[:, np.newaxis]
        xzs_onehot = np.concatenate([xs_onehot, zs_onehot], axis=1).tolist()

        tensor_xzs = torch.stack([torch.Tensor(i)
                                  for i in xzs_onehot])

        pi, sigma, mu = self.model(tensor_xzs)

        if b_generate_samples:
            samples = sample(pi, sigma, mu).data.numpy().reshape(-1)
            for i in range(num_points - 1):
                samples = np.vstack(
                    (samples, sample(pi, sigma, mu).data.numpy().reshape(-1)))
            samples = np.mean(samples, axis=0)
        else:
            mu = mu.detach().numpy().reshape(len(xs), -1)
            pi = pi.detach().numpy()  # .reshape(-1,2)
            samples = np.sum(np.multiply(pi, mu), axis=1)

        if self.is_normalized:
            # de-normalize the data
            samples = [self.denormalize(
                i, self.meany, self.widthy) for i in samples]
            xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                           for i in xs])
            # zs = np.array([self.denormalize(i, self.meanz, self.widthz)
            #                for i in zs])

        if b_show_plot:

            if not self.is_training_data_denormalized and self.b_store_training_data:
                self.xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                                    for i in self.xs])
                self.ys = np.array([self.denormalize(i, self.meany, self.widthy)
                                    for i in self.ys])
                # self.zs = np.array([self.denormalize(i, self.meanz, self.widthz)
                #                     for i in self.zs])
                self.is_training_data_denormalized = True
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if len(self.xs) > 2000:
                idx = np.random.randint(0, len(self.xs), 2000)
                ax.scatter(self.xs[idx], self.zs[idx], self.ys[idx])
            else:
                ax.scatter(self.xs, self.zs, self.ys)

            if len(xs) > 2000:
                idx = np.random.randint(0, len(xs), 2000)
                ax.scatter(np.array(xs)[idx], np.array(
                    zs)[idx], np.array(samples)[idx])
            else:
                ax.scatter(xs, zs, samples)
            ax.set_xlabel('query range attribute')
            ax.set_ylabel('group by attribute')
            ax.set_zlabel('aggregate attribute')
            plt.show()
        return samples

    def predict2d(self, xs, runtime_config, b_show_plot=False, b_generate_samples=False):
        if self.is_normalized:
            xs = np.array([self.normalize(i, self.meanx, self.widthx)
                           for i in xs])

        tensor_xs = torch.stack([torch.Tensor(i)
                                 for i in xs])

        pi, sigma, mu = self.model(tensor_xs)
        if b_generate_samples:
            samples = sample(pi, sigma, mu).data.numpy().reshape(-1)
            for i in range(num_points - 1):
                samples = np.vstack(
                    (samples, sample(pi, sigma, mu).data.numpy().reshape(-1)))
            samples = np.mean(samples, axis=0)
        else:
            mu = mu.detach().numpy().reshape(len(xs), -1)
            pi = pi.detach().numpy()  # .reshape(-1,2)

            samples = np.sum(np.multiply(pi, mu), axis=1)

        if self.is_normalized:
            # de-normalize the data
            samples = [self.denormalize(
                i, self.meany, self.widthy) for i in samples]
            xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                           for i in xs])

        if b_show_plot:
            if not self.is_training_data_denormalized:
                self.xs = np.array([self.denormalize(i, self.meanx, self.widthx)
                                    for i in self.xs])
                self.ys = np.array([self.denormalize(i, self.meany, self.widthy)
                                    for i in self.ys])
                self.is_training_data_denormalized = True
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(self.xs, self.ys)
            ax1.scatter(xs, samples)
            ax1.set_xlabel('query range attribute')
            ax1.set_ylabel('aggregate attribute')
            plt.show()

        samples = list(samples)
        return samples

    def normalize(self, x, mean, width):
        """normalize x

        Args:
            x([type]): [description]
            mean([type]): [description]
            width([type]): [description]

        Returns:
            [type]: [description]
        """
        return (x - mean) / width * 2

    def denormalize(self, x, mean, width):
        return 0.5 * width * x + mean


class KdeMdn:
    """This is the implementation of density estimation using MDN"""

    def __init__(self, config, b_store_training_data=False, b_normalize_data=True):
        if b_store_training_data:
            self.xs = None  # query range
            self.zs = None  # group by balue
        self.b_store_training_data = b_store_training_data
        self.meanx = None
        self.widthx = None
        self.config = config
        self.b_normalize_data = b_normalize_data

    def fit(self, zs: list, xs: list, runtime_config,  lr=0.001,  n_workers=0):
        """ fit the density for the data, to support group by queries.

        Args:
            zs (list): the group values
            xs (list): the independent variables.
            b_normalize (bool, optional): normalize the data before training the MDN network. Defaults to True.
            num_gaussians (int, optional): the number of gaussions in the MDN network. Defaults to 20.
            num_epoch (int, optional): number of epoches for training. Defaults to 20.
            n_mdn_layer_node (int, optional): the node number in the hidden layer. Defaults to 20.
            lr (float, optional): learning rate. Defaults to 0.001.
            hidden (int, optional): the number of hidden layers before the MDN layer. Defaults to 1.
            b_grid_search (bool, optional): use grid search to tune hyper parameters. Defaults to True.

        Raises:
            Exception: [description]
            ValueError: "hidden layers must be 1 or 2."

        Returns:
            KdeMdn: the fitted model.
        """
        # print("gb", zs)
        num_gaussians = self.config.config["n_gaussians_density"]
        num_epoch = self.config.config["n_epoch"]
        n_mdn_layer_node = self.config.config["n_mdn_layer_node_density"]
        hidden = self.config.config["n_hidden_layer"]
        b_grid_search = self.config.config["b_grid_search"]
        encoder = self.config.config["encoder"]
        device = runtime_config["device"]

        if not self.config.config["b_grid_search"]:
            if self.b_store_training_data:
                self.zs = zs
                self.xs = xs
            else:
                self.zs = None
                self.xs = None

            if self.b_normalize_data:
                self.meanx = (np.max(xs) + np.min(xs)) / 2
                self.widthx = np.max(xs) - np.min(xs)

                xs = np.array([self.normalize(i, self.meanx, self.widthx)
                               for i in xs])
                # self.b_normalize_data = True
                if encoder == "no":
                    convert2float = True
                    if convert2float:
                        try:
                            zs_float = []
                            for item in zs:
                                if item[0] == "":
                                    zs_float.append([0.0])
                                else:
                                    zs_float.append([(float)(item[0])])
                            zs = np.array(zs_float)
                        except:
                            raise Exception

                    self.meanz = (np.max(zs) + np.min(zs)) / 2
                    self.widthz = np.max(zs) - np.min(zs)

                    zs = np.array([self.normalize(i, self.meanz, self.widthz)
                                   for i in zs])

            if encoder == "onehot":
                self.enc = OneHotEncoder(handle_unknown='ignore')
                zs_encoded = self.enc.fit_transform(zs).toarray()
                # len(self.enc.categories_[0])
                input_dim = sum([len(i) for i in self.enc.categories_]) + 0
                tensor_zs = torch.stack([torch.Tensor(i)
                                         for i in zs_encoded])  # transform to torch tensors
            elif encoder == "binary":
                columns = list(range(len(zs[0])))
                self.enc = ce.BinaryEncoder(cols=columns)
                zs_encoded = self.enc.fit_transform(zs).to_numpy()
                input_dim = len(self.enc.base_n_encoder.feature_names) + 0
                tensor_zs = torch.stack([torch.Tensor(i)
                                         for i in zs_encoded])
            elif encoder == "embedding":
                sentences = columns2sentences(zs, xs, ys_data=None)
                self.enc = WordEmbedding()
                self.enc.fit(sentences, gbs=["gb"],dim=self.config.config["n_embedding_dim"])
                gbs_data = zs.reshape(1,-1)[0]
                zs_encoded = self.enc.predicts(gbs_data)
                tensor_zs = torch.stack([torch.Tensor(i)
                                         for i in zs_encoded])
                input_dim =  self.enc.dim
                # raise TypeError("embedding is not supported yet.")
            else:
                input_dim = 1
                tensor_zs = torch.stack([torch.Tensor(i)
                                         for i in zs])
            xs = xs[:, np.newaxis]
            tensor_xs = torch.stack([torch.Tensor(i) for i in xs])

            # move variables to device
            tensor_xs = tensor_xs.to(device)
            tensor_zs = tensor_zs.to(device)

            # print("zs_encoded", zs_encoded)
            # # print("gb_encoded", tensor_zs)
            # raise Exception

            my_dataset = torch.utils.data.TensorDataset(
                tensor_zs, tensor_xs)  # create your dataloader
            my_dataloader = torch.utils.data.DataLoader(
                my_dataset, batch_size=self.config.config["batch_size"], shuffle=False, num_workers=n_workers)

            # initialize the model
            if hidden == 1:
                self.model = nn.Sequential(
                    nn.Linear(input_dim, n_mdn_layer_node),  # self.dim_input
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    MDN(n_mdn_layer_node, 1, num_gaussians, device)
                )
            elif hidden == 2:
                self.model = nn.Sequential(
                    nn.Linear(input_dim, n_mdn_layer_node),  # self.dim_input
                    nn.Tanh(),
                    nn.Linear(n_mdn_layer_node, n_mdn_layer_node),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    MDN(n_mdn_layer_node, 1, num_gaussians, device)
                )
            else:
                raise ValueError("hidden layers must be 1 or 2.")

            self.model = self.model.to(device)

            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            decayRate = 0.96
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=decayRate)
            for epoch in range(num_epoch):
                if runtime_config["v"]:
                    if epoch % 1 == 0:
                        print("< Epoch {}".format(epoch))
                # train the model
                for minibatch, labels in my_dataloader:
                    self.model.zero_grad()
                    # move variables to device
                    minibatch.to(device)
                    labels.to(device)
                    pi, sigma, mu = self.model(minibatch)
                    loss = mdn_loss(pi, sigma, mu, labels, device)
                    loss.backward()
                    optimizer.step()
                my_lr_scheduler.step()
            # turn the model to eval mode.
            self.model.eval()
            print("finish mdn training...")
            return self
        else:  # grid search
            # , b_normalize=b_normalize
            return self.fit_grid_search(zs, xs, runtime_config)

    def fit_grid_search(self, zs: list, xs: list, runtime_config):  # , b_normalize=True
        """ use grid search to tune the hyper parameters.

        Args:
            zs (list): the group by values.
            xs (list): the independent variables.
            b_normalize (bool, optional): normalize the independent variables before training the MDN model. Defaults to True.

        Returns:
            KdeMdn: the fitted model.
        """

        param_grid = {'epoch': [5], 'lr': [0.001], 'node': [
            5, 10, 20], 'hidden': [1, 2], 'gaussian': [5, 10, 20]}
        # param_grid = {'epoch': [2], 'lr': [0.001],
        #               'node': [4], 'hidden': [1], 'gaussian': [10]}
        errors = []
        combinations = it.product(*(param_grid[Name] for Name in param_grid))
        combinations = list(combinations)
        combs = []
        for combination in combinations:
            idx = 0
            comb = {}
            for key in param_grid:
                comb[key] = combination[idx]
                idx += 1
            combs.append(comb)

        self.b_store_training_data = True
        for para in combs:
            print("Grid search for parameter set :", para)
            config = self.config.copy()
            config.config["n_gaussians_density"] = para['gaussian']
            config.config["n_epoch"] = para['epoch']
            config.config["n_mdn_layer_node_density"] = para['node']
            config.config["n_hidden_layer"] = para['hidden']
            config.config["b_grid_search"] = False
            instance = KdeMdn(config, b_store_training_data=True, b_normalize_data=self.b_normalize_data).fit(
                zs, xs, runtime_config,  lr=para['lr'])  # b_normalize=b_normalize,
            errors.append(instance.score(runtime_config))

        index = errors.index(min(errors))
        para = combs[index]
        print("Finding the best configuration for the network", para)

        self.b_store_training_data = False
        # release space
        self.x_points = None
        self.y_points = None
        self.z_points = None
        self.sample_x = None
        self.sample_g = None
        self.sample_average_y = None
        config = self.config.copy()
        config.config["n_gaussians_density"] = para['gaussian']
        config.config["num_epoch"] = para['epoch']
        config.config["n_mdn_layer_node_density"] = para['node']
        config.config["n_hidden_layer"] = para['hidden']
        config.config["b_grid_search"] = False
        instance = KdeMdn(config, b_store_training_data=False, b_normalize_data=self.b_normalize_data).fit(
            zs, xs, runtime_config,  lr=para['lr'])  # b_normalize=b_normalize,
        print("-"*80)
        return instance

    def predict(self, zs: list, xs: list, runtime_config, b_plot=False,):
        """ provide density estimations for given points. zs and xs must of the same size.

        Args:
            zs (list): the group by values.
            xs (list): the independent variables.
            b_plot (bool, optional): plot the predictions along with the training data. Defaults to False.
            n_division (int, optional): the number of divisions for predictions. Defaults to 100.

        Raises:
            Exception: [description]

        Returns:
            list: the predictions.
        """
        encoder = self.config.config["encoder"]
        device = runtime_config["device"]
        # torch.set_num_threads(4)
        # convert group zs from string to int
        if encoder == "no":
            convert2float = True
            if convert2float:
                try:
                    zs_float = []
                    for item in zs:
                        if item[0] == "":
                            zs_float.append([0.0])
                        else:
                            zs_float.append([(float)(item[0])])
                    zs = zs_float
                except:
                    raise Exception

        if self.b_normalize_data:
            # print(xs, self.meanx, self.widthx)
            xs = self.normalize(xs, self.meanx, self.widthx)

        zs = np.array(zs)  # [:, np.newaxis]

        if encoder == "onehot":
            zs_encoded = self.enc.transform(zs).toarray()
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs_encoded])
        elif encoder == "binary":
            zs_encoded = self.enc.transform(zs).to_numpy()
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs_encoded])
        elif encoder == "embedding":
            zs_transformed =  zs.reshape(1,-1)[0]
            zs_encoded = self.enc.predicts(zs_transformed)
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs_encoded])
        else:
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs])
        tensor_zs = tensor_zs.to(device)
        self.model = self.model.to(device)

        pis, sigmas, mus = self.model(tensor_zs)

        pis = pis.cpu()
        sigmas = sigmas.cpu()
        mus = mus.cpu()

        # print("mus,", mus)
        # print("pis,", pis)
        # print("sigmas,", sigmas)

        mus = mus.detach().numpy().reshape(len(zs), -1)  # [0]
        pis = pis.detach().numpy()  # [0]  # .reshape(-1,2)
        sigmas = sigmas.detach().numpy().reshape(
            len(sigmas), -1)  # [0]
        # print("mus,", mus)
        # print("pis,", pis)
        # print("sigmas,", sigmas)
        # with open('/home/u1796377/Desktop/tmp/mus.pkl', 'wb') as f:
        #     dill.dump(mus, f)
        # with open('/home/u1796377/Desktop/tmp/pis.pkl', 'wb') as f:
        #     dill.dump(pis, f)
        # with open('/home/u1796377/Desktop/tmp/sigmas.pkl', 'wb') as f:
        #     dill.dump(sigmas, f)
        # with open('/home/u1796377/Desktop/tmp/xs.pkl', 'wb') as f:
        #     dill.dump(xs, f)

        if not b_plot:
            # result = [gm(pi, mu,
            #              sigma, xs, b_plot=False) for pi, mu, sigma in zip(pis, mus, sigmas)]
            # print("result", result, type(result))
            result = np.array([np.multiply(stats.norm(mus, sigmas).pdf(x),
                                           pis).sum(axis=1).tolist() for x in xs]).transpose()
            # print("result", result, type(result))
            # raise Exception
            # scale up the probability, due to normalization of the x axis.
            result = result / self.widthx * 2
            return result
        else:
            return gm(pis[0], mus[0], sigmas[0], xs, b_plot=b_plot, n_division=runtime_config["n_division"])
    
    def var(self, zs, runtime_config):
        encoder = self.config.config["encoder"]
        device = runtime_config["device"]
        if encoder == "onehot":
            zs_encoded = self.enc.transform(zs).toarray()
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs_encoded])
        elif encoder == "binary":
            zs_encoded = self.enc.transform(zs).to_numpy()
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs_encoded])
        elif encoder == "embedding":
            # zs_transformed =  zs.reshape(1,-1)[0]
            zs_transformed =  np.array(zs).reshape(1,-1)[0]
            zs_encoded = self.enc.predicts(zs_transformed)
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs_encoded])
        else:
            tensor_zs = torch.stack([torch.Tensor(i)
                                     for i in zs])
        tensor_zs = tensor_zs.to(device)
        self.model = self.model.to(device)

        pis, sigmas, mus = self.model(tensor_zs)

        pis = pis.cpu()
        sigmas = sigmas.cpu()
        mus = mus.cpu()
        mus = mus.detach().numpy().reshape(len(zs), -1)  # [0]
        pis = pis.detach().numpy()  # [0]  # .reshape(-1,2)
        sigmas = sigmas.detach().numpy().reshape(
            len(sigmas), -1)  # [0]
        print("pis",pis)
        print("sigmas",sigmas)
        print("mus",mus)
        print("mean, width", self.meanx, self.widthx)
        sigmas= sigmas * 0.5 * self.widthx# + self.meanx
        mus = mus * 0.5 * self.widthx + self.meanx
        print("mus",mus)
        print("mus[0]",mus[0])
        print("pis[0]",pis[0])
        print("sigma[0]", sigmas[0])

        mu_avg_2 = np.power(np.sum(np.multiply(mus,pis,dtype="float64"), axis=1),2)
        print("mu_avg_2",mu_avg_2[0])
        mu_2 = np.power(mus,2,dtype="float64")
        sigmas_2 = np.power(sigmas,2,dtype="float64")
        adds = np.add(mu_2, sigmas_2,dtype="float64")
        print("adds", adds[0])
        sums =np.multiply(adds, pis,dtype="float64").sum(axis=1,dtype="float64")
        print("sums",sums[0])
        print("types",type(sums[0]))
        result = np.subtract(sums, mu_avg_2,dtype="float64").tolist()
        result = np.sqrt(result)
        print("result",result)
        print(len(result))
        result = dict(zip(zs, result))
        return result


    def normalize(self, x: list, mean: float, width: float):
        """normalize the data

        Args:
            x (list): the data points to be normalized.
            mean (float): the mean value of x.
            width (float): the range of x.

        Returns:
            list: the normalized data.
        """
        return (x - mean) / width * 2

    def denormalize(self, x, mean, width):
        """de-normalize the data

        Args:
            x (list): the data points to be de-normalized.
            mean (float): the mean value of x.
            width (float): the range of x.

        Returns:
            list: the de-normalized data.
        """
        return 0.5 * width * x + mean

    def plot_density_3d(self, n_division=20):
        """plot the 3d density curves.

        Args:
            n_division (int, optional): the number of divisions. Defaults to 20.

        Raises:
            ValueError:  "b_store_training_data must be set to True to enable the plotting function."
        """
        if not self.b_store_training_data:
            raise ValueError(
                "b_store_training_data must be set to True to enable the plotting function.")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(211, projection='3d')
            zs_plot = self.zs.reshape(1, -1)[0]
            hist, xedges, yedges = np.histogram2d(
                self.xs, zs_plot, bins=n_division)
            # plt.scatter(zs, xs)

            # Construct arrays for the anchor positions of the 16 bars.
            xpos, ypos = np.meshgrid(
                xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = 0

            # Construct arrays with the dimensions for the 16 bars.
            dx = dy = 0.5 * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
            ax.set_xlabel("range predicate")
            ax.set_ylabel("group by attribute")
            ax.set_zlabel("frequency")

            ax1 = fig.add_subplot(212, projection='3d')
            zs_set = list(set(zs_plot))
            for z in zs_set:
                xxs, yys = self.predict(
                    [[z]], 200, b_plot=True, n_division=n_division)
                xxs = [self.denormalize(xi, self.meanx, self.widthx)
                       for xi in xxs]
                yys = [yi / self.widthx * 2 for yi in yys]
                zzs = [z] * len(xxs)
                ax1.plot(xxs, zzs, yys)
            ax1.set_xlabel("range predicate")
            ax1.set_ylabel("group by attribute")
            ax1.set_zlabel("frequency")
            plt.show()

    def plot_density_per_group(self, n_division=100):
        """ plot the density for a specific group.

        Args:
            n_division (int, optional): the number of division in each group. Defaults to 100.

        Raises:
            ValueError: "b_store_training_data must be set to True to enable the plotting function."
        """
        if not self.b_store_training_data:
            raise ValueError(
                "b_store_training_data must be set to True to enable the plotting function.")
        else:

            zs_plot = self.zs.reshape(1, -1)[0]
            df = pd.DataFrame({"z": zs_plot, "x": self.xs})

            df = df.dropna(subset=["z", "x"])
            gp = df.groupby(["z"])

            zs_set = list(gp.groups.keys())
            # zs_set = list(set(zs_plot)).sort()
            z_min = zs_set[0]  # the minimial value of the paramater a
            z_max = zs_set[-1]  # the maximal value of the paramater a
            # the value of the parameter a to be used initially, when the graph is created
            z_init = zs_set[5]

            self.fig = plt.figure(figsize=(8, 8))

            # first we create the general layount of the figure
            # with two axes objects: one for the plot of the function
            # and the other for the slider
            plot_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
            slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

            # in plot_ax we plot the function with the initial value of the parameter a
            one_group = gp.get_group(z_init)

            x_plot = one_group['x']
            z_plot = one_group["z"]

            plt.axes(plot_ax)  # select sin_ax
            plt.title('Density Estimation')
            plt.xlabel("Query range attribute")
            plt.ylabel("Frequency")
            main_plot, _, _ = plt.hist(x_plot, bins=100)
            # plt.xlim(0, 2 * math.pi)
            # plt.ylim(-1.1, 1.1)

            # here we create the slider
            self.a_slider = Slider(slider_ax,  # the axes object containing the slider
                                   'groupz',  # the name of the slider parameter
                                   z_min,  # minimal value of the parameter
                                   z_max,  # maximal value of the parameter
                                   valinit=z_init  # initial value of the parameter
                                   )

            # plot the density estimation on another y axis

            ax_frequency = plt.gca()
            ax_density = ax_frequency.twinx()
            ax_density.set_ylabel("Density", color="tab:red")
            xxs, yys = self.predict([[z_init]], 200, b_plot=True)
            xxs = [self.denormalize(xi, self.meanx, self.widthx) for xi in xxs]
            yys = [yi / self.widthx * 2 for yi in yys]
            #
            # plt.plot(xxs, yys)
            ax_density.plot(xxs, yys, "r")

            # Next we define a function that will be executed each time the value
            # indicated by the slider changes. The variable of this function will
            # be assigned the value of the slider.
            def update(groupz):
                # sin_plot.set_ydata(np.sin(a * x))  # set new y-coordinates of the plotted points
                group_approx = min(zs_set, key=lambda x: abs(x - groupz))
                print("result for group " + str(group_approx))
                one_group = gp.get_group(group_approx)
                x_plot = one_group['x']
                # main_plot
                plt.axes(plot_ax)
                plt.cla()
                plt.hist(x_plot, bins=100)
                plt.title('Density Estimation')
                plt.xlabel("Query range attribute")
                plt.ylabel("Frequency")

                ax_frequency = plt.gca()
                # if ax_density is None:
                #     ax_density = ax_frequency.twinx()
                ax_density.cla()
                ax_density.set_ylabel("Density", color="tab:red")
                xxs, yys = self.predict([[group_approx]], 200, b_plot=True)
                xxs = [self.denormalize(xi, self.meanx, self.widthx)
                       for xi in xxs]
                yys = [yi / self.widthx * 2 for yi in yys]
                #
                # plt.plot(xxs, yys)
                ax_density.plot(xxs, yys, "r")

                self.fig.canvas.draw_idle()  # redraw the plot

            # the final step is to specify that the slider needs to
            # execute the above function when its value changes
            self.a_slider.on_changed(update)

            plt.show()

    def serialize(self, file: str):
        """serialize the model to file, using dill.

        Args:
            file (str): the path to store the model.
        """
        with open(file, 'wb') as f:
            dill.dump(self, f)

    def bin_wise_error(self, g, n_division=20, b_show_plot=True):
        if not self.b_store_training_data:
            raise ValueError("b_store_training_data must be set to True to enable the plotting function for bin-wise "
                             "comparison.")
        else:
            from scipy import integrate

            zs_plot = self.zs.reshape(1, -1)[0]
            df = pd.DataFrame({"z": zs_plot, "x": self.xs})
            df = df.dropna(subset=["z", "x"])
            gp = df.groupby(["z"])

            zs_set = list(gp.groups.keys())
            z_min = zs_set[0]  # the minimial value of the paramater a
            z_max = zs_set[-1]  # the maximal value of the paramater a
            # zs_set[5]  # the value of the parameter a to be used initially, when the graph is created
            z_init = g

            if b_show_plot:
                self.fig = plt.figure(figsize=(8, 8))

            # first we create the general layount of the figure
            # with two axes objects: one for the plot of the function
            # and the other for the slider
            if b_show_plot:
                plot_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
                slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

            # in plot_ax we plot the function with the initial value of the parameter a
            one_group = gp.get_group(z_init)

            x_plot = one_group['x']
            z_plot = one_group["z"]

            if b_show_plot:
                plt.axes(plot_ax)  # select sin_ax
                plt.title('Density Estimation')
                plt.xlabel("Query range attribute")
                plt.ylabel("Frequency")
            main_plot, bins, patches = plt.hist(x_plot, bins=n_division)

            def predict_func(x):
                return self.predict([z_init], x)

            frequencies = []
            approxs = []
            total = sum(main_plot)
            for patch in patches:
                left, right, frequency = patch._x0, patch._x1, patch._y1/total
                approx = integrate.quad(predict_func, left, right)[0]
                frequencies.append(frequency)
                approxs.append(approx)

            errors = [abs(f-p) for f, p in zip(frequencies, approxs)]

            if b_show_plot:
                plt.clf()
                plt.hist(errors, bins=20)
                plt.show()
            return sum(errors)

    def score(self, runtime_config):
        """evaluate the model, based on training data.

        Raises:
            ValueError: "b_store_training_data must be set to True to enable the score() function."

        Returns:
            float: the total error from a random sample of points.
        """
        gs = ["g1", "g2", "g3", "g4", "g5"]
        if not self.b_store_training_data:
            raise ValueError(
                "b_store_training_data must be set to True to enable the score() function.")
        else:
            zs_plot = self.zs  # .reshape#(1, -1)[0]
            # print(zs_plot)
            data = {gs[i]: [row[i] for row in self.zs]
                    for i in range(len(self.zs[0]))}
            data["x"] = self.xs
            df = pd.DataFrame(data)
            columns = list(df.columns.values)
            df = df.dropna(subset=columns)
            gp = df.groupby(columns[:-1])  # gp = df.groupby(["z"])

            zs_set = list(gp.groups.keys())
            # print(zs_set)
            # random choose
            random.seed(0)
            zs_set = random.sample(zs_set, min(len(zs_set), 20))
            # print("zs_set"+__file__, zs_set)
            errors = []
            # for g in zs_set:
            #     errors.append(self.bin_wise_error(
            #         g, n_division=10, b_show_plot=False))
            # print("finish score")
            # return sum(errors)
            freq_all = []
            for g in zs_set:
                main_plot, bins, patches = plt.hist(
                    gp.get_group(g)["x"], bins=runtime_config["n_division"])
                total = sum(main_plot)
                freq_g = []
                for patch in patches:
                    left, right, frequency = patch._x0, patch._x1, patch._y1/total
                    freq_g.append(frequency)
                freq_all.append(freq_g)
            freq_all = np.array(freq_all)
            freq_all = freq_all.transpose()
            # print(freq_all)
            # print(freq_all.shape)
            # print(zs_set)
            pred_all = []
            for patch in patches:
                left, right = patch._x0, patch._x1
                pre_density, _, step = prepare_reg_density_data(
                    self, left, right, zs_set, None, runtime_config)

                preds = approx_count(pre_density, step)
                pred_all.append(list(preds))
            pred_all = np.array(pred_all)
            # print(pred_all)
            # print(pred_all.shape)

            errors = np.absolute(np.subtract(freq_all, pred_all))
            # print(sum(sum(errors)))
            return sum(sum(errors))

            # results = dict(zip(groups, preds))

            #     errors.append(self.bin_wise_error(
            #         g, n_division=10, b_show_plot=False))
            # return sum(errors)


class KdeMdnOneModel(GenericMdn):
    def __init__(self, config, b_store_training_data=False, b_normalize_data=True):
        if b_store_training_data:
            self.xs = None  # query range
            self.zs = None  # group by balue
        self.b_store_training_data = b_store_training_data
        self.meanx = None
        self.widthx = None
        self.config = config
        self.b_normalize_data = b_normalize_data

    def fit(self):
        pass

    def fit_grid_search(self):
        pass

    def score(self):
        pass


def TestGenericMdn():
    generic_mdn = KdeMdnOneModel()
    generic_mdn.meanx = 5
    generic_mdn.widthx = 10
    xs = np.array([1, 2, 3, 4, 5])
    print(generic_mdn.normalize(xs))
    print(generic_mdn.denormalize(xs))


def test1():
    x = np.random.uniform(low=1, high=10, size=(1000,))
    # z = np.random.uniform(low=1, high=10, size=(1000,))
    z = np.random.randint(0, 7, size=(1000,))
    noise = np.random.normal(1, 5, 1000)
    y = x ** 2 - z ** 2 + noise
    print(min(x), max(x))
    print(min(y), max(y))

    xz = np.concatenate((x[:, np.newaxis], z[:, np.newaxis]), axis=1)

    regMdn = RegMdn(dim_input=2)
    # regMdn.fit(xz, y, num_epoch=200, b_show_plot=False)
    regMdn.fit(xz, y, num_epoch=400, b_show_plot=False)

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
    pres_train = df_train.PRES.values[:, np.newaxis]
    pm25_train = df_train.pm25.values
    pres_test = df_test.PRES.values
    pm25_test = df_test.pm25.values

    regMdn = RegMdn(dim_input=1)
    regMdn.fit(pres_train, pm25_train, num_epoch=100, b_show_plot=False)
    print(regMdn.predict([[1000], [1005], [1010], [1015], [
          1020], [1025], [1030], [1035]], b_show_plot=True))
    print(regMdn.predict([[1000.5], [1005.5], [1010.5], [1015.5], [1020.5], [1025.5], [1030.5], [1035.5]],
                         b_show_plot=True))
    xxs = np.linspace(np.min(pres_train), np.max(pres_train), 100)
    # print(regMdn.predict(xxs,b_show_plot=True))


def test_pm25_3d():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['pm25', 'PRES', 'TEMP'])
    df_train = df  # .head(1000)
    df_test = df  # .tail(1000)
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
    regMdn = RegMdn(dim_input=2, b_store_training_data=True)
    regMdn.fit(xzs_train, pm25_train, num_epoch=100,
               b_show_plot=False, num_gaussians=10)
    print(regMdn.predict(xzs_test, b_show_plot=True))
    # regMdn.predict(xzs_train, b_show_plot=True)


def test_pm25_2d_density():
    import pandas as pd
    from dbestclient.tools.running_parameters import DbestConfig, RUNTIME_CONF as runtime_config
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    # file = "/Users/scott/projects/pm25.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['PRES', 'pm25'])
    df_train = df  # .head(2000)
    df_test = df  # .tail(1000)
    pres_train = df_train.PRES.values[:, np.newaxis]
    pm25_train = df_train.pm25.values  # df_train.pm25.values
    pres_test = df_test.PRES.values
    temp_test = df_test.PRES.values  # df_test.pm25.values

    # data = df_train.groupby(["PRES"]).get_group(1010.0)["pm25"].values
    # plt.hist(data,bins=50)
    # plt.show()
    # raise Exception()
    print(pres_train)
    print(pm25_train)

    regMdn = KdeMdn(DbestConfig(), b_store_training_data=True)
    regMdn.fit(pres_train, pm25_train, runtime_config)
    regMdn.plot_density_per_group()
    # regMdn.predict(pres_train, b_show_plot=True)
    # regMdn.predict(pres_test, b_show_plot=True)

    # print(regMdn.predict([[1010]], 200, b_plot=False))
    # # xxs, yys = regMdn.predict([[1030]], 200, b_plot=True)
    # xxs, yys = regMdn.predict([[1010]], 200, b_plot=True)
    # xxs = [regMdn.denormalize(xi, regMdn.meanx, regMdn.widthx) for xi in xxs]
    # yys = [yi / regMdn.widthx * 2 for yi in yys]
    #
    # plt.plot(xxs, yys)
    # plt.show()


def test_ss_2d_density():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    file = "/data/tpcds/40G/ss_600k_headers.csv"
    # file = "/Users/scott/projects/ss_600k_headers.csv"
    df = pd.read_csv(file, sep='|')
    df = df.dropna(subset=['ss_sold_date_sk', 'ss_store_sk', 'ss_sales_price'])
    df_train = df.head(5000)
    df_test = df  # .head(1000)
    g_train = df_train.ss_store_sk.values[:, np.newaxis]
    x_train = df_train.ss_sold_date_sk.values  # df_train.pm25.values
    g_test = df_test.ss_store_sk.values
    # temp_test = df_test.PRES.values  #df_test.pm25.values

    # data = df_train.groupby(["ss_store_sk"]).get_group(1)["ss_sold_date_sk"].values
    # plt.hist(data, bins=100)
    # plt.show()
    # raise Exception()

    kdeMdn = KdeMdn(b_store_training_data=True, b_one_hot=True)
    kdeMdn.fit(g_train, x_train, num_epoch=1,
               num_gaussians=10, b_grid_search=False)

    # kdeMdn=de_serialize("/Users/scott/projects/mdn.dill")

    # kdeMdn.plot_density_3d()

    # regMdn = RegMdn(dim_input=1, b_store_training_data=True)
    # regMdn.fit(g_train, x_train, num_epoch=100, b_show_plot=False, num_gaussians=5)
    #
    # print(kdeMdn.predict([[1]], 2451119, b_plot=True))
    print(kdeMdn.predict([1, 2], [2451119, 2451120, 2451121], b_plot=False))
    # xxs, p = kdeMdn.predict([1, 2],  [2451119], b_plot=True)
    # xxs = [kdeMdn.denormalize(xi, kdeMdn.meanx, kdeMdn.widthx)
    #        for xi in xxs]
    # print(xxs, p)
    # # yys = [regMdn.denormalize(yi, regMdn.meany, regMdn.widthy) for yi in yys]
    # plt.plot(xxs, p)
    # plt.show()


# def test_pm25_3d_density():
#     import pandas as pd
#     file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
#     df = pd.read_csv(file)
#     df = df.dropna(subset=['pm25', 'PRES', 'TEMP'])
#     df_train = df.head(1000)
#     df_test = df.tail(1000)
#     pres_train = df_train.PRES.values
#     temp_train = df_train.TEMP.values
#     pm25_train = df_train.PRES.values  # df_train.pm25.values
#     pres_test = df_test.PRES.values
#     pm25_test = df_test.pm25.values
#     temp_test = df_test.PRES.values  # df_test.TEMP.values
#     xzs_train = np.concatenate(
#         (temp_train[:, np.newaxis], pres_train[:, np.newaxis]), axis=1)
#     xzs_test = np.concatenate(
#         (temp_test[:, np.newaxis], pres_test[:, np.newaxis]), axis=1)
#     regMdn = RegMdn(dim_input=2)
#     regMdn.fit(xzs_train, pm25_train, num_epoch=1000, b_show_plot=True)
#     regMdn.predict(xzs_test, b_show_plot=True)
#     regMdn.predict(xzs_train, b_show_plot=True)

def test_gm():
    from sklearn import mixture
    import random
    kde = mixture.GaussianMixture(n_components=2, covariance_type='spherical')
    kde.fit(np.random.rand(100, 1))
    # x = np.array(np.linspace(-5, 15, 100)).reshape(-1, 1)
    # print(x)
    # y = kde.predict(x)

    kde.weights_ = np.array([0.5, 0.5])
    kde.means_ = np.array([[1], [3]])
    kde.covariances_ = np.array([10, 20])

    x = np.array(np.linspace(-5, 15, 1000)).reshape(-1, 1)
    print(x)
    y = kde.score_samples(x)
    y = np.exp(y)
    print(y)

    plt.plot(x, y)
    plt.show()


def test_gmm():
    weights = [0.5, 0.5]
    mus = [0, 10]
    vars = [4, 4]
    xs = np.array(np.linspace(-5, 15, 10))
    # results = [gm(weights, mus, vars, x) for x in xs]
    print([gm(weights, mus, vars, x) for x in xs])
    results = gm(weights, mus, vars, xs)
    print(results)
    plt.plot(xs, results)
    plt.show()


def test_ss_3d():
    import pandas as pd
    file = "/data/tpcds/1G/ss_10k.csv"
    # file = "/data/tpcds/1t/ss_1m.csv"
    df = pd.read_csv(file, sep="|", usecols=[
                     'ss_sales_price', 'ss_sold_date_sk', 'ss_store_sk'])
    # df = df.dropna(subset=['ss_list_price', 'ss_sales_price', 'ss_store_sk'])
    df = df.dropna(subset=['ss_sales_price', 'ss_sold_date_sk', 'ss_store_sk'])
    # one_hot = pd.get_dummies(df["ss_store_sk"])
    # df = df.drop("ss_store_sk",axis=1)
    # df = df.join(one_hot)
    # print(df)
    # row_one_hot = pd.DataFrame({'ss_store_sk':"1.0"})#
    # row_one_hot['ss_store_sk']=row_one_hot['ss_store_sk'].astype('category',categories=["1.0","2.0","4.0","7.0","8.0","10.0"])
    # print(pd.get_dummies(row_one_hot))
    # row = pd.DataFrame({'ss_sales_price':99.9})
    # raise Exception()

    df_train = df  # .head(5000)
    df_test = df  # .head(5000)
    x_train = df_train.ss_sold_date_sk.values
    z_train = df_train.ss_store_sk.values
    # print(x_train)
    # print(z_train)
    # z_train=z_train[:,np.newaxis]
    # print(z_train)
    # enc = OneHotEncoder(handle_unknown='ignore')
    # z_train = enc.fit_transform(z_train).toarray()
    # print(z_train)

    # raise Exception()
    y_train = df_train.ss_sales_price.values
    x_test = df_test.ss_sold_date_sk.values
    y_test = df_test.ss_sales_price.values
    z_test = df_test.ss_store_sk.values
    # z_test = z_test[:, np.newaxis]
    # z_test = enc.transform(z_test).toarray()

    xzs_train = np.concatenate(
        (x_train[:, np.newaxis], z_train[:, np.newaxis]), axis=1)
    xzs_test = np.concatenate(
        (x_test[:, np.newaxis], z_test[:, np.newaxis]), axis=1)
    regMdn = RegMdn(dim_input=2, n_mdn_layer_node=20)
    regMdn.fit(xzs_train, y_train, num_epoch=10, b_show_plot=False)
    print(regMdn.predict(xzs_test, b_show_plot=True))


def test_onehot():
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)


def bin_wise_error_ss():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    file = "/data/tpcds/40G/ss_600k_headers.csv"
    # file = "/Users/scott/projects/ss_600k_headers.csv"
    df = pd.read_csv(file, sep='|')
    df = df.dropna(subset=['ss_sold_date_sk', 'ss_store_sk', 'ss_sales_price'])
    # df = df.head(10000)
    df_train = df.head(1000)
    df_test = df  # .head(1000)
    g_train = df_train.ss_store_sk.values[:, np.newaxis]
    x_train = df_train.ss_sold_date_sk.values  # df_train.pm25.values
    g_test = df_test.ss_store_sk.values
    # temp_test = df_test.PRES.values  #df_test.pm25.values

    # data = df_train.groupby(["ss_store_sk"]).get_group(1)["ss_sold_date_sk"].values
    # plt.hist(data, bins=100)
    # plt.show()
    # raise Exception()

    kdeMdn = KdeMdn("cpu", b_store_training_data=True, b_one_hot=True)
    kdeMdn.fit(g_train, x_train, b_grid_search=False)
    # kdeMdn.fit_grid_search(g_train, x_train)

    # kdeMdn=de_serialize("/Users/scott/projects/mdn.dill")

    kdeMdn.score()


def test_RegMdnGroupBy():
    import pandas as pd
    file = "/home/u1796377/Programs/dbestwarehouse/pm25.csv"
    df = pd.read_csv(file)
    df = df.dropna(subset=['pm25', 'PRES', 'TEMP'])
    df_train = df  # .head(1000)
    df_test = df  # .tail(1000)
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
    regMdn = RegMdnGroupBy(b_store_training_data=True)
    regMdn.fit(pres_train, temp_train, pm25_train, n_epoch=5, n_gaussians=10)
    print(pres_train, temp_train, pm25_train)
    print("*"*10)
    # print(regMdn.predict(pres_train[:5], temp_train[:5], b_plot=False))
    # print("*"*10)
    print(regMdn.predict([1010, 1020], [-2, -2], b_plot=False))
    # regMdn.score()


if __name__ == "__main__":
    # test_pm25_2d_density()
    # test_pm25_2d_density()
    # test_pm25_3d()
    # test_ss_3d()
    # test_ss_3d()
    # test_ss_2d_density()
    # test_gmm()
    # bin_wise_error_ss()
    # test_RegMdnGroupBy()
    # test_ss_2d_density()
    TestGenericMdn()
