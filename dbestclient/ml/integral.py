#
# Created by Qingzhi Ma on Fri Mar 13 2020
#
# Copyright (c) 2020 Department of Computer Science, University of Warwick
# Copyright 2020 Qingzhi Ma
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np


def approx_integrate(func: callable, x_lb: float, x_ub: float, n_division=20) -> float:
    """ simulate the integral using user-defined functions.

    Args:
        func (callable): the integral funcion, must be able to predict for a list of points.
        x_lb (float): lower bound
        x_ub (float): upper bound
        n_division (int, optional): the mesh division number. Defaults to 20.

    Returns:
        float: the approximate integral.
    """
    grid, step = np.linspace(x_lb, x_ub, n_division, retstep=True)
    # print(grid)
    # print(step)
    # print(func(grid)[0:-1].sum()*step)
    # print(func(grid)[1:].sum()*step)
    predictions = func(grid)
    return (0.5*(predictions[0]+predictions[-1]) + predictions[1:-1])*step
    # return (func(grid)[0:-1].sum() + func(grid)[1:].sum())*0.5*step


def prepare_density_data(func: callable, x_lb: float, x_ub: float, n_division=20, groups: list = None) -> dict:
    """provide the approximate results for all groups, in one call to the MDN.

    Args:
        func (callable): the MDN network
        x_lb (float): lower bound
        x_ub (float): upper bound
        n_division (int, optional): the mesh division number. Defaults to 20.
        groups (float): the groups that need to calculate. Defaults to None, and results for all groups are returned.

    Returns:
        dict: approximate anwers, with group as the key, and predictions as values.
    """
    if groups is None:
        groups = func.groupby_values

    group_values = np.linspace(x_lb, x_ub, n_division)*len(groups)
    print(group_values)

    return {}


def prepare_reg_density_data(density: callable, x_lb: float, x_ub: float, reg: callable = None, groups: list = None):
    return {}, {}

def approx_


def sin_(points: list) -> float:
    """ sin function, for testing purposes.

    Args:
        points (list[float]): points.

    Returns:
        float: the value.
    """
    return np.sin(points)


if __name__ == "__main__":
    print(integrate(sin_, 0, 3.1415926, 200))
