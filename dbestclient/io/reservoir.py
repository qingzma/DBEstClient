# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

"""
Python implementation for
http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
"""

from __future__ import division, print_function, with_statement

from math import log
from random import random
from sys import stderr, stdin

import numpy as np
import pandas as pd

try:
    range = xrange
except NameError:
    pass


class ReservoirSampling:
    def __init__(self, headers):
        self.header = headers
        self.n_total_point = None
        self.sampledf = None
        self.sampledfmean = None

    def build_reservoir(self, file, R, threshold=None, verbose=False, split_char=",", save2file=None, n_total_point=None, usecols=None):

        self.n_total_point = n_total_point['total'] if n_total_point is not None else sum(
            1 for _ in open(file)) - 1

        with open(file, 'r') as data:
            if verbose:
                def p(s, *args):
                    print(s.format(*args), file=stderr)
            else:
                def p(*_):
                    pass

            if threshold is None:
                threshold = 4 * R
            res = []

            iterator = iter(data)
            if self.header is None:
                # skip the first header row
                first_row = next(iterator)
                self.header = first_row.replace("\n", '').split(split_char)
            try:
                j = 0
                # iterator = iter(data)
                while True:
                    j += 1
                    item = next(iterator)
                    if len(res) < R:
                        item = item.replace("\n", '').split(split_char)
                        p('> Adding element nb {0}: {1!r}', len(res), item)
                        res.append(item)

                    elif j < threshold:
                        k = int(random() * j)
                        if k < R:
                            p('> [p={0}/{1:>9}] Swap element nb {2:>5}: {3!r} replaces {4!r}',
                              R, j, k, item, res[k])
                            item = item.replace("\n", '').split(split_char)
                            res[k] = item
                    else:
                        gap = int(log(random()) / log(1 - R / j))
                        j += gap
                        for _ in range(gap):
                            item = next(iterator)
                        k = int(random() * R)
                        p('> After skipping {0:>9} lines, swap element nb {1:>5}: {2!r} replaces {3!r}',
                          gap, k, item, res[k])
                        item = next(iterator).replace(
                            "\n", '').split(split_char)
                        res[k] = item

            except KeyboardInterrupt:
                print('\n! User interrupted the process, stopping now\n', file=stderr)
            except StopIteration:
                pass

            # print(res)
            # print(self.header)
            self.sampledf = pd.DataFrame(res, columns=self.header)
            # print(self.sampledf)
            if usecols is not None:
                self.sampledf = self.sampledf[usecols]

                for col in usecols[0:2]:  # [0:-1]:
                    self.sampledf[col] = pd.to_numeric(
                        self.sampledf[col], errors='coerce')
                # convert the group by column to string
                if len(usecols) == 3:
                    # self.sampledf = self.sampledf.dropna(subset=[usecols[-1]])
                    self.sampledf[usecols[-1]
                                  ] = self.sampledf[usecols[-1]].astype(str)
                self.sampledf = self.sampledf.dropna(subset=usecols)

            if save2file is not None:
                self.sampledf.to_csv(save2file, index=False)

    def getyx(self, y, x, dropna=True, b_return_mean=False, groupby=None):
        # drop non-numerical values.
        # if dropna:
        #     self.sampledf = self.sampledf.dropna(subset=[y, x])
        # if groupby is not None:
        #     self.sampledf = self.sampledf.dropna(subset=[groupby])
        #     self.sampledf[groupby] = pd.to_numeric(self.sampledf[groupby], errors='coerce').fillna(0)
        # self.sampledf[x] = pd.to_numeric(self.sampledf[x], errors='coerce').fillna(0)
        # self.sampledf[y] = pd.to_numeric(self.sampledf[y], errors='coerce').fillna(0)

        if b_return_mean:
            gb = self.sampledf.groupby(x)
            means = gb.mean()

            keys = np.array(list(gb.groups.keys()))
            means = means.values.reshape(-1)
            # print("keys",keys)
            # print("means",means)
            self.sampledfmean = pd.DataFrame({x: keys, y: means})
            return self.sampledfmean, self.sampledf
        else:
            return self.sampledf
        # return self.sampledf[y].values, self.sampledf[x].values.reshape(-1,1)
        # else:
        #     xyvalues={} #  {'groupby': groupby_attribute}
        #     sample_grouped = self.sampledf.groupby(by=groupby_attribute)
        #     for name, group in sample_grouped:
        #         xyvalues[name]=group
        #     return xyvalues

    def get_frequency(self, y, x, dropna=True):
        # drop non-numerical values.
        if dropna:
            self.sampledf = self.sampledf.dropna(subset=[y, x])
        self.sampledf[x] = pd.to_numeric(
            self.sampledf[x], errors='coerce').fillna(0)
        self.sampledf[y] = pd.to_numeric(
            self.sampledf[y], errors='coerce').fillna(0)

        gb = self.sampledf.groupby(x)
        counts = gb.count()[y]
        # print(counts)

        keys = np.array(list(gb.groups.keys()))
        counts = counts.values.reshape(-1)
        # print("keys",keys)
        # print("counts",counts)
        ft = {}
        for key, count in zip(keys, counts):
            ft[key] = count
        return ft


if __name__ == '__main__':
    file = '/home/u1796377/Programs/dbestwarehouse/pm25.csv'
    # with open(file, 'r') as f:
    sampler = ReservoirSampling()
    sampler.build_reservoir(file, 10000, verbose=False)
    xy = sampler.getyx('pm25', 'PRES')
    ft = sampler.get_frequency('pm25', 'PRES')
    print(ft)
    # print(xy)
    #
    # print(xy.values)

    #
    # import pandas as pd
    # data = pd.DataFrame(sample)
    # print(data)
    #
    # data = pd.read_csv(file).iterrows()
    # sample = build_reservoir(data, 3, verbose=True)
    # print(sample)

    # import argparse
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('size', help="Reservoir size", type=int)
    # parser.add_argument('-t', '--threshold',
    #                     help=('threshold to start using gaps, default '
    #                           ' is 4 times the reservoir size'),
    #                     type=int)
    # parser.add_argument('-v', '--verbose', action='store_true')
    # args = parser.parse_args()
    #
    # for row in build_reservoir(stdin,
    #                            R=args.size,
    #                            threshold=args.threshold,
    #                            verbose=args.verbose):
    #     print(row, end="")
