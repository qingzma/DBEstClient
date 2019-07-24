# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

"""
Python implementation for
http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
"""

from __future__ import with_statement, print_function, division
try:
    range = xrange
except NameError:
    pass

from sys import stderr, stdin
from random import random
from math import log
import pandas as pd


class ReservoirSampling:
    def __init__(self):
        self.header = None
        self.n_total_point = None
        self.sampledf = None

    def build_reservoir(self, file, R, threshold=None, verbose=False,split_char=","):

        self.n_total_point = sum(1 for _ in open(file)) - 1

        with open(file,'r') as data:
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
            # skip the first header row
            first_row = next(iterator)
            self.header = first_row.replace("\n",'').split(split_char)
            try:
                j = 0
                # iterator = iter(data)
                while True:
                    j += 1
                    item = next(iterator)
                    if len(res) < R:
                        item = item.replace("\n",'').split(split_char)
                        p('> Adding element nb {0}: {1!r}', len(res), item)
                        res.append(item)

                    elif j < threshold:
                        k = int(random() * j)
                        if k < R:
                            p('> [p={0}/{1:>9}] Swap element nb {2:>5}: {3!r} replaces {4!r}', R, j, k, item, res[k])
                            item = item.replace("\n",'').split(split_char)
                            res[k] = item
                    else:
                        gap = int(log(random()) / log(1 - R / j))
                        j += gap
                        for _ in range(gap):
                            item = next(iterator)
                        k = int(random() * R)
                        p('> After skipping {0:>9} lines, swap element nb {1:>5}: {2!r} replaces {3!r}', gap, k, item, res[k])
                        item = next(iterator).replace("\n", '').split(split_char)
                        res[k] = item

            except KeyboardInterrupt:
                print('\n! User interrupted the process, stopping now\n', file=stderr)
            except StopIteration:
                pass

            self.sampledf =  pd.DataFrame(res, columns=self.header)


    def getyx(self, y, x, dropna=True):
        # drop non-numerical values.
        if dropna:
            self.sampledf = self.sampledf.dropna(subset=[y, x])
        self.sampledf[x] = pd.to_numeric(self.sampledf[x], errors='coerce').fillna(0)
        self.sampledf[y] = pd.to_numeric(self.sampledf[y], errors='coerce').fillna(0)

        return self.sampledf
        # return self.sampledf[y].values, self.sampledf[x].values.reshape(-1,1)
        # else:
        #     xyvalues={} #  {'groupby': groupby_attribute}
        #     sample_grouped = self.sampledf.groupby(by=groupby_attribute)
        #     for name, group in sample_grouped:
        #         xyvalues[name]=group
        #     return xyvalues


if __name__ == '__main__':
    file = '../../resources/pm25.csv'
    with open(file, 'r') as f:
        sample = ReservoirSampling().build_reservoir(f,10000, verbose=False)
    print(sample)


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