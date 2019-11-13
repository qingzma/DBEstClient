# Created by Qingzhi Ma at 08/11/2019
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from __future__ import with_statement, print_function, division


from sys import stderr, stdin
from random import random
from math import log
import pandas as pd


class FileNoSampling:
    def __init__(self):
        self.header = None
        self.n_total_point = None
        self.sampledf = None

    def read(self, file,split_char=","):
        self.sampledf = pd.read_csv(file)
        self.n_total_point = len(self.sampledf)
        self.header = self.sampledf.columns


    def getyx(self, y, x, dropna=True):
        # drop non-numerical values.
        if dropna:
            self.sampledf = self.sampledf.dropna(subset=[y, x])
        self.sampledf[x] = pd.to_numeric(self.sampledf[x], errors='coerce').fillna(0)
        self.sampledf[y] = pd.to_numeric(self.sampledf[y], errors='coerce').fillna(0)

        return self.sampledf


if __name__ == '__main__':
    file = '../../resources/pm25.csv'
    with open(file, 'r') as f:
        sample = FileNoSampling().read(f)
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



