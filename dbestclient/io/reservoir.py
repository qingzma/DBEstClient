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
from statsmodels.compat.pandas import frequencies

from dbestclient.tools.variables import UseCols

# try:
#     range = xrange
# except NameError:
#     pass


class ReservoirSampling:
    def __init__(self, headers):
        self.header = headers
        self.n_total_point = None
        self.sampledf = None
        self.sampledfmean = None
        self.usecols = None
        self.origin_sample = None

    def build_reservoir(self, file, R, threshold=None, verbose=False, split_char=",", save2file=None, n_total_point=None, usecols=None):
        self.usecols = usecols
        self.n_total_point = n_total_point['total'] if n_total_point is not None else sum(
            1 for _ in open(file)) - 1

        print("Reading data file...")
        with open(file, 'r') as data:
            if verbose:
                def p(s, *args):
                    print(s.format(*args), file=stderr)
            else:
                def p(*_):
                    pass

            # check if the size R (number of rows in the sample) is passed, if not, return the file as a sample.
            if R is None:
                R = self.n_total_point
            if isinstance(R, str):
                R = self.n_total_point

            if threshold is None:
                threshold = 4 * R
            res = []

            iterator = iter(data)
            if self.header is None:
                # skip the first header row
                first_row = next(iterator)
                self.header = first_row.replace(
                    "\n", '').lower().split(split_char)
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

            # save the sample as pandas dataframe
            self.origin_sample = pd.DataFrame(res, columns=self.header)
            if usecols is not None:
                # print("self.origin_sample", self.origin_sample)
                # print("use_columns,", usecols)
                columns_as_continous, columns_as_categorical, _ = UseCols(
                    usecols).get_continous_and_categorical_cols()

                for col in columns_as_categorical:
                    self.origin_sample[col] = self.origin_sample[col].astype(
                        str)

                for col in columns_as_continous:
                    self.origin_sample[col] = self.origin_sample[col].apply(
                        pd.to_numeric, errors='coerce')

                self.origin_sample = self.origin_sample[columns_as_continous+columns_as_categorical].dropna(
                    subset=columns_as_continous)
                # print("self.origin_sample",)
                # print(self.origin_sample)
                # raise Exception

            self.sampledf = pd.DataFrame(res, columns=self.header)
            # print(self.sampledf)
            if usecols is not None:
                # # firstlt check whether y is categorical, if so, convert the categorical attributes to real values.
                # if usecols['y'][1] == "categorical":
                #     print("usecols['y'][1] is categorical")
                #     raise Exception

                # process the usecols from dic to list
                columns_continous = [usecols['y'][0]]

                if usecols['x_continous']:
                    columns_continous = columns_continous + \
                        usecols['x_continous']

                columns_categorial = []
                if usecols["x_categorical"]:
                    # columns = columns + usecols['x_categorical']
                    columns_categorial = columns_categorial + \
                        usecols['x_categorical']
                if usecols["gb"]:
                    for col in usecols["gb"]:
                        if col not in columns_continous + columns_categorial:
                            columns_categorial.append(col)
                        else:
                            print(
                                "SQL meets the condition where Group By attributes and X attributes have common attributes: " + col)

                            # columns = columns + usecols['gb']
                    # gb_cols = ["gb_"+i for i in usecols['gb']]
                    # columns_categorial = columns_categorial + gb_cols
                usecols_list = columns_continous + columns_categorial

                # print(self.sampledf)
                # print(self.sampledf["tenantid"])
                # print(usecols)
                # print("usecols_list", usecols_list)

                self.sampledf = self.sampledf[usecols_list]
                # print(self.sampledf)
                # print(columns_continous)
                # print(columns_categorial)
                # print(self.sampledf)

                # convert continuous X attributes to float, execept for those that repeated in the GROUP BY clause.
                # print(usecols)
                # print(usecols['x_continous'])
                if usecols["gb"] is not None:
                    columns_to_float = [
                        item for item in usecols['x_continous'] if item not in usecols["gb"]]
                else:
                    columns_to_float = [
                        item for item in usecols['x_continous']]
                # print("columns_to_float", columns_to_float)
                for col in columns_to_float:
                    self.sampledf[col] = self.sampledf[col].apply(
                        pd.to_numeric, errors='coerce')
                # self.sampledf[columns_continous] = self.sampledf[columns_continous].apply(
                #     pd.to_numeric, errors='coerce')
                # self.sampledf.dropna()
                # print(self.sampledf)
                # for col in columns_continous:  # [0:-1]:
                #     print("col,", col)
                #     print("df column " + str(col))
                #     print(self.sampledf[col])

                #     self.sampledf[col] = pd.to_numeric(
                #         self.sampledf[col], errors='coerce')
                #     print(self.sampledf)
                #     print("*"*80)
                # convert the group by column to string

                for col in columns_categorial:
                    self.sampledf[col] = self.sampledf[col].astype(str)

                # if col in both X and Group BY, convert it to float
                if usecols["gb"] is not None:
                    columns_common = [
                        item for item in usecols['x_continous'] if item in usecols["gb"]]

                    for col in columns_common:
                        self.sampledf[col] = self.sampledf[col].apply(
                            pd.to_numeric, errors='coerce')

                if usecols['y'][1] == "categorical":
                    self.sampledf[usecols['y'][0]
                                  ] = self.sampledf[usecols['y'][0]].astype(str)
                else:
                    self.sampledf[usecols['y'][0]
                                  ] = self.sampledf[usecols['y'][0]].apply(
                        pd.to_numeric, errors='coerce')
                # print("usecols", usecols)
                self.sampledf = self.sampledf.dropna(subset=usecols_list)
                # print("self.sampledf", self.sampledf)
                # print("type is ", self.sampledf.dtypes)
                # raise Exception
                self.columns_categorical = columns_categorial
                self.column_continous = columns_continous

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
        # if dropna:
        #     self.sampledf = self.sampledf.dropna(subset=[y, x])
        # self.sampledf[x] = pd.to_numeric(
        #     self.sampledf[x], errors='coerce').fillna(0)
        # self.sampledf[y] = pd.to_numeric(
        #     self.sampledf[y], errors='coerce').fillna(0)

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

    def get_groupby_frequency_and_data(self):
        print("get frequency info from data....")
        # print("self.sampledf", self.sampledf)
        total_frequency = {}
        data = {}
        if self.usecols["x_categorical"]:
            total_frequency["if_contain_x_categorical"] = True
            data["if_contain_x_categorical"] = True
            gb = self.sampledf.groupby(self.usecols["x_categorical"])
            for grp, values in gb:
                # print(grp, type(grp))

                # print("*"*40)
                if isinstance(grp, tuple):
                    key = ",".join(grp)
                else:
                    key = grp
                # print(key)
                # print('key', key)
                # print(values)
                # print([self.usecols["y"]] +
                #       self.usecols["x_continous"] +
                #       self.usecols["gb"])
                columns_to_use = [self.usecols["y"][0]]
                columns_to_float = [
                    item for item in self.usecols['x_continous'] if item not in self.usecols["gb"]]
                columns_to_use = columns_to_use + \
                    columns_to_float + self.usecols["gb"]
                gb_data = values[columns_to_use]
                # gb_data = values[[self.usecols["y"]] +
                #                  self.usecols["x_continous"] +
                #                  self.usecols["gb"]]

                # to get the frequency for each sub group
                # print(gb_data)

                # check if DISTINCT is involved in the SQL
                # DISTINCT is not involved
                if not self.usecols["y"][2]:
                    gb_by_group_by_atrributes = values.groupby(
                        self.usecols["gb"]).size().to_frame(name='count').reset_index()
                    # print(gb_by_group_by_atrributes)
                    # for row in gb_by_group_by_atrributes.itertuples():
                    #     print(row)

                    frequency = {}
                    for row in gb_by_group_by_atrributes.itertuples():
                        # print(row)
                        # print(type(row))
                        columns_str = []
                        for item in list(row[1:-1]):
                            columns_str.append(str(item))
                        key1 = ",".join(columns_str)
                        # print(key1)
                        # raise Exception
                        count = row[-1]
                        frequency[key1] = count
                        # print(key1, count)
                    # print("grp", grp, frequency)
                    # print("*"*100)
                    # print()
                else:
                    # print("start distinct sampling")
                    # print("gb,", gb)
                    # print("values")
                    # print(values)
                    gb_by_group_by_atrributes = values.groupby(
                        self.usecols["gb"])[columns_to_use].nunique()  #

                    # print(gb_by_group_by_atrributes)
                    frequency = {}
                    for row in gb_by_group_by_atrributes.itertuples():
                        # print(row)
                        # print(row[0])
                        # print(type(row[0]))
                        columns_str = []
                        if isinstance(row[0], int):
                            key1 = str(row[0])
                        else:
                            for item in list(row[0]):
                                columns_str.append(str(item))
                            key1 = ",".join(columns_str)
                        # print(key1)
                        # print(row[0])
                        # print(row[1])
                        count = row[1]
                        frequency[key1] = count
                total_frequency[key] = frequency
                data[key] = gb_data

        else:
            # check if DISTINCT is involved in the SQL
            # DISTINCT is not involved
            if not self.usecols["y"][2]:
                total_frequency["if_contain_x_categorical"] = False
                data["if_contain_x_categorical"] = False
                # groupbys = self.usecols["gb"]
                gb = self.sampledf.groupby(self.usecols["gb"]).size().to_frame(
                    name='count').reset_index()
                # frequency = {}
                for row in gb.itertuples():
                    # print(row)
                    # print(type(row))
                    key = ",".join(list(row[1:-1]))
                    count = row[-1]
                    total_frequency[key] = count
                    # print(key, count)
                data["data"] = self.sampledf
            else:
                total_frequency["if_contain_x_categorical"] = False
                data["if_contain_x_categorical"] = False
                gb = self.sampledf.groupby(
                    self.usecols["gb"])[columns_to_use].nunique()  #

                # print(gb_by_group_by_atrributes)
                # frequency = {}
                for row in gb.itertuples():
                    # print(row)
                    # print(row[0])
                    # print(type(row[0]))
                    columns_str = []
                    if isinstance(row[0], int):
                        key1 = str(row[0])
                    else:
                        for item in list(row[0]):
                            columns_str.append(str(item))
                        key1 = ",".join(columns_str)
                    # print(key1)
                    # print(row[0])
                    # print(row[1])
                    count = row[1]
                    # frequency[key1] = count
                    total_frequency[key1] = count
                data["data"] = self.sampledf

        # for key in total_frequency:
        #     print(key, total_frequency[key])
        #     print("-"*20)
        total_frequency["x_categorical_columns"] = self.usecols["x_categorical"]

        # get the distinct values of categorical x attributes, which will be used in where clause, like where x2>=10

        categorical_distinct_values = {}
        for col in self.usecols["x_categorical"]:
            # print("col", col)
            try:
                float(self.sampledf[col].iloc[0])
                distinct_values = self.sampledf[col].unique().tolist()
                categorical_distinct_values[col] = distinct_values
            except ValueError:
                pass
        total_frequency["categorical_distinct_values"] = categorical_distinct_values
        if not total_frequency['if_contain_x_categorical']:
            total_frequency.pop("categorical_distinct_values")
            total_frequency.pop("x_categorical_columns")
        # print("total_frequency,", total_frequency)
        return total_frequency, data

    def get_columns_from_original_sample(self, gb, x, y):
        return self.origin_sample[gb].values, self.origin_sample[x].values.reshape(1, -1)[0], self.origin_sample[y].values.reshape(1, -1)[0]

    def get_frequency_of_categorical_columns_for_gbs(self, gbs, categoricals):
        frequencies = {}
        gb = self.origin_sample.groupby(categoricals)
        for grp, values in gb:
            # print(grp, type(grp))

            # print("*"*40)
            if isinstance(grp, tuple):
                key = ",".join(grp)
            else:
                key = grp
            # print("key", key)
            # print(values)
            gb_in_gb = values.groupby(
                gbs).size().to_frame(name='count').reset_index()
            # print(gb_in_gb)

            frequency = {}
            for row in gb_in_gb.itertuples():
                columns_str = []
                for item in list(row[1:-1]):
                    columns_str.append(str(item))
                key1 = ",".join(columns_str)
                count = row[-1]
                frequency[key1] = count
            frequencies[key] = frequency
        # print(frequencies)
        return frequencies


# if __name__ == '__main__':
#     file = '/home/u1796377/Programs/dbestwarehouse/pm25.csv'
#     # with open(file, 'r') as f:
#     sampler = ReservoirSampling()
#     sampler.build_reservoir(file, 10000, verbose=False)
#     xy = sampler.getyx('pm25', 'PRES')
#     ft = sampler.get_frequency('pm25', 'PRES')
#     print(ft)
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
