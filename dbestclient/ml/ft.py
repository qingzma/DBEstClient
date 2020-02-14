# Created by Qingzhi Ma at 14/02/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
# this is the frequency table implementation.

import sys
import math
import pandas as pd


class GroupByFrequencyTableKeyStr:
    def __init__(self):
        self.fts = {}

    def init_from_file(self, file, groupby_attr, x, sep=","):
        df = pd.read_csv(file, sep=sep, dtype={groupby_attr: object})
        df = df.groupby([groupby_attr, x]).size().reset_index(name='counts')
        df[groupby_attr] = df[groupby_attr].astype(str)

        last_group =None
        for index, row in df.iterrows():
            # print(row[groupby_attr],row[x],row["counts"])
            if row[groupby_attr] != last_group:
                self.fts[row[groupby_attr]]=FrequencyTableKeyFloat()

            self.fts[row[groupby_attr]].add(row[x], row["counts"])
            last_group = row[groupby_attr]

    def print(self):
        for key in self.fts:
            print(key)
            self.fts[key].print()






class FrequencyTableKeyFloat:
    """ This is the frequency table implementation."""

    def __init__(self, ft=None):
        """
        Initilize the frequency table with a dictionary
        :param ft: a dictionary.
        """
        # init the frequency table ft
        if ft is not None:
            key_types = set([type(key) for key in ft])
            if len(key_types) != 1:
                raise KeyError("During initialization, mixed key types for FrequencyTableKeyFloat.")
            elif not isinstance(next(iter(ft)), float):
                print(next(iter(ft)))
                raise KeyError("During initialization, key type is not float for FrequencyTableKeyFloat.")
            else:
                self.ft = ft
        else:
            self.ft = {}
        # the sorted keys
        self.sortedKeys = sorted(self.ft.keys())

        self.counts = 0
        for key in self.ft.keys():
            self.counts += self.ft[key]

    def add(self, key, value):
        """
        Insert new values to the frequency table.
        :param key: [float]
        :param value: [int]
        """
        if (not isinstance(key, float)) or (not isinstance(value, int)):
            print("The date type inserted to FrequencyTableKeyFloat is not correct.")
        else:
            # check if the key is present in the frequency table
            if key in self.ft:
                self.ft[key] += value
            else:
                self.ft[key] = value
                self.sort()

            self.counts += value

    def print(self):
        """
        print out the frequency table.
        """
        for key in self.sortedKeys:
            print(key, self.ft[key])
        print("_________________________")

    def sort(self):
        """
        sort the frequency table based on the keys.
        """
        self.sortedKeys = sorted(self.ft.keys())

    def get_first_key_for_integral(self, xmin):
        return self.sortedKeys[binary_search_ge(xmin, self.sortedKeys)]

    # def get_last_key_for_integral(self, xmax):
    #     # index = binary_search_ge(xmax, self.sortedKeys)
    #     # if index ==0:
    #     #     raise RuntimeError("No condition is met.")
    #     return self.sortedKeys[binary_search_gt(xmax, self.sortedKeys)]


def binary_search_ge(x, search_list):
    """
    binary search for the first value from a list, whose value is greater or equal to x
    :param x: float, the number to compare with
    :param search_list: sorted list of numbers.
    :return: float, the first value in the list satisfying the condition.
    """
    # iterations = 1
    left = 0
    right = len(search_list) - 1
    mid = (right + left) // 2
    while right != left:
        if search_list[mid] >= x:
            right = mid
            # print("right is changed to "+str(right)+ ", left is "+str(left))
        else:
            left = mid + 1
            # print("left is changed to " + str(left))
        mid = (right + left) // 2
        # iterations +=1
        # if (iterations>20):
        #     raise RuntimeError("Maximum iteration achieved.")
    return mid


# def binary_search_gt(x, search_list):
#     """
#     binary search for the first value from a list, whose value is greater than x
#     :param x: float, the number to compare with
#     :param search_list: sorted list of numbers.
#     :return: float, the first value in the list satisfying the condition.
#     """
#     iterations = 1
#     left = 0
#     right = len(search_list) - 1
#     mid = (right + left) // 2
#     # mid = math.ceil((right + left) / 2)
#     while right != left:
#         if search_list[mid] >= x:
#             right = mid
#             print("right is changed to "+str(right)+ ", left is "+str(left))
#         else:
#             left = mid + 1
#             # print("left is changed to " + str(left))
#         mid = (right + left) // 2
#         # mid = math.ceil((right + left) / 2)
#         iterations +=1
#         if (iterations>20):
#             raise RuntimeError("Maximum iteration achieved.")
#     return mid

def test_groupby():
    file = '/data/tpcds/40G/ss_600k_headers.csv'
    groupFT = GroupByFrequencyTableKeyStr()
    groupFT.init_from_file(file,"ss_store_sk","ss_sales_price",sep="|")
    groupFT.print()




if __name__ == "__main__":
    # ft = FrequencyTableKeyFloat({15.0: 11, 11.0: 15})
    # ft.print()
    # ft.add(12.0, 12)
    # ft.print()
    # ft.add(12.0, 12)
    # ft.print()
    # print(ft.get_first_key_for_integral(11))
    # print(ft.get_last_key_for_integral(12))
    # print(binary_search_ge(3.1, [0, 1, 2, 3, 4, 5, 6, 7]))

    test_groupby()
