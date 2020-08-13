# Created by Qingzhi Ma at 20/02/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import re
from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.pyplot import axis, legend


def read_results(file, b_remove_null=True, split_char="\s"):
    """read the group by value and the corresponding aggregate within
    a given range, used to compare the accuracy.the

    Output: a dict contating the

    Args:
        file (file): path to the file
    """

    key_values = {}
    with open(file) as f:
        # print("Start reading file " + file)
        index = 1
        for line in f:
            # ignore empty lines

            if line != "":
                if line.strip():
                    # print(line)
                    # print("key_value",line)
                    key_value = line.replace('OH (1)',"OH    ").replace(
                        "(", " ").replace(")", " ").replace(";", "").replace("\n", "").replace(
                        "                                                                                                  ","").replace(
                        "	",",").replace("OH    ", "OH")  # .replace(",", "")
                    # print("key_value",key_value)
                    # self.logger.logger.info(key_value)
                    key_value = re.split(split_char, key_value)
                    # print(key_value)
                    if key_value[0] == "" or key_value[0] == "NULL":
                        continue
                    # remove empty strings caused by sequential blank spaces.
                    key_value = list(filter(None, key_value))
                    if key_value[0] != '0':

                        key_value[0] = key_value[0].replace(",", "")
                        # print(key_value)
                        # print(int('506.0'))
                        key_values[key_value[0]] = float(key_value[1])

    if ('NULL' in key_values) and b_remove_null:
        key_values.pop('NULL', None)
    if ('0' in key_values) and b_remove_null:
        key_values.pop('0', None)

    # print("key_values",key_values)
    return key_values


def compare_dicts(true: dict, pred: dict) -> dict:
    """compare the error between two dicts

    Args:
        true (dict): truth
        pred (dict): prediction

    Returns:
        dict: relative error
    """
    res = []
    for key in true:
        # print("key", key)
        res.append(abs(((true[key]-pred[key])/true[key])))
        # print(key)
    # print(res)
    # plt.hist(res,bins=50)
    # plt.show()
    return res



def plt_workload(agg_func="avg", suffix="_ss1t_gg4.txt", b_plot=True, b_merge_result_for_group=False, b_two_methods=False):
    mdn_errors = []
    kde_errors = []
    # prapare the files.

    if agg_func == "count":
        files = [7]
    elif  agg_func == "sum":
        files = [10,11,12]
    elif agg_func == "avg":
        files = [13,14,15]
    else:
        raise TypeError("wrong aggregate function.")
    for file_name in files:
        # prefix = agg_func+str(i)
        mdn = read_results("experiments/flights/results/mdn1m/" +
                           str(file_name)+".txt", split_char=",")
        truth = read_results(
            "experiments/flights/results/truth/"+str(file_name)+".csv", split_char=",")
        mdn_error = compare_dicts(truth, mdn)
        mdn_errors.append(mdn_error)
        if b_two_methods:
            kde = read_results(
                "experiments/flights/results/deepdb/"+str(file_name)+".txt", split_char=",")
            kde_error = compare_dicts(truth, kde)
            kde_errors.append(kde_error)

    # mdn_errors = mdn_errors*100
    # if b_two_methods:
    #     kde_errors = kde_errors*100

    if b_plot:
        if b_merge_result_for_group:
            mdn_errors = np.array(mdn_errors)
            mdn_errors = np.mean(mdn_errors, axis=0)

            if b_two_methods:
                kde_errors = np.array(kde_errors)
                kde_errors = np.mean(kde_errors, axis=0)
        else:
            mdn_errors = np.array(list(chain.from_iterable(mdn_errors)))
            if b_two_methods:
                kde_errors = np.array(list(chain.from_iterable(kde_errors)))

        mdn_errors = mdn_errors*100
        if b_two_methods:
            kde_errors = kde_errors*100

        plt.hist(mdn_errors, bins=50, color="r", alpha=0.2, label="DBEst-MDN")
        if b_two_methods:
            plt.hist(kde_errors, bins=50, color="b", alpha=0.6, label="DBEst")
        plt.legend()
        plt.title("Histogram of relative error for " + agg_func.upper())
        plt.ylabel("Frequency")
        plt.xlabel("Relative error")
        fmt = '%.2f%%'
        xticks = mtick.FormatStrFormatter(fmt)
        plt.gca().xaxis.set_major_formatter(xticks)
        # plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

        plt.text(10, 150, "DBEst-MDN error " +
                 "{0:.2f}".format(np.mean(mdn_errors)) + "%")
        if b_two_methods:
            plt.text(10, 180, "DBEst error " +
                     "{0:.2f}".format(np.mean(kde_errors)) + "%")

        plt.show()
    print(agg_func)
    print("MDN error " + str(np.mean(mdn_errors)) + "%")
    if b_two_methods:
        print("DBEst error " + str(np.mean(kde_errors)) + "%")
    if b_two_methods:
        return np.mean(mdn_errors), np.mean(kde_errors)
    else:
        return np.mean(mdn_errors)


if __name__ == "__main__":
    plt_workload(agg_func="count", suffix=".txt", b_plot=False,
                    b_merge_result_for_group=False, b_two_methods=False)
    plt_workload(agg_func="sum", suffix=".txt", b_plot=False,
                    b_merge_result_for_group=False, b_two_methods=False)
    plt_workload(agg_func="avg", suffix=".txt", b_plot=False,
                    b_merge_result_for_group=False, b_two_methods=False)