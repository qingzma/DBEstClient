# Created by Qingzhi Ma at 20/02/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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

            if line!="":
                if line.strip():
                    # print(line)
                    key_value = line.replace(
                        "(", " ").replace(")", " ").replace(";", "").replace("\n", "")  # .replace(",", "")
                    # print(key_value)
                    # self.logger.logger.info(key_value)
                    key_value = re.split(split_char, key_value)
                    # print(key_value)
                    if key_value[0] == "":
                        continue
                    # remove empty strings caused by sequential blank spaces.
                    key_value = list(filter(None, key_value))
                    if key_value[0] != '0':

                        key_value[0] = key_value[0].replace(",", "")
                        # print(key_value)
                        key_values[key_value[0]] = float(key_value[1])

    if ('NULL' in key_values) and b_remove_null:
        key_values.pop('NULL', None)
    if ('0' in key_values) and b_remove_null:
        key_values.pop('0', None)

    return key_values

def compare_dicts(true, pred):
    res=[]
    for key in true:
        res.append(abs(((true[key]-pred[key])/true[key]) )  )
        # print(key)
    # print(res)
    # plt.hist(res,bins=50)
    # plt.show()
    return res

def plot_count():
    mdn = read_results("mdn40/count1gg20.txt",split_char=",")
    truth = read_results("mdn40/count1truth.txt")
    kde = read_results("mdn40/count1gg10.txt")
    res0 = compare_dicts(truth, mdn)
    res1 = compare_dicts(truth, kde)
    res0 = [res * 100 for res in res0]
    res1 = [res * 100 for res in res1]
    plt.hist(res0, bins=50, color="r", alpha=0.2, label="MDN")
    plt.hist(res1, bins=50, color="b", alpha=0.6, label="DBEst")
    plt.legend()
    plt.title("Histogram of relative error for COUNT")
    plt.ylabel("Frequency")
    plt.xlabel("Relative error")
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    plt.gca().xaxis.set_major_formatter(xticks)
    # plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    plt.text(10, 3, "MDN error " + str(sum(res0) / len(res0)) + "%")
    plt.text(10, 2, "DBEst error " + str(sum(res1) / len(res1)) + "%")
    plt.show()

def plt501():
    mdn = read_results("mdn501/ss1t_1m_gg4.txt",split_char=",")
    truth = read_results("groundtruth/count1.result")
    kde = read_results("DBEst/count1.txt")
    res0 = compare_dicts(truth, mdn)
    res1 = compare_dicts(truth, kde)
    res0 = [res * 100 for res in res0]
    res1 = [res * 100 for res in res1]
    plt.hist(res0, bins=50, color="r", alpha=0.2, label="MDN")
    plt.hist(res1, bins=50, color="b", alpha=0.6, label="DBEst")
    plt.legend()
    plt.title("Histogram of relative error for COUNT")
    plt.ylabel("Frequency")
    plt.xlabel("Relative error")
    fmt = '%.2f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    plt.gca().xaxis.set_major_formatter(xticks)
    # plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    plt.text(10, 3, "MDN error " + str(sum(res0) / len(res0)) + "%")
    plt.text(10, 2, "DBEst error " + str(sum(res1) / len(res1)) + "%")

    print("MDN error " + str(sum(res0) / len(res0)) + "%")
    print("DBEst error " + str(sum(res1) / len(res1)) + "%")

    plt.show()


if __name__=="__main__":
    plt501()

