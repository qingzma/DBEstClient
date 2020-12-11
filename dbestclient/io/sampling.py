# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import pandas as pd

from dbestclient.io.reservoir import ReservoirSampling
from dbestclient.io.stratifiedreservoir import StratifiedReservoir


class DBEstSampling:
    def __init__(self, headers, usecols, n_jobs=1, mdl_name="mdl",warehouse="dbestwarehouse"):
        self.n_sample_point = None
        self.n_total_point = None
        self.sample = None
        self.sample_mean = None
        self.headers = headers
        self.usecols = usecols
        self.original_table_size = None
        self.scaling_factor = None
        self.method = None
        self.n_jobs = n_jobs
        self.mdl_name = mdl_name
        self.warehouse=warehouse

    def make_sample(
        self,
        file,
        ratio,
        method="uniform",
        split_char=",",
        file2save=None,
        num_total_records=None,
    ):
        self.method = method.lower()
        if method == "uniform":
            # # if  ratio is provided, then make samples using the ratio (or size)
            # if ratio is not None:
            if isinstance(ratio, str):
                print("The given table is treated as a uniform sample")
                self.sample = ReservoirSampling(headers=self.headers)
                self.sample.build_reservoir(
                    file,
                    None,
                    split_char=split_char,
                    save2file=file2save,
                    n_total_point=num_total_records,
                    usecols=self.usecols,
                )
                return
            if float(ratio) > 1:  # here the ratio is the number of tuples in the sample
                ratio = int(ratio)
                self.n_sample_point = ratio
                self.sample = ReservoirSampling(headers=self.headers)
                self.sample.build_reservoir(
                    file,
                    ratio,
                    split_char=split_char,
                    save2file=file2save,
                    n_total_point=num_total_records,
                    usecols=self.usecols,
                )
                self.n_total_point = self.sample.n_total_point
                # print("total point", self.n_total_point)
                # print("sample point",self.n_sample_point)

                self.scaling_factor = self.n_total_point / ratio
                # return self.sample,    # data, scaling factor
            else:
                print(
                    "The given table is treated as a uniform sample, and it is obtained with sampling rate "
                    + str(ratio)
                )
                self.sample = ReservoirSampling(headers=self.headers)
                self.sample.build_reservoir(
                    file,
                    None,
                    split_char=split_char,
                    save2file=file2save,
                    n_total_point=num_total_records,
                    usecols=self.usecols,
                )
                self.n_total_point = self.sample.n_total_point / float(ratio)
                self.scaling_factor = 1 / float(ratio)
                # return self.sample, 1/float(ratio)
            # # if ratio is not provided, then the whole dataset is used to train the model.
            # else:
            #     self.sample = pd.read_csv(file)
            #     self.n_sample_point = len(self.sample)
            #     self.n_total_point = self.n_sample_point

            # delete the headers after models are trained, to save sapce.
            self.headers = None
            return
        elif method.lower() == "stratified":
            ratio = int(ratio)
            if ratio < 10:
                print("The maximum number of tuples per group is too small")
                return

            self.sample = StratifiedReservoir(
                file, file_header=self.headers, n_jobs=self.n_jobs, capacity=ratio,mdl_name=self.mdl_name,warehouse=self.warehouse
            )
            self.sample.make_sample_for_sql_condition(
                usecols=self.usecols, split_char=split_char
            )
        else:
            print("other sampling methods are not implemented, abort.")

    def getyx(self, y, x, dropna=True, b_return_mean=False, groupby=None):
        if self.method == "uniform":
            return self.sample.getyx(
                y, x, dropna=dropna, b_return_mean=b_return_mean, groupby=groupby
            )
        elif self.method == "stratified":
            return self.sample.get_categorical_features_label()
        else:
            raise TypeError("Unexpected method for sampling.")

    def get_groupby_frequency_data(self):
        if self.method == "uniform":
            return self.sample.get_groupby_frequency_and_data()
        elif self.method == "stratified":
            return self.sample.get_ft()
        else:
            raise TypeError("Unexpected method for sampling.")


# if __name__ == '__main__':
#     files = '../../resources/pm25.csv'
#     sampler = DBEstSampling()
#     # sampler.sample1()
#     sampler.make_sample(file=files, ratio=10)
#     # print(sampler.sample)
