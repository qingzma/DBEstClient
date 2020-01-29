# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.io.reservoir import ReservoirSampling
import  pandas as pd


class DBEstSampling:
    def __init__(self, headers):
        self.n_sample_point = None
        self.n_total_point = None
        self.sample = None
        self.sample_mean = None
        self.headers=headers

    def make_sample(self, file, ratio,  method='uniform', split_char=',', file2save=None, num_total_records=None):
        if method == 'uniform':
            # # if  ratio is provided, then make samples using the ratio (or size)
            # if ratio is not None:
            if float(ratio) > 1: # here the ratio is the number of tuples in the sample
                ratio = int(ratio)
                self.n_sample_point = ratio
                self.sample = ReservoirSampling(headers=self.headers)
                self.sample.build_reservoir(file,ratio,split_char=split_char, save2file=file2save,n_total_point=num_total_records)
                self.n_total_point =  self.sample.n_total_point
                # print("total point", self.n_total_point)
                # print("sample point",self.n_sample_point)

                return self.sample
            else:
                print("sampling with probability is not implemented yet, abort")
            # # if ratio is not provided, then the whole dataset is used to train the model.
            # else:
            #     self.sample = pd.read_csv(file)
            #     self.n_sample_point = len(self.sample)
            #     self.n_total_point = self.n_sample_point
        else:
            print("other sampling methods are not implemented, abort.")

    def getyx(self, y, x, dropna=True, b_return_mean=False,groupby=None):
        return self.sample.getyx(y,x, dropna=dropna, b_return_mean=b_return_mean,groupby=groupby)


if __name__ == '__main__':
    files = '../../resources/pm25.csv'
    sampler = DBEstSampling()
    # sampler.sample1()
    sampler.make_sample(file=files, ratio=10)
    # print(sampler.sample)


