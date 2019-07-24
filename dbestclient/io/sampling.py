# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.io.reservoir import ReservoirSampling

class DBEstSampling:
    def __init__(self):
        self.n_sample_point = None
        self.n_total_point = None
        self.sample = None

    def make_sample(self, file, ratio,  method='uniform', split_char=','):
        if method == 'uniform':
            if float(ratio) > 1: # here the ratio is the number of tuples in the sample
                ratio = int(ratio)
                self.n_sample_point = ratio
                self.sample = ReservoirSampling()
                self.sample.build_reservoir(file,ratio,split_char=split_char)
                self.n_total_point =  self.sample.n_total_point

                return self.sample
            else:
                print("sampling with probability is not implemented yet, abort")
        else:
            print("other sampling methods are not implemented, abort.")

    def getyx(self, y, x, dropna=True):
        return self.sample.getyx(y,x, dropna=dropna)


if __name__ == '__main__':
    files = '../../resources/pm25.csv'
    sampler = DBEstSampling()
    # sampler.sample1()
    sampler.make_sample(file=files, ratio=10)
    # print(sampler.sample)


