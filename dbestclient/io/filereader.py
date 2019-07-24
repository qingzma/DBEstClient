# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

import pandas as pd
import numpy as np


class CsvReader:
    def __init__(self):
        self.df = None
        self.n_row = None
        self.header = None

    def read(self, file, y, x, dropna=True):
        """
            read a csv file, and return y, x pairs.
        :param file: the file path
        :param y: header of y
        :param x: header of x
        :param dropna: boolean, whether drop Nan values.
        :return: y,x pairs
        """
        self.df = pd.read_csv(file)
        self.n_row = self.df.shape[0]
        self.header = self.df.columns

        if dropna:
            self.df = self.df.dropna(subset=[y, x])

        return self.df[y].values, self.df[x].values.reshape(-1,1), self.n_row



if __name__ == "__main__":
    reader = CsvReader()
    print(reader.read("../../resources/pm25.csv", y='pm25', x='PRES', dropna=True))

