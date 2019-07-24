# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.io import filereader


class GetXY:
    def __init__(self, backend='None'):
        self.backend = backend
        self.freader = None

    def read(self, file, y, x):
        if self.backend == 'None':
            self.freader = filereader.CsvReader()
            return self.freader.read(file, y, x)[0:2]
        else:
            print("Other backend server is not currently supported.")
            return 9999,9999

    def get_n_points(self):
        return self.freader.n_row





if __name__ == "__main__":
    reader = GetXY(backend=None)
    print(reader.read("../../resources/pm25.csv", y='pm25', x='PRES'))