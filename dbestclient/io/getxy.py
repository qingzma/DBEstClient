# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.io import filereader


class GetXY:
    def __init__(self, backend='None'):
        self.backend = backend

    def read(self, file, y, x):
        if self.backend == 'None':
            return filereader.CsvReader().read(file, y, x)
        else:
            print("Other backend server is not currently supported.")
            return 9999,9999



if __name__ == "__main__":
    reader = GetXY(backend=None)
    print(reader.read("../../resources/pm25.csv", y='pm25', x='PRES'))