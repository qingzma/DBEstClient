# Created by Qingzhi Ma at 2020-11-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

from datetime import datetime
from random import randint, shuffle


class StratifiedReservoir:
    """Create a stratified reservoir sampling.
    """

    def __init__(self, file_name, file_header=None, n_jobs=1, capacity: int = 1000):
        self.header = None
        self.file_name = file_name
        self.file_header = file_header
        self.relevant_header = None
        self.relevant_header_idx = []
        self.ft = {}
        self.data_categoricals = {}
        self.data_features = {}
        self.data_labels = {}
        self.sample = {}
        self.n_jobs = n_jobs
        self.capacity = capacity

    def make_sample(self, gb_cols: list, equality_cols: list, feature_cols: list, label_cols: list, split_char=',', b_shuffle=False):
        print("Start making a sample as requested...")
        t1 = datetime.now()
        # pre-process the file header
        if self.file_header is None:
            with open(self.file_name, 'r') as f:
                self.file_header = f.readline()
        headers = self.file_header.split(split_char)

        categorical_cols = gb_cols+equality_cols if equality_cols is not None else gb_cols
        categorical_cols_idx = [headers.index(item)
                                for item in categorical_cols]
        if feature_cols is not None:
            feature_cols_idx = [headers.index(item) for item in feature_cols]
        label_cols_idx = [headers.index(item) for item in label_cols]

        if feature_cols is not None:
            self.relevant_header_idx = categorical_cols_idx+feature_cols_idx+label_cols_idx
        else:
            self.relevant_header_idx = categorical_cols_idx+label_cols_idx

        # print(gb_cols_idx)
        # print(extra_cols_idx)
        # print(self.relevant_header_idx)

        # self.relevant_header = groupby_cols + extra_cols
        if self.n_jobs == 1:
            cnt = 0
            with open(self.file_name, 'r') as f:
                for line in f:
                    cnt += 1
                    splits = line.split(split_char)
                    categoricals = [splits[i] for i in categorical_cols_idx]
                    if feature_cols is not None:
                        features = [splits[i] for i in feature_cols_idx]
                    labels = [splits[i] for i in label_cols_idx]
                    key = list2key(categoricals)
                    if key in self.sample:
                        if len(self.sample[key]) < self.capacity:
                            self.sample[key].append(
                                categoricals+features+labels if feature_cols is not None else categoricals+labels)
                        else:
                            j = randint(0, cnt)
                            if j < self.capacity:
                                self.sample[key][j] = categoricals + features + \
                                    labels if feature_cols is not None else categoricals+labels
                    else:
                        self.sample[key] = [
                            categoricals+features+labels] if feature_cols is not None else [categoricals+labels]

        else:
            pass

        # prepare the frequency table
        for key in self.sample:
            self.ft[key] = len(self.sample[key])

        # shuffle
        if b_shuffle:
            for key in self.sample:
                shuffle(self.sample[key])

        # post-process the sample into 3 parts: categoricals, features and labels.
        self.data_categoricals = {}
        self.data_features = {}
        self.data_labels = {}
        for key in self.sample:
            self.data_categoricals[key] = [[row[idx] for idx in range(0, len(
                categoricals))] for row in self.sample[key]]
            if feature_cols is not None:
                self.data_features[key] = [[row[idx] for idx in range(len(categoricals), len(
                    categoricals)+len(features))] for row in self.sample[key]]
                self.data_labels[key] = [row[idx] for idx in range(len(categoricals)+len(features),
                                                                   len(categoricals)+len(features) + len(labels)) for row in self.sample[key]]
            else:
                self.data_labels[key] = [row[idx] for idx in range(len(categoricals),
                                                                   len(categoricals) + len(labels)) for row in self.sample[key]]
        self.sample = None
        # for key in self.ft:
        #     print(key, self.ft[key])

        # for key in self.ft:
        #     # print(key, self.ft[key])
        #     print(key, self.sample[key])
        #     print(key, data_categoricals[key])
        #     print(key, data_features[key])
        #     print(key, data_labels[key])
        #     exit()

        print(
            f"Finish making the sample, time cost is {(datetime.now()-t1).total_seconds():.4f} seconds.")

    def get_categorical_features_label(self):
        return self.data_categoricals, self.data_features, self.data_labels

    def erase(self):
        self.data_categoricals = None
        self.data_features = None
        self.data_labels = None


class Reservoir:
    def __init__(self, capacity=1000):
        self.ft = None
        self.sample = {}  # the key is groupby_cols + extra_cols
        self.capacity = capacity

    def update_ft(self, key: list):
        pass

    def update_samples(self, key: list, row: str):
        pass


def list2key(lst: list) -> str:
    return ','.join(lst)
