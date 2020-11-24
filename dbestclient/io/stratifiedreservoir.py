# Created by Qingzhi Ma at 2020-11-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk


import subprocess
import os
# from glob import glob
from datetime import datetime
from random import randint, shuffle
import dill
import numpy as np


class StratifiedReservoir:
    """Create a stratified reservoir sampling.
    """

    def __init__(self, file_name, file_header=None, n_jobs=1, capacity: int = 1000):
        self.header = None
        self.file_name = file_name
        self.file_header = file_header
        # self.relevant_header = None
        self.relevant_header_idx = []
        # self.ft_sample = {}
        self.ft_table = {}
        self.data_categoricals = {}
        self.data_features = {}
        self.data_labels = {}
        self.sample = {}
        self.n_jobs = n_jobs
        self.capacity = capacity
        self.gb_cols = None
        self.equality_cols = None
        self.feature_cols = None
        self.label_cols = None

        # self.b_skip_first_row = None
        if n_jobs == 1 and file_header is not None:
            self.b_skip_first_row = False
        else:
            self.b_skip_first_row = True

    def make_sample_for_sql_condition(self, usecols, split_char=',', b_shuffle=False, b_fast=False, b_return_sample=False):
        gb_cols = usecols["gb"]
        equality_cols = usecols["x_categorical"]
        feature_cols = usecols["x_continous"]
        label_cols = usecols["y"]
        return self.make_sample(gb_cols, equality_cols, feature_cols, [label_cols[0]], split_char, b_shuffle, b_fast, b_return_sample)

    def make_sample(self, gb_cols: list, equality_cols: list, feature_cols: list, label_cols: list, split_char=',', b_shuffle=False, b_fast=False, b_return_sample=False):
        print("Start making a sample as requested...")
        t1 = datetime.now()
        # pre-process the file header
        if self.file_header is None:
            with open(self.file_name, 'r') as f:
                file_header_from_file = f.readline().replace("\n", '')
                headers = file_header_from_file.split(split_char)
        elif isinstance(self.file_header, list):
            headers = self.file_header
        else:
            print("self.file_header", self.file_header)
            headers = self.file_header.split(split_char)
        print("headers", headers, "-"*200)
        print("gb_cols",  gb_cols)
        print("equality_cols",  equality_cols)
        print("feature_cols",  feature_cols)
        print("label_cols", label_cols)

        self.gb_cols = gb_cols
        self.equality_cols = equality_cols
        self.feature_cols = feature_cols
        self.label_cols = label_cols

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

        if self.n_jobs == 1:
            cnt = 0
            with open(self.file_name, 'r') as f:
                for line in f:
                    if self.b_skip_first_row:  # self.file_header is None:
                        # self.file_header = file_header
                        self.b_skip_first_row = False
                        continue
                    cnt += 1
                    if cnt % 1000000 == 0:
                        print(f'processed {cnt/1000000:5.0f} million records.')
                    splits = line.split(split_char)
                    splits[len(splits)-1] = splits[len(splits) -
                                                   1].replace("\n", '')
                    categoricals = [splits[i] for i in categorical_cols_idx]
                    if feature_cols is not None:
                        features = [splits[i] for i in feature_cols_idx]
                    labels = [splits[i] for i in label_cols_idx]
                    key = list2key(categoricals)
                    if key in self.sample:
                        self.ft_table[key] += 1
                        if len(self.sample[key]) < self.capacity:
                            self.sample[key].append(
                                categoricals+features+labels if feature_cols is not None else categoricals+labels)
                        else:
                            if not b_fast:  # naive simple algorithm
                                j = randint(0, cnt)
                                if j < self.capacity:
                                    self.sample[key][j] = categoricals + features + \
                                        labels if feature_cols is not None else categoricals+labels
                            else:  # fast algorithm
                                # need to provide an extra ft to store the frequency.
                                pass

                    else:
                        self.ft_table[key] = 1
                        self.sample[key] = [
                            categoricals+features+labels] if feature_cols is not None else [categoricals+labels]

            if self.file_header is None:
                self.file_header = file_header_from_file
            if b_return_sample:
                return self.sample, self.ft_table

        else:
            from multiprocessing import Pool as PoolCPU
            print("parallel sampling is only available in linux.")
            line_num = int(subprocess.check_output(
                ['wc', '-l', self.file_name]).split()[0])
            # print(line_num)
            n_per_file = int(line_num/self.n_jobs)+1

            split_command = f"split -l {n_per_file} --numeric-suffixes {self.file_name} tmp_split_"
            process = subprocess.Popen(
                split_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            files = []
            for i in range(self.n_jobs):
                files.append("tmp_split_"+str(i) if i >=
                             10 else "tmp_split_0"+str(i))

            pool = PoolCPU(processes=self.n_jobs)
            results = []
            fts = []
            instances = []
            for idx, file in enumerate(files):
                # if idx == 0 and self.file_header is None:
                #     i = pool.apply_async(
                #         StratifiedReservoir(file, file_header=file_header_from_file, n_jobs=1, capacity=self.capacity).make_sample, (gb_cols, equality_cols, feature_cols, label_cols, split_char, False, b_fast, True))
                if idx == 0:
                    i = pool.apply_async(
                        StratifiedReservoir(file, file_header=self.file_header, n_jobs=1, capacity=self.capacity).make_sample, (gb_cols, equality_cols, feature_cols, label_cols, split_char, False, b_fast, True))
                else:
                    fh = self.file_header if self.file_header is not None else file_header_from_file
                    i = pool.apply_async(
                        StratifiedReservoir(file, file_header=fh, n_jobs=1, capacity=self.capacity).make_sample, (gb_cols, equality_cols, feature_cols, label_cols, split_char, False, b_fast, True))
                instances.append(i)

            for i in instances:
                result, ft = i.get()
                results.append(result)
                fts.append(ft)
                # print("--"*20)
                # print(result)

            self.sample = results[0]
            self.ft_table = fts[0]
            for result, ft in zip(results[1:], fts[1:]):
                for key in result:
                    if key not in self.sample:
                        self.sample[key] = result[key]
                        self.ft_table[key] = ft[key]
                    elif self.ft_table[key] + ft[key] <= self.capacity:
                        self.sample[key] = self.sample[key]+result[key]
                        self.ft_table[key] += ft[key]
                    else:
                        new_result = self.sample[key]+result[key]
                        shuffle(new_result)
                        self.sample[key] = new_result[:self.capacity]
                        self.ft_table[key] += ft[key]

            # delete temporary files
            for file in files:
                os.remove(file)
            # delete temporary files
            # for file in glob('./tmp_split_*'):
            #     os.remove(file)

        # # prepare the frequency table
        # for key in self.sample:
        #     self.ft[key] = len(self.sample[key])

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
                categorical_cols_idx))] for row in self.sample[key]]
            if feature_cols is not None:
                self.data_features[key] = [[row[idx] for idx in range(len(categorical_cols_idx), len(
                    categorical_cols_idx)+len(feature_cols_idx))] for row in self.sample[key]]
                self.data_labels[key] = [row[idx] for idx in range(len(categorical_cols_idx)+len(feature_cols_idx),
                                                                   len(categorical_cols_idx)+len(feature_cols_idx) + len(label_cols_idx)) for row in self.sample[key]]
            else:
                self.data_labels[key] = [row[idx] for idx in range(len(categorical_cols_idx),
                                                                   len(categorical_cols_idx) + len(label_cols_idx)) for row in self.sample[key]]

        # cntt = 0
        # for key in self.ft_table:
        #     print(key, self.ft_table[key])
        #     cntt += self.ft_table[key]
        # print("cntt", cntt)

        # for key in self.ft_table:
        #     # print(key, self.ft[key])
        #     print(key, self.sample[key])
        #     print(key, self.data_categoricals[key])
        #     if self.data_features:
        #         print(key, self.data_features[key])
        #     print(key, self.data_labels[key])
        #     # exit()

        # self.sample = None
        print(
            f"Finish making the sample, time cost is {(datetime.now()-t1).total_seconds():.4f} seconds.")
        return self.data_categoricals, self.data_features, self.data_labels

    def get_categorical_features_label(self):
        return np.array(self.data_categoricals), self.data_features, self.data_labels

    def get_ft(self):
        return self.ft_table

    def size(self):
        cnt = 0
        for key in self.ft_table:
            cnt += self.ft_table[key]
        return cnt

    def erase(self):
        self.data_categoricals = None
        self.data_features = None
        self.data_labels = None

    def serialize2file(self, file):
        with open(file, 'wb') as f:
            dill.dump(self, f)


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
