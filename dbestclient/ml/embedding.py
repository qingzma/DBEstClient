#
# Created by Qingzhi Ma on Fri Jun 05 2020
#
# Copyright (c) 2020 Department of Computer Science, University of Warwick
# Copyright 2020 Qingzhi Ma
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import multiprocessing
import os

import pandas as pd

from gensim.models import Word2Vec


class WordEmbedding:
    def __init__(self):
        self.embedding = None

    def describing(self, Indata):
        describe = {}
        for j in range(0, len(Indata.columns.values)):
            valueList = []
            valueList.append(len(Indata.iloc[:, j].unique()))
            valueList.append(
                round(len(Indata.iloc[:, j].unique())/Indata.shape[0], 3))
            describe[Indata.columns.values[j]] = valueList
        # print(describe)
        return describe

    def fit(self, file_address, rate, group_by_column, EMB_DIM, n_iteration):
        group_by_column = group_by_column
        rate = rate
        data = pd.read_csv(file_address, sep='|', header=0)
        IndexOfGroupBy = 0

        for i in range(0, len(data.columns.values)):
            if data.columns.values[i] == group_by_column:
                IndexOfGroupBy = i
        print(str(IndexOfGroupBy)+"  "+data.columns.values[IndexOfGroupBy])

        des = self.describing(data)

        sentences = []
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]):
                if (j != IndexOfGroupBy) and (float(des[data.columns.values[j]][1]) < rate) and (math.isnan(data.iloc[i, IndexOfGroupBy]) == False) and (math.isnan(data.iloc[i, j]) == False):
                    temp = []
                    temp.append(str(
                        data.columns.values[IndexOfGroupBy])+" "+str(int(data.iloc[i, IndexOfGroupBy])))
                    temp.append(
                        str(data.columns.values[j])+" "+str(data.iloc[i, j]))
                    sentences.append(temp)
        EMB_DIM = EMB_DIM  # number of dimension
        print("the embedding process has been started")

        w2v = Word2Vec(sentences, size=EMB_DIM, window=1, min_count=1,
                       negative=20, iter=n_iteration, workers=multiprocessing.cpu_count())
        word_vectors = w2v.wv  # Matix of model
        vocab = w2v.wv.vocab   # Vocabulary

        # print(word_vectors)
        count = 0
        Group = {}
        for each in vocab:
            if group_by_column in each:
                Group[each.split(" ")[1].split(".")[0]
                      ] = word_vectors.vectors[count]
            count = count+1
        print("finish")
        self.embedding = Group
        return Group

    def predict(self, key):
        return(self.embedding[key])

    def predicts(self, keys):
        results = []
        for key in keys:
            results.append(self.embedding[key])
        return results


if __name__ == "__main__":
    word_embedding = WordEmbedding()
    # file_name, Max_rate,groupby_column,Embedding_dimension_size,number_of_iteration
    word_embedding.fit("/data/tpcds/40G/ss_600k_headers.csv",
                       0.2, "ss_store_sk", 20, 1)
