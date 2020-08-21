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
import numpy as np

from gensim.models import Word2Vec


class WordEmbedding:
    def __init__(self):
        self.embedding = None
        self.dim = None
        print("start training embedding")

    # def describing(self, Indata):
        # describe = {}
        # for j in range(0, len(Indata.columns.values)):
        #     valueList = []
        #     valueList.append(len(Indata.iloc[:, j].unique()))
        #     valueList.append(
        #         round(len(Indata.iloc[:, j].unique())/Indata.shape[0], 3))
        #     describe[Indata.columns.values[j]] = valueList
        # # print(describe)
        # return describe

    def fit(self, sentences, gbs, dim=20 ,window=1, min_count=1,negative=60,iter=50,workers=30):
        # group_by_column = group_by_column
        # rate = rate
        # data = pd.read_csv(file_address, sep='|', header=0)
        # IndexOfGroupBy = 0

        # for i in range(0, len(data.columns.values)):
        #     if data.columns.values[i] == group_by_column:
        #         IndexOfGroupBy = i
        # print(str(IndexOfGroupBy)+"  "+data.columns.values[IndexOfGroupBy])

        # des = self.describing(data)

        # sentences = []
        # for i in range(0, data.shape[0]):
        #     for j in range(0, data.shape[1]):
        #         if (j != IndexOfGroupBy) and (float(des[data.columns.values[j]][1]) < rate) and (math.isnan(data.iloc[i, IndexOfGroupBy]) == False) and (math.isnan(data.iloc[i, j]) == False):
        #             temp = []
        #             temp.append(str(
        #                 data.columns.values[IndexOfGroupBy])+" "+str(int(data.iloc[i, IndexOfGroupBy])))
        #             temp.append(
        #                 str(data.columns.values[j])+" "+str(data.iloc[i, j]))
        #             sentences.append(temp)
        # EMB_DIM = EMB_DIM  # number of dimension
        # print("the embedding process has been started")

        if len(gbs)>1:
            raise TypeError("Embedding only supports one GROUP BY attribute at this moment, use use binay or onehot encoding instead.")

        w2v = Word2Vec(sentences, size=dim, window=1, min_count=1,
                       negative=20, iter=iter, workers=multiprocessing.cpu_count())
        word_vectors = w2v.wv  # Matix of model
        vocab = w2v.wv.vocab   # Vocabulary
        self.dim = dim

        # print("vocab", vocab)
        # print(word_vectors)
        count = 0
        Group = {}
        group_by_column =  gbs[0]
        for each in vocab:
            if group_by_column in each:
                Group[each.split(" ")[1].split(".")[0]
                        ] = word_vectors.vectors[count]
            count = count+1
        # print("finish")
        self.embedding = Group
        # print("embedding", Group)

        print("finish training embedding")
        return Group
        

    def predict(self, key):
        return(self.embedding[key])

    def predicts_low_efficient(self, keys):
        results = np.array(self.embedding[keys[0]])
        # print(results)
        for key in keys[1:]:
            # print("results", results)
            # print("key", key, self.embedding[key])
            
            results = np.vstack((results, self.embedding[key]))
            
        return results
    
    def predicts(self, keys):
        print("start embedding inference")
        results = []
        # print(results)
        for key in keys:
            results.extend(self.embedding[key])
        results = np.reshape(results, (-1, self.dim))
        print("end embedding inference")
        return results

def dataframe2sentences(df:pd.DataFrame, gbs:list):
    headers = df.columns#.to_list()
    sentences = []
    no_gbs = list(set(headers)-set(gbs))
    # print("no_gbs",no_gbs)
    # print("gbs",gbs)
    for row in df.itertuples():
        front_words = []
        for gb in gbs:
            # print("gb",gb)
            # print("row",row)
            # print("column",getattr(row, gb))
            # front_words = front_words + gb
            front_words.append(gb+ " "+ str(getattr(row, gb)))
        # print('front_words',front_words)
        for no_gb in no_gbs:
            each_sentence = list(front_words)
            each_sentence.append(no_gb + " "+str(getattr(row, no_gb)))
            sentences.append(each_sentence)
            # sentences.append([front_words + no_gb + " "+str(getattr(row, no_gb))])
    # for row in df.itertuples():
    #     sentences.append([headers[0]+" "+ str(row[1]), headers[1]+" "+ str(row[2])])
    #     sentences.append([headers[0]+" "+ str(row[1]), headers[2]+" "+ str(row[3])])
    #     print(row[1],row[2],row[3])
    # print(headers)
    # print(sentences)
    return sentences

def columns2sentences(gbs_data, xs_data, ys_data=None):
    # print("gbs_data",gbs_data)
    # print("xs_data",xs_data)
    # print("ys_data",ys_data)
    if len(gbs_data[0])>1:
        raise TypeError("Embedding only supports one GROUP BY attribute at this moment, use use binay or onehot encoding instead.")
    # cols_gb=["cols_gb"]
    # cols_x = ["cols_x0"]
    # cols_y = ["cols_y"] if ys_data is not None else []
    gbs_data = gbs_data.reshape(1,-1)[0]
    # print("gbs_data",gbs_data)
    if ys_data is None:
        df = pd.DataFrame({"gb":gbs_data, "x":xs_data})
    else:
        df = pd.DataFrame({"gb":gbs_data, "x":xs_data,"y":ys_data})
    

    return dataframe2sentences(df, ["gb"])


if __name__ == "__main__":
    from datetime import datetime
    header=[
    "ss_sold_date_sk","ss_sold_time_sk","ss_item_sk","ss_customer_sk","ss_cdemo_sk","ss_hdemo_sk",
                                  "ss_addr_sk","ss_store_sk","ss_promo_sk","ss_ticket_number","ss_quantity","ss_wholesale_cost",
                                  "ss_list_price","ss_sales_price","ss_ext_discount_amt","ss_ext_sales_price",
                                  "ss_ext_wholesale_cost","ss_ext_list_price","ss_ext_tax","ss_coupon_amt","ss_net_paid",
                                  "ss_net_paid_inc_tax","ss_net_profit","none"]
    file = "/Users/scott/Documents/workspace/data/tpcds/10g/ss_10g_20.csv"
    df = pd.read_csv(file, sep="|", names=header, usecols=['ss_sales_price','ss_sold_date_sk','ss_store_sk'])
    # print(df)
    sentenses = dataframe2sentences(df,gbs=["ss_store_sk"])
    word_embedding = WordEmbedding()
    word_embedding.fit(sentenses, gbs=["ss_store_sk"])
    print(word_embedding.predict('92'))
    print("*"*20)
    t1= datetime.now()
    print(word_embedding.predicts(['92','70','4']))
    t2 = datetime.now()
    print("time cost is ", (t2-t1).total_seconds())

    t1= datetime.now()
    print(word_embedding.predicts_low_efficient(['92','70','4']))
    t2 = datetime.now()
    print("time cost is ", (t2-t1).total_seconds())
    # # file_name, Max_rate,groupby_column,Embedding_dimension_size,number_of_iteration
    # word_embedding.fit("/data/tpcds/40G/ss_600k_headers.csv",
    #                    0.2, "ss_store_sk", 20, 1)
