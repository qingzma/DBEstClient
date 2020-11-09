
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


NumberOFAtrributes=0

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

    def fit(self, sentences, gbs, dim=20 ,window=1, min_count=1,negative=20,iter=30,workers=30):
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

        #if len(gbs)>1:
        #    raise TypeError("Embedding only supports one GROUP BY attribute at this moment, use use binay or onehot encoding instead.")
        if len(gbs)>1:
            w2v = Word2Vec(sentences, size=int(dim/2), window=1, min_count=1,
                       negative=10, iter=iter, workers=multiprocessing.cpu_count(),ns_exponent=0.2)#,ns_exponent=0.0
        else:
            w2v = Word2Vec(sentences, size=dim, window=1, min_count=1,
                       negative=30, iter=iter, workers=multiprocessing.cpu_count())#,ns_exponent=0.0
        word_vectors = w2v.wv  # Matix of model
        vocab = w2v.wv.vocab   # Vocabulary
        self.dim = dim

        #print("vocab", vocab)
        # print(word_vectors)
        count = 0
        Group = {}
        #group_by_column =  gbs[0]
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        count=0
        for each in vocab:
            if "gb" in each:
                Group[each.split(" ")[1]]=word_vectors.vectors[count]
            count=count+1
        #print(list(Group.keys())[0:10])
        #print(list(vocab.keys())[0:10])
        
            
        #    if group_by_column in each:
        #        Group[each.split(" ")[1]] = word_vectors.vectors[count]
        #        #Group[each.split(" ")[1].split(".")[0]
        #                #] = word_vectors.vectors[count]
        #    count = count+1
        # print("finish")
        del(word_vectors) ############################
        del(vocab)        ############################
        self.embedding = Group
        #print("embedding", self.embedding.keys(), "len(Group.keys()):", len(Group.keys()))
        
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
        #print("keys ",keys[0:10])
		
        results = []
        #a=set(keys)
        #b=set(self.embedding.keys())
        #c=a-b
        #print("difference is: ",len(c))
		
        #print("len(key) ",len(keys)," , ", len(keys[0])," key ", keys[0])
        for key in keys:
            if len(key[0])>=2:
                ttt=list(self.embedding[key[0]])
            #print(key," len(key) ",len(key))
            #print("ttt ",ttt, " type ", type(ttt))
			
                for i in range(1,len(key)):
                    ttt1=list(self.embedding[key[i]])
                    #print("ttt1 :",ttt1, type(ttt1))
                    ttt=ttt+ttt1
                
                #print("Current ",ttt," len(ttt) ",len(ttt))
                #print("t "+t)
                results.append(ttt)
                
            else:
                #results.extend(self.embedding[key])
                #print ("else")
                #print("keys")
                ttt=list(self.embedding[key[0]])
                #print(key," len(key) ",len(key))
                #print("ttt ",ttt, " type ", type(ttt))
                for i in range(1,len(key)):
                    ttt1=list(self.embedding[key[i]])
                    #print("ttt1 :",ttt1, type(ttt1))
                    ttt=ttt+ttt1
                
                #print("Current ",ttt," len(ttt) ",len(ttt))
                #print("t "+t)
                results.append(ttt)
        #print(results)
        results = np.reshape(results, (-1, self.dim))
		
        print("end embedding inference")
        #print(results)
        return results

def dataframe2sentences(df:pd.DataFrame, gbs:list):
    headers = df.columns#.to_list()
    sentences = []
    no_gbs = list(set(headers)-set(gbs))
    # print("no_gbs",no_gbs)
    # print("gbs",gbs)
    for row in df.itertuples():
        #front_words = []
        CCC=0
        for gb in gbs:
            while (CCC<len(str(getattr(row, gb)).split(","))):
                front_words = []
                #tt=str(getattr(row, gb))
                #ngb=tt.split(",")
                # print("gb",gb)
                # print("row",row)
                # print("column",getattr(row, gb))
                # front_words = front_words + gb
                front_words.append(gb+ " "+ str(getattr(row, gb)).split(",")[CCC])
                CCC=CCC+1
                #print('front_words',front_words)
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
    #print(" sentences ",sentences[0:10])
    return sentences

def columns2sentences(gbs_data, xs_data, ys_data=None):
    # print("gbs_data",gbs_data)
    # print("xs_data",xs_data)
    # print("ys_data",ys_data)
    #if len(gbs_data[0])>1:
    #    raise TypeError("Embedding only supports one GROUP BY attribute at this moment, use use binay or onehot encoding instead.")
    # cols_gb=["cols_gb"]
    # cols_x = ["cols_x0"]
    # cols_y = ["cols_y"] if ys_data is not None else []
    #print("gbs_data", gbs_data)
    NumberOFAtrributes=len(gbs_data[0])
    #gbs_data = gbs_data.reshape(1,-1)[0]
    #print("gbs_data", gbs_data[0:10])
	
    #cn=0
    #for each_e in gbs_data:
    #    if each_e=='2776.68':
    #        cn=cn+1
            #print ("@@yesy ther is@@",each_e)
    # print("gbs_data",gbs_data)
    #print("after compresing ",cn)
    #print(similar("Apple","Appel"))
    new_gbs_data=[]
    #print("NumberOFAtrributes ",NumberOFAtrributes)
    #print(len(gbs_data))
    #print(len(gbs_data[0]))
    for k in range(0, len(gbs_data)):
        temp=""
        for i in range(0,NumberOFAtrributes):
            temp=temp+gbs_data[k][i]+","
        #print(temp)
        new_gbs_data.append(temp[:-1])
        #print(new_gbs_data)
    gbs_data=new_gbs_data
    if ys_data is None:
        df = pd.DataFrame({"gb":gbs_data, "x":xs_data})
    else:
        #print("gbs_data", gbs_data)
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
    #print(word_embedding.predict('92'))
    #print("*"*20)
    t1= datetime.now()
    #print(word_embedding.predicts(['92','70','4']))
    t2 = datetime.now()
    print("time cost is ", (t2-t1).total_seconds())

    t1= datetime.now()
    print(word_embedding.predicts_low_efficient(['92','70','4']))
    t2 = datetime.now()
    print("time cost is ", (t2-t1).total_seconds())
    # # file_name, Max_rate,groupby_column,Embedding_dimension_size,number_of_iteration
    # word_embedding.fit("/data/tpcds/40G/ss_600k_headers.csv",
    #                    0.2, "ss_store_sk", 20, 1)
