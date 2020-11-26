import multiprocessing

import numpy as np

# import pandas as pd
from gensim.models import Word2Vec

# from numpy.core.defchararray import startswith

# https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c


class SkipGram:
    def __init__(self):
        # self.embedding = None
        self.dim = None
        self.usecols = None
        self.embeddings = {}
        self.header_categorical = None

    def fit(
        self,
        # gb_data,
        # equal_data,
        categorical_data,
        range_data,
        label_data=None,
        usecols=None,
        dim=20,
        window=1,
        min_count=0,
        negative=30,
        iters=30,
        workers=-1,
        NG=1,
        b_reg=True,
    ):
        # categoricals = (
        #     np.concatenate((gb_data, equal_data), axis=1)
        #     if equal_data is not None
        #     else gb_data
        # )
        categoricals = (
            np.concatenate(
                (categorical_data, range_data[:, np.newaxis].astype(str)), axis=1
            )
            if range_data is not None
            else categorical_data
        )
        if b_reg:
            categoricals = (
                np.concatenate(
                    (categoricals, label_data[:, np.newaxis].astype(str)), axis=1
                )
                if label_data is not None
                else categorical_data
            )

        if usecols is not None:
            self.usecols = usecols
            # print("self.usecols", self.usecols)
            header = (
                [
                    usecols["gb"]
                    + usecols["x_categorical"]
                    + usecols["x_continous"]
                    + [usecols["y"][0]]
                ]
                if b_reg
                else [usecols["gb"] + usecols["x_categorical"] + usecols["x_continous"]]
            )
            self.header_categorical = (
                usecols["gb"] + usecols["x_categorical"]
                if usecols["x_categorical"] is not None
                else usecols["gb"]
            )
        else:
            tmp = [
                "aa_",
                "bb_",
                "cc_",
                "dd_",
                "ee_",
                "ff_",
                "gg_",
                "hh_",
                "ii_",
                "jj_",
                "kk_",
                "ll_",
                "mm_",
                "nn_",
            ]
            self.header_categorical = tmp[: len(categorical_data[0])]
            # print(categoricals, "---->>>>>")
            # print(categoricals[0])
            # print(len(categoricals[0]))
            # header = tmp[:3]
            header = [tmp[: len(categoricals[0])]]

        # print(categoricals)
        # print(header)
        headers = np.repeat(header, len(categoricals), axis=0)
        # print("*" * 70)
        # print(headers, "-" * 20, ">" * 20)
        sentences = np.core.defchararray.add(headers, categoricals).tolist()
        # print(sentences)

        workers = multiprocessing.cpu_count() if workers == -1 else 1
        model = Word2Vec(
            sentences,
            size=int(dim / NG),
            window=window,
            min_count=min_count,
            negative=negative,
            iter=iters,
            workers=workers,
        )

        # word_vectors = model.wv  # Matix of model
        vocab = model.wv.vocab  # Vocabulary
        self.dim = dim * len(self.header_categorical)
        # print("dim is", self.dim)
        # print(model["citylondon"])

        for word in vocab:
            # print("word", word)
            for head in self.header_categorical:
                # print("head", head)
                if word.startswith(head):
                    self.embeddings[word] = model.wv[word]
        # print(self.embeddings.keys())
        print("finish training embedding.")
        return self

    def predicts(self, keys):
        headers = np.repeat([self.header_categorical], len(keys), axis=0)
        # print(headers)
        sentences = np.core.defchararray.add(headers, keys)  # .tolist()
        # print(sentences)

        predictions = None
        for words in sentences:
            # print(words)
            prediction = np.array([])
            for word in words:
                # print(self.embeddings[word])
                prediction = np.append(prediction, self.embeddings[word])[
                    np.newaxis, :
                ]  # .tolist()
                # print("prediction")
                # print(prediction)
            predictions = (
                np.append(predictions, prediction, axis=0)
                if predictions is not None
                else prediction
            )
        return predictions
        # print(keys)
