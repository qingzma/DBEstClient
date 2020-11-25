import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing

# https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c


class SkipGram:
    def __init__(self):
        self.embedding = None
        self.dim = None
        self.usecols = None

    def fit(
        self,
        gb_data,
        equal_data,
        range_data,
        label_data,
        usecols,
        dim=20,
        window=1,
        min_count=0,
        negative=20,
        iter=30,
        workers=-1,
        NG=1,
    ):
        self.usecols = usecols
        print(gb_data)
