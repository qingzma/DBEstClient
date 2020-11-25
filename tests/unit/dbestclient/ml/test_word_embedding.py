import unittest

import numpy as np
from dbestclient.ml.wordembedding import SkipGram


class TestSkipGram(unittest.TestCase):
    def test_1(self):
        gb = np.array([["london", "male"], ["paris", "female"]])
        equals = np.array([["flat", "50mile"], ["apartment", "60mile"]])
        ranges = np.array([[20], [30]])
        labels = np.array([15000, 16000])
        usecols = {
            "y": "salary",
            "x_continous": ["age"],
            "x_categorical": ["accomodation", "distance"],
            "gb": ["city", "gender"],
        }
        SkipGram().fit(gb, equals, ranges, labels, usecols)


if __name__ == "__main__":
    unittest.main()
