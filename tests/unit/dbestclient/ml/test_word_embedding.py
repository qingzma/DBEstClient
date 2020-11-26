import unittest

import numpy as np
from dbestclient.ml.wordembedding import SkipGram


class TestSkipGram(unittest.TestCase):
    def test_reg(self):
        gb = np.array([["london", "male"], ["paris", "female"], ["", ""]])
        # equals = np.array([["flat", "50mile"], ["apartment", "60mile"], ["", ""]])
        ranges = np.array([20, 30, 40])
        labels = np.array([15000.032, 16000.80, 18000])
        usecols = {
            "y": ["salary"],
            "x_continous": ["age"],
            "x_categorical": [],  # ["accomodation", "distance"],
            "gb": ["city", "gender"],
        }
        sg = SkipGram().fit(gb, ranges, labels, usecols, workers=1)
        gbs = np.array(
            [
                ["london", "male", "apartment", "60mile"],
                ["paris", "female", "apartment", "60mile"],
                ["", "female", "apartment", "60mile"],
            ]
        )

        gbs = np.array(
            [
                [
                    "london",
                    "male",
                ],
                [
                    "paris",
                    "female",
                ],
                [
                    "",
                    "female",
                ],
            ]
        )

        results = sg.predicts(gbs)
        # print("results")
        # print(results)

    def test_reg_no_cols(self):
        gb = np.array([["london", "male"], ["paris", "female"], ["", ""]])
        # equals = np.array([["flat", "50mile"], ["apartment", "60mile"], ["", ""]])
        ranges = np.array([20, 30, 40])
        labels = np.array([15000.032, 16000.80, 18000])
        usecols = {
            "y": ["salary"],
            "x_continous": ["age"],
            "x_categorical": [],  # ["accomodation", "distance"],
            "gb": ["city", "gender"],
        }
        sg = SkipGram().fit(gb, ranges, labels, usecols=None, workers=1)
        gbs = np.array(
            [
                ["london", "male", "apartment", "60mile"],
                ["paris", "female", "apartment", "60mile"],
                ["", "female", "apartment", "60mile"],
            ]
        )

        gbs = np.array(
            [
                [
                    "london",
                    "male",
                ],
                [
                    "paris",
                    "female",
                ],
                [
                    "",
                    "female",
                ],
            ]
        )

        results = sg.predicts(gbs)
        # print("results")
        # print(results)

    def test_kde(self):
        gb = np.array([["london", "male"], ["paris", "female"], ["", ""]])
        # equals = np.array([["flat", "50mile"], ["apartment", "60mile"], ["", ""]])
        ranges = np.array([20, 30, 40])
        labels = np.array([15000.032, 16000.80, 18000])
        usecols = {
            "y": ["salary"],
            "x_continous": ["age"],
            "x_categorical": [],  # ["accomodation", "distance"],
            "gb": ["city", "gender"],
        }
        sg = SkipGram().fit(gb, ranges, None, usecols, workers=1, b_reg=False)
        gbs = np.array(
            [
                ["london", "male", "apartment", "60mile"],
                ["paris", "female", "apartment", "60mile"],
                ["", "female", "apartment", "60mile"],
            ]
        )

        gbs = np.array(
            [
                [
                    "london",
                    "male",
                ],
                [
                    "paris",
                    "female",
                ],
                [
                    "",
                    "female",
                ],
            ]
        )

        results = sg.predicts(gbs)
        # print("results")
        print(results)

    def test_kde_no_cols(self):
        gb = np.array([["london", "male"], ["paris", "female"], ["", ""]])
        # equals = np.array([["flat", "50mile"], ["apartment", "60mile"], ["", ""]])
        ranges = np.array([20, 30, 40])
        labels = np.array([15000.032, 16000.80, 18000])
        usecols = {
            "y": ["salary"],
            "x_continous": ["age"],
            "x_categorical": [],  # ["accomodation", "distance"],
            "gb": ["city", "gender"],
        }
        sg = SkipGram().fit(gb, ranges, None, usecols=None, workers=1, b_reg=False)
        gbs = np.array(
            [
                ["london", "male", "apartment", "60mile"],
                ["paris", "female", "apartment", "60mile"],
                ["", "female", "apartment", "60mile"],
            ]
        )

        gbs = np.array(
            [
                [
                    "london",
                    "male",
                ],
                [
                    "paris",
                    "female",
                ],
                [
                    "",
                    "female",
                ],
            ]
        )

        results = sg.predicts(gbs)
        # print("results")
        # print(results)


if __name__ == "__main__":
    unittest.main()
    # TestSkipGram().test_kde_no_cols()
