#! /usr/bin/python
# encoding: utf-8

import numpy as np
from math import sqrt
from collections import Counter


class KNeighborsClassifier:

    def __init__(self, k):
        assert k >= 1, "k must be >= 1"
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train isn't equal to the size of y_train"
        assert self.k <= x_train.shape[0], \
            "the size of x_train must be at least k"

        self.x_train = x_train
        self.y_train = y_train

        return self

    def predict(self, x_predict):
        assert self.x_train is not None and self.y_train is not None, \
            "the model hasn't been fitted!"
        assert x_predict.shape[1] == self.x_train.shape[1], \
            "the feature's number of x_predict must be equal to x_train"

        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x_predict):
        assert x_predict.shape[0] == self.x_train.shape[1], \
            "the feature's number of x_predict must be equal to x_train"

        dist = [sqrt(np.sum((x_predict - x) ** 2))
                for x in self.x_train]

        nearest = np.argsort(dist)

        top_k_y = [self.y_train[i] for i in nearest[:self.k]]

        votes = Counter(top_k_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k
