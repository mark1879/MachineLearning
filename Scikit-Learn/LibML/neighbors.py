#! /usr/bin/python
# encoding: utf-8

import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


class KNeighborsClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self._k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train"
        assert self._k <= x_train.shape[0], \
            "the size of x_train must be at least k."

        self._x_train = x_train
        self._y_train = y_train

        return self

    def predict(self, x_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._x_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert x_predict.shape[1] == self._x_train.shape[1], \
            "the feature number of x_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in x_predict]

        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] == self._x_train.shape[1], \
            "the feature number of x must be equal to x_train"

        distances = [sqrt(np.sum((train - x) ** 2))
                     for train in self._x_train]
        nearest = np.argsort(distances)

        top_k_y = [self._y_train[i] for i in nearest[:self._k]]
        votes = Counter(top_k_y)

        return votes.most_common(1)[0][0]

    def score(self, x_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self._k
