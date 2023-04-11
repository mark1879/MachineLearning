#! /usr/bin/python
# encoding: utf-8

import numpy as np


def train_test_split(x, y, test_ratio=0.2, seed=None):
    assert x.shape[0] == y.shape[0], "the size of x must equals to the size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_ratio must be valid, 0.0 <= test_ratio <= 1.0"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(x))

    test_size = int(len(x) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    x_train = x[train_indexes]
    y_train = y[train_indexes]

    x_test = x[test_indexes]
    y_test = y[test_indexes]

    return x_train, x_test, y_train, y_test
