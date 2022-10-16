import numpy as np
from .metrics import r2_score


class LinearRegression:
    """多元线性回归"""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None

    def fit_normal(self, x_train, y_train):
        """计算公式：向量化实现，最小二乘法求导。公式求解较复杂，算法复杂度高-O(3)"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self.theta_ = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    def predict(self, x_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert x_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])

        return x_b.dot(self.theta_)

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
