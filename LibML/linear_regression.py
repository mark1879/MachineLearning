import numpy as np
import numpy as np
from .metrics import r2_score


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None

    def fit_gd(self, x_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train"

        def J(theta, x_b, y):
            try:
                return np.sum((y - x_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, x_b, y):
            # 对 J 求导
            res = np.empty(len(theta))
            res[0] = np.sum(x_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (x_b.dot(theta) - y).dot(x_b[:, i])
            return res * 2 / len(x_b)

        def gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(theta, x_b, y) - J(last_theta, x_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self.theta_ = gradient_descent(x_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    def predict(self, x_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert x_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return x_b.dot(self.theta_)

    def score(self, x_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
