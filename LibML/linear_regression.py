import numpy as np
from .metrics import r2_score


class LinearRegression:
    """多元线性回归"""

    def __init__(self):
        self.coef = None
        self.intercept = None
        self._theta = None

    def fit_normal(self, x_train, y_train):
        """计算公式：向量化实现，最小二乘法求导。公式求解较复杂，算法复杂度高-O(n^3)"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.intercept = self._theta[0]
        self.coef = self._theta[1:]

        return self

    def fit_gd(self, x_train, y_train, eta=0.01, n_iters=1e4):
        """批量梯度下降法训练 LinearRegression 模型"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equals to the size of y_train"

        def _J(theta, x_b, y):
            try:
                return np.sum((y - x_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        # def _dJ(theta, x_b, y):
        #     res = np.empty(len(theta))
        #     res[0] = np.sum(x_b.dot(theta) - y)
        #     for i in range(1, len(theta)):
        #         res[i] = (x_b.dot(theta) - y).dot(x_b[:, i])
        #
        #     return res * 2 / len(x_b)

        def _dJ(theta, x_b, y):
            return x_b.T.dot(x_b.dot(theta) - y) * 2. / len(y)

        def _gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = _dJ(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(_J(theta, x_b, y) - _J(last_theta, x_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])

        self._theta = _gradient_descent(x_b, y_train, initial_theta, eta, n_iters)
        self.intercept = self._theta[0]
        self.coef = self._theta[1:]

        return self

    def fit_sgd(self, x_train, y_train, n_iters=50, t0=5, t1=50):
        """随机梯度下降法训练 LinearRegression 模型"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, x_b_i, y_i):
            return x_b_i * (x_b_i.dot(theta) - y_i) * 2.

        def sgd(x_b, y, initial_theta, n_iters, t0=5, t1=50):
            def learning_rate(t):
                return t0 / (t + t1)

            # 简易实现
            # theta = initial_theta
            # for cur_iter in range(n_iters):
            #     rand_i = np.random.randint(len(x_b))
            #     gradient = dJ_sgd(theta, x_b[rand_i], y[rand_i])
            #     theta = theta - learning_rate(cur_iter) * gradient
            #
            # return theta

            theta = initial_theta
            m = len(x_b)
            for i_iter in range(n_iters):
                indexes = np.random.permutation(m)
                x_b_new = x_b[indexes, :]
                y_b_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, x_b_new[i], y_b_new[i])
                    theta = theta - learning_rate(i_iter * m + i) * gradient

            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = sgd(x_b, y_train, initial_theta, n_iters, t0, t1)
        self.intercept = self._theta[0]
        self.coef = self._theta[1:]


    def predict(self, x_predict):
        assert self.intercept is not None and self.coef is not None, \
            "must fit before predict!"
        assert x_predict.shape[1] == len(self.coef), \
            "the feature number of x_predict must be equal to x_train"

        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])

        return x_b.dot(self._theta)

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
