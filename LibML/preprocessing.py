import numpy as np


class StandardScaler:
    def __init__(self):
        self.scale_ = None
        self.mean_ = None

    def fit(self, x):
        """根据训练数据集 x 获得数据的均值和方差"""
        assert x.ndim == 2, "The dimension of x must be 2"

        self.mean_ = np.array([np.mean(x[:, i]) for i in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[:, i]) for i in range(x.shape[1])])

        return self

    def transform(self, x):
        """将 x 根据这个 StandardScaler 进行均值方差归一化处理"""
        assert x.ndim == 2, "The dimension of x must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform!"
        assert x.shape[1] == len(self.mean_) and x.shape[1] == len(self.scale_), \
            "the feature number of X must be equal to mean_ and std_"

        x_standard = np.empty(shape=x.shape, dtype=float)
        for col in range(x.shape[1]):
            x_standard[:, x] = (x[:, col] - self.mean_[col]) / self.scale_[col]

        return x_standard
