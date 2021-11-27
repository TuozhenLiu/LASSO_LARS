import pandas as pd
import numpy as np
from src.LARS_pure import LARS_pure
from src.coef_calculate import coef_calculate
from src.path_plot import path_plot


class LARS(object):
    def __init__(self, x, y, fit_intercept=True, standarize=True):
        self.__y = y
        self.__n = np.shape(x)[0]
        if (fit_intercept == True) & (standarize == False):
            self.__x = np.c_[np.ones((self.__n, 1)), x]
        else:
            self.__x = x
        self.__p = np.shape(self.__x)[1]
        self.__xtx = np.dot(self.__x.T, self.__x)
        self.__xty = np.dot(self.__x.T, y)
        self.__fit_intercept = fit_intercept
        self.__standarize = standarize
        self.__x_mean = np.mean(self.__x, axis=0)
        self.__y_mean = np.mean(y)

    def get_path(self, plot=False):
        y_std = np.std(self.__y, ddof=1)
        self.alphas, self.coefs = LARS_pure(self.__xtx, self.__xty,
                                             self.__x_mean, self.__y_mean,
                                             y_std, self.__n, self.__p,
                                             self.__standarize,
                                             self.__fit_intercept)
        if plot:
            path_plot(self.coefs)

    def fit(self, alpha):
        self.coef_ = coef_calculate(alpha, self.alphas, self.coefs)

    def cv_fit(self, alpha, k=10):
        indexs = np.array_split(np.random.permutation(np.arange(0, self.__n)), k)

        def cvk(index):
            tx = self.__x[index]
            tn, tp = np.shape(tx)
            if tn == 1:
                tx = tx.reshape((1, self.__p))
            tn_ = self.__n - tn
            ty = self.__y[index]
            txt = tx.T
            txx_ = self.__xtx - np.dot(txt, tx)
            txy_ = self.__xty - np.dot(txt, ty)
            tx_sum = np.sum(tx, axis=0)
            ty_sum = np.sum(ty)
            tx_mean_ = (self.__n * self.__x_mean - tx_sum) / tn_
            ty_mean_ = (self.__n * self.__y_mean - ty_sum) / tn_
            ty_std = np.std(ty, ddof=1)

            talphas, tcoefs = LARS_pure(txx_, txy_, tx_mean_, ty_mean_,
                                         ty_std, tn, tp, self.__standarize,
                                         self.__fit_intercept)
            tcoef_ = coef_calculate(alpha, talphas, tcoefs)

            if (self.__fit_intercept == True) & (self.__standarize == True):
                tx = np.c_[np.ones((tn, 1)), tx]
            ty_pred = np.dot(tcoef_, tx.T)
            err = ty_pred - ty
            err = err * err
            return np.sum(err, axis=1)

        cv_err = np.sum(np.array([cvk(index) for index in indexs]), axis=0) / self.__n
        min_k = np.argmin(cv_err)
        self.cv_coef = self.coef_[min_k]
        print('best alpha:', alpha[min_k])

    def predict(self, tx, coef):
        tn = np.shape(tx)[0]
        if (self.__fit_intercept == True) & (self.__standarize == True):
            tx = np.c_[np.ones((tn, 1)), tx]
        return np.dot(coef, tx.T).T