"""
============================
Gradient Boosting regression
============================

Demonstrate Gradient Boosting on the Boston housing dataset.

This example fits a Gradient Boosting model with least squares loss and
500 regression trees of depth 4.
"""
from _tkinter import _flatten

from numpy import linspace
from sklearn.model_selection import (cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV)
import sklearn
# from tkinter import _flatten
import tkinter

__all__ = [_flatten]

import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.utils import column_or_1d

# #############################################################################
# Load data
import linData

filename = 'E:\linBox\data\8yinWu.txt'
zu = linData.dataHandle()
zu.set_file(filename)  # 文件名
zu.set_mode(0)  # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(85)  # 当mode =0 ，为总样本数， 当 =1， 为需要的测试集数目  当 =2 ，为留一法取相应的值作为测试集

zu.fenZu()  # 分组并获取训练与测试集合
x_tra = zu.get_x()
y_tra = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()

i_can = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
j_can = linspace(0.001, 0.7, 30)
k_can = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# #############################################################################
# Fit regression model
params = {'n_estimators': 10}
clf = ensemble.GradientBoostingRegressor(**params)
params_grid = {
    # 'n_estimators': [ 7, 8, 9, 10, 11, 12],
    # 'learning_rate': [  0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ],
    # 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    'n_estimators': i_can,
    'learning_rate': j_can,
    'max_depth': k_can
}
gs = GridSearchCV(clf, params_grid, cv=5, verbose=100, n_jobs=4)
gs.fit(x_tra, y_tra)
print('best Parameters:', gs.best_params_)
clf.fit(x_tra, y_tra)

print('R2(训练集)      ', sklearn.metrics.r2_score(y_tra, gs.predict(x_tra)))
print('R2(测试集)      ', sklearn.metrics.r2_score(y_test, gs.predict(x_test)))
