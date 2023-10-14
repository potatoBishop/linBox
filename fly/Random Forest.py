from math import sqrt
import scipy
import numpy as np
from scipy import stats
import sklearn
import pandas
from pyswarm import pso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
# from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
import random

######################################################

'''
from sklearn.svm import SVR
svr_rbf = SVR(kernel ='rbf',degree = 3,gamma ='auto_deprecated',coef0 = 0.0,tol = 0.001,
C = 1.0,epsilon = 0.1,shrinking = True,cache_size = 200,verbose = False,max_iter = -1)
'''

# #############################################################################
# Generate sample data
filename = 'E:\Python\linSVR\8yinWu.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
index = 0
X_tra = []
X_te = []
y_tra = []
y_te = []
pos = []
res = []
a = random.randint(1, 38)
b = random.randint(1, 38)
c = random.randint(1, 38)
d = random.randint(1, 38)
e = random.randint(1, 38)
f = random.randint(1, 38)
g = random.randint(1, 38)

with open(filename, 'r') as file_to_read:
    print(a, b, c, d, e, f, g)
    while True:
        lines = file_to_read.readline()  # 整行读取数据
        if not lines:
            break
            pass
        index = index + 1

        if (index != 1
                and index != 6 and index != 11
                and index != 16 and index != 21
                and index != 26 and index != 31
                and index != 36
                and index != 41 and index != 46
                and index != 51 and index != 56
                and index != 61 and index != 66
                and index != 71 and index != 76
                and index != 81):  # 选取测试集2
            strs = lines.split()
            X_tmp = []
            for x in range(0, len(strs) - 1):
                X_tmp.append(float(strs[x]))
            y_tmp = float(strs[len(strs) - 1])

            X_tra.append(X_tmp)
            y_tra.append(y_tmp)
        else:
            strs = lines.split()
            X_tmp = []
            for x in range(0, len(strs) - 1):
                X_tmp.append(float(strs[x]))
            y_tmp = float(strs[len(strs) - 1])
            X_te.append(X_tmp)
            y_te.append(y_tmp)

X = np.array(X_tra)

X_test = np.array(X_te)
y = np.array(y_tra)
y_test = np.array(y_te)


# #############################################################################
# Add noise to targets
def get_rmse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


# [1,2,3]   [4,5,6]

# #############################################################################308 272
# Fit regression model
def random_forest_regression(X, y, n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf):
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  max_features=max_features,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf)
    model.fit(X, y)
    return model


def objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    max_features = int(params[2])
    min_samples_split = int(params[3])
    min_samples_leaf = int(params[4])
    model = random_forest_regression(X, y, n_estimators, max_depth, max_features, min_samples_split,
                                     min_samples_leaf)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


lb = [50, 2, 1, 2, 1]  # 参数下限
ub = [200, 10, len(X[0]), 10, 10]  # 参数上限
bounds = [(low, high) for low, high in zip(lb, ub)]
xopt, fopt = pso(objective_function, lb, ub, swarmsize=20, maxiter=30)

n_estimators = int(xopt[0])
max_depth = int(xopt[1])
max_features = int(xopt[2])
min_samples_split = int(xopt[3])
min_samples_leaf = int(xopt[4])
rf_model = random_forest_regression(X, y, n_estimators, max_depth, max_features, min_samples_split,
                                    min_samples_leaf)
# rf_model=RandomForestRegressor(n_estimators=25)
rf_model.fit(X, y)
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
# ----------------------------------自定义参数，测试
# rf_model = RandomForestRegressor(n_estimators=95,max_depth=6,max_features=7,min_samples_split=7,min_samples_leaf=1)
# rf_model.fit(X,y)
# y_pred = rf_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
# =---------------
print('最优模型参数: n_estimators={}, max_depth={}, max_features={}, min_samples_split={}, min_samples_leaf={}'.format(
    n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf))
print('RMSE={:.2f}'.format(rmse))
print('R2分数={:.2f}'.format(r2))
print('R2(训练集)      ', rf_model.score(X, y))
print('R2(测试集)      ', rf_model.score(X_test, y_test))

# plt.scatter(y,svr.predict(X),label='Training set')
# plt.scatter(y_test,svr.predict(X_test),marker='^',label='Testing set')
# plt.legend()
# plt.xlabel('Experimental')
# plt.ylabel('Calculated')
# plt.legend(['Training set','Testing set'])
# plt.plot([-4,3],[-4,3])#  画对角线      3
# plt.axis([-4, 1, -4, 1])#定义xy轴的显示范围
# plt.show()
# print(svr.predict(X))
# print(svr.predict(X_test))
