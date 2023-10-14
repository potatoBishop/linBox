import numpy as np
from pyswarm import pso
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import sklearn.ensemble as ensemble  # ensemble learning: 集成学习
import jihe

filename = '8yinWu.txt'
zu = jihe.jihe()
zu.set_file(filename)  # 文件名
zu.set_mode(1)  # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(17)  # 当mode =0 ，为总样本数， 当 =1， 为需要的测试集数目  当 =2 ，为留一法取相应的值作为测试集

zu.fenZu()  # 分组并获取训练与测试集合
x_tra = zu.get_x()
y_tra = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()
# ======================================================================================fly
def random_forest_regression(X, y, n_estimators2, max_depth2, max_features2, min_samples_split2, min_samples_leaf2):
    model = RandomForestRegressor(n_estimators=n_estimators2,
                                  max_depth=max_depth2,
                                  max_features=max_features2,
                                  min_samples_split=min_samples_split2,
                                  min_samples_leaf=min_samples_leaf2)
    model.fit(X, y)
    return model


def objective_function(params):
    n_estimators3 = int(params[0])
    max_depth3 = int(params[1])
    max_features3 = int(params[2])
    min_samples_split3 = int(params[3])
    min_samples_leaf3 = int(params[4])
    model = random_forest_regression(x_tra, y_tra,
                                     n_estimators3,
                                     max_depth3,
                                     max_features3,
                                     min_samples_split3,
                                     min_samples_leaf3)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

lb = [50, 2, 1, 2, 1]  # 参数下限
ub = [200, 10, len(x_tra[0]), 10, 10]  # 参数上限
# lb = [200, 2, 1, 4, 3]  # 参数下限
# ub = [400, 30, len(x_tra[0]), 10, 10]  # 参数上限
bounds = [(low, high) for low, high in zip(lb, ub)]
xopt, fopt = pso(objective_function, lb, ub, swarmsize=20, maxiter=30)

n_estimators = int(xopt[0])
max_depth = int(xopt[1])
max_features = int(xopt[2])
min_samples_split = int(xopt[3])
min_samples_leaf = int(xopt[4])
rf_model = random_forest_regression(x_tra, y_tra,
                                    n_estimators,
                                    max_depth,
                                    max_features,
                                    min_samples_split,
                                    min_samples_leaf)
# rf_model=RandomForestRegressor(n_estimators=25)
rf_model.fit(x_tra, y_tra)
y_pred = rf_model.predict(x_test)
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

print('最优模型参数: ')
print('n_estimators={}'.format(n_estimators))
print('max_depth={}'.format(max_depth))
print('max_features={}'.format(max_features))
print('min_samples_split={}'.format(min_samples_split))
print('min_samples_leaf={}'.format(min_samples_leaf))


print('RMSE={:.2f}    '.format(rmse))
print('R2分数={:.2f}   '.format(r2))
print('R2(训练集)      ', rf_model.score(x_tra, y_tra))
print('R2(测试集)      ', rf_model.score(x_test, y_test))







# =======================================================================================used
# param_grid = {
#     'criterion': ['entropy', 'gini'],
#     'max_depth': [5, 6, 7, 8],           # 深度：这里是森林中每棵决策树的深度
#     'n_estimators': [11, 13, 15],        # 决策树个数-随机森林特有参数
#     'max_features': [0.3, 0.4, 0.5],     # 每棵决策树使用的变量占比-随机森林特有参数（结合原理）
#     'min_samples_split': [4, 8, 12, 16]  # 叶子的最小拆分样本量
# }
#
#
# rfc = ensemble.RandomForestClassifier()
# rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='roc_auc', cv=4)
# rfc_cv.fit(x_tra, y_tra)
#
# # 使用随机森林对测试集进行预测
# test_est = rfc_cv.predict(x_test)
# print('随机森林精确度...')
# print(metrics.classification_report(test_est, y_test))
# print('随机森林 AUC...')
# fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test)
# # 构造 roc 曲线
# print('AUC = %.4f' % metrics.auc(fpr_test, tpr_test))
