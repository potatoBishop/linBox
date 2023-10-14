import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.metrics._regression import _check_reg_targets, _assemble_r2_explained_variance
from sklearn.utils import check_consistent_length, validate_params

import linData


# def LOO(size, kk):
#     num = 1
#     zu = linData.dataHandle()
#     while num <= size:
#
#         num = num + 1


def get_RMSE(testY, testPredict):
    # 均方根误差：RMSE（Root Mean Squard Error）RMSE=sqrt（MSE）
    # 范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大。
    return np.sqrt(mean_squared_error(testY, testPredict))

def get_MSE(testY, testPredict):
    # 均方误差：MSE（Mean Squared Error）
    # 范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大
    return mean_squared_error(testY, testPredict)

def get_MAE(testY, testPredict):
    # 平均绝对误差：MAE（Mean Absolute Error）
    # 范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大。
    return mean_absolute_error(testY, testPredict)

def get_R2(testY, testPredict):
    # 决定系数：R2（R-Square）
    return r2_score(testY, testPredict)

# def get_LOO_R2(
#             y_true,
#             y_pred,
#             *,
#             sample_weight=None,
#             multioutput="uniform_average",
#             force_finite=True,
#     ):
#     # 原版的R2无法应用于留一法 testY : testPredict = n : 1
#     y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
#     check_consistent_length(y_true, y_pred, sample_weight)
#
#     weight = 1.0
#
#     numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
#     denominator = (
#             weight * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2
#     ).sum(axis=0, dtype=np.float64)
#
#     return _assemble_r2_explained_variance(
#         numerator=numerator,
#         denominator=denominator,
#         n_outputs=y_true.shape[1],
#         multioutput=multioutput,
#         force_finite=force_finite,
#     )
#
#
# @validate_params(
#     {
#         "y_true": ["array-like"],
#         "y_pred": ["array-like"],
#     },
#     prefer_skip_nested_validation=True,
# )
def get_LOO_R2(testY, testPredict):
    ssr = ((testPredict - testY.mean()) ** 2).sum()  # 预测数据和原始均值之差 的平方和
    sst = ((testY - testY.mean()) ** 2).sum()  # 原始数据 和 均值之差  的平方和
    r2 = ssr / sst
    return r2

def get_MAPE(y_true, y_pred):
    # 平均绝对百分比误差（Mean Absolute Percentage Error）
    # 范围[0,+∞)，MAPE 为0%表示完美模型，MAPE 大于 100 %则表示劣质模型。
    # 真实数据不能为 0
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def get_SMAPE(y_true, y_pred):
    # 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100