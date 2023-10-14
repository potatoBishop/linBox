"""寻找参数"""
from matplotlib.pyplot import plot
from playsound import playsound
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import linData
import linSVR
import linTools

'''主要参数'''
kk = linSVR.LinKernel()
kk.set_core_kind(3)

zu = linData.dataHandle()
max_score = -1000   # 最好分数
count = 0           # 循环次数

# 需要调节的参数
# filename = 'E:\Python\linSVR\8yinWu.txt'
filename = "E:\Python\linSVR\\7yinWu.txt"
# dd = 1          # degree
# cc = 4.92388     # 惩罚系数
# qq = 0.894736  # 混合核函数内的比例系数
# tt = 0
# gg = 23.35721      # gamma
# ee = 0.1        # epsilon 判断是否进行惩罚
# coco = -1    # coef0

# dd = 1          # degree
# cc = 1.16878237  # 惩罚系数
# qq = 0.8  # 混合核函数内的比例系数
# tt = 0.2
# gg = 5.21339552     # gamma
# ee = 0.1        # epsilon 判断是否进行惩罚
# coco = 123.05937071   # coef0


# dd = 1          # degree
# cc = 110.01377655 # 惩罚系数
# qq = 0.8  # 混合核函数内的比例系数
# tt = 0.17575
# gg = 100.01377655    # gamma
# ee = 0.1        # epsilon 判断是否进行惩罚
# coco = 107.35968878  # coef0

dd = 1          # degree
cc = 2.7985174# 惩罚系数
qq = 0.40942945  # 混合核函数内的比例系数
tt = 1 - 0.40942945
gg = 2.7430496    # gamma
ee = 0.1        # epsilon 判断是否进行惩罚
coco = 0.99992022 # coef0

ci = 85
num = 0

# 设置模型
# 设置最优参数
kk.set_degree(dd)
kk.set_cc(cc)
kk.set_qq(qq)
kk.set_tt(tt)
kk.set_gamma(gg)
kk.set_coef(coco)

# 获取训练与测试集合
zu.set_file(filename)   # 文件名
zu.set_mode(2)          # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(85)         # 总样本数

sum_loo_r2 = 0
sum_loo_RMSE = 0


def get_MSE(y_true, y_pred):
    return (y_true - y_pred)**2

while num < ci:
    num = num + 1
    zu.set_idNeed(num)      # 设置留一法的需求ID
    zu.fenZu()              # 分组并获取训练与测试集合
    x_tra = zu.get_x()
    y_tra = zu.get_y()
    x_test = zu.get_x_test()
    y_test = zu.get_y_test()
    # x_all = zu.get_x_all()
    # y_all = zu.get_y_all()

    # 查看结果
    print("第", num, "次随机训练")
    final_svr = kk.do_svr(x_tra, y_tra)
    y_predict = final_svr.predict( x_test )

    loo_r2   = linTools.get_LOO_R2(y_test, y_predict)
    loo_RMSE = linTools.get_RMSE(y_test, final_svr.predict(x_test))

    print('留一法R2        ', loo_r2)
    print('RMSE(测试集)    ', loo_RMSE)

    sum_loo_r2   = sum_loo_r2   + loo_r2
    sum_loo_RMSE = sum_loo_RMSE + loo_RMSE

print("/n/n/n")
print("留一法 训练集 R2 均值  ", sum_loo_r2/ci)
print("留一法 训练集 RMSE 均值  ", sum_loo_RMSE/ci)



