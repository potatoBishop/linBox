"""寻找参数"""
from matplotlib.pyplot import plot
from playsound import playsound
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.svm import SVR

import linData
import linSVR
import linTools

'''主要参数'''
kk = linSVR.LinKernel()
kk.set_core_kind(0)

zu = linData.dataHandle()

# 需要调节的参数

# dd = 1          # degree
# cc = 4.92388     # 惩罚系数
# qq = 0.894736  # 混合核函数内的比例系数
# tt = 0
# gg = 23.35721      # gamma
# ee = 0.1        # epsilon 判断是否进行惩罚
# coco = -1    # coef0

# dd = 1          # degree            最好
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

# 高湛跑我
# dd = 2          # degree
# cc = 2.7985174# 惩罚系数
# qq = 0.40942945  # 混合核函数内的比例系数
# tt = 1 - 0.40942945
# gg = 2.7430496    # gamma
# ee = 0.1        # epsilon 判断是否进行惩罚
# coco = 0.99992022 # coef0

# dd = 1          # degree
# cc = 32.79991523  # 惩罚系数
# qq = 0.8  # 混合核函数内的比例系数
# tt = 0.2
# gg = 442.84382767    # gamma
# ee = 0.1        # epsilon 判断是否进行惩罚
# coco = 346.23850763  # coef0

# '''最新pso '''
# dd = 1          # degree
# cc = 59.41113713  # 惩罚系数
# qq = 0.89258073  # 混合核函数内的比例系数
# tt = 0.0696896
# gg = 0.23706747   # gamma
# coco = 0.84950112  # coef0
# ee = 0.1        # epsilon 判断是否进行惩罚

# '''最新pso 2023年10月11日20点11分 '''
# dd   = 1          # degree
# cc   = 3.61640306e+01  # 惩罚系数
# qq   = 1.25584208e-02  # 混合核函数内的比例系数
# tt   = 9.76380514e-01
# gg   = 4.43295719e+00   # gamma
# coco = 9.87869311e-01  # coef0
# ee   = 0.1        # epsilon 判断是否进行惩罚

'''最新pso 2023年10月11日20点35分  '''
# dd   = 2          # degree
# cc   = 0.08134405     # 惩罚系数
# qq   = 0.70580042      # 混合核函数内的比例系数
# tt   = 0.19506833
# gg   = 13.27995656  # gamma
# coco = 31.00650271    # coef0
# ee   = 0.1        # epsilon 判断是否进行惩罚

quick = [2 , 0.5769422586731177 , 0.2663928207862879 , 9.935440401733374 , 0.46492885276087637 , 159.7514421926793]

dd   = quick[0]   # degree
cc   = quick[1]   # 惩罚系数
qq   = quick[2]   # 混合核函数内的比例系数
tt   = quick[3]
gg   = quick[4]   # gamma
coco = quick[5]   # coef0
ee   = 0.1        # epsilon 判断是否进行惩罚


# filename = 'E:\Python\linSVR\8yinWu.txt'
filename = "E:\Python\linSVR\\7yinWu.txt"
# filename = "E:\linBox\data\gaozhan.txt"

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
zu.set_mode(1)          # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(85)         # 总样本数
zu.set_numNeed(17)

sum_r2_train = 0
sum_r2_test = 0
sum_RMSE_train = 0
sum_RMSE_test = 0

max_score = -1000   # 最好分数
count = 0           # 循环次数
ci = 200
num = 0

while num < ci:
    num = num + 1
    zu.fenZu()              # 分组并获取训练与测试集合
    x_tra = zu.get_x()
    y_tra = zu.get_y()
    x_test = zu.get_x_test()
    y_test = zu.get_y_test()

    # 查看结果
    print("第", num, "次随机训练")
    final_svr = kk.do_svr(x_tra, y_tra)
    # final_svr = SVR(kernel="linear", C=cc, gamma=gg, degree=dd, coef0=coco)
    # final_svr.fit(x_tra, y_tra)

    r2_train   = final_svr.score(x_tra, y_tra)
    r2_test    = final_svr.score(x_test, y_test)
    RMSE_train = linTools.get_RMSE(y_tra, final_svr.predict(x_tra))
    RMsE_test   = linTools.get_RMSE(y_test, final_svr.predict(x_test))

    print('训练集R2        ', r2_train)
    print('测试集R2        ', r2_test)
    print('RMSE(训练集)    ', RMSE_train)
    print('RMSE(测试集)    ', RMsE_test)

    sum_r2_train = sum_r2_train + r2_train
    sum_r2_test = sum_r2_test + r2_test
    sum_RMSE_train = sum_RMSE_train + RMSE_train
    sum_RMSE_test = sum_RMSE_test + RMsE_test

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("参数")
print("degree:", dd, " C:", cc, " qq:", qq, " tt:", tt, " gamma:", gg, " coef0:", coco)
print("训练集 R2 均值  ", sum_r2_train/ci)
print("测试集 R2 均值  ", sum_r2_test/ci)
print("训练集 RMSE 均值  ", sum_RMSE_train/ci)
print("测试集 RMSE 均值  ", sum_RMSE_test/ci)



