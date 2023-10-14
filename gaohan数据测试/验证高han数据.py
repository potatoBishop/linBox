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

# 最好的位置：[1.70000000e+02 9.99832151e-01 1.06475491e-01 5.62875822e-03]
# 训练集R2 =  0.7950087525721858   测试集R2 =  0.8213239857162792

# 需要调节的参数
# '''最新pso 2023年10月11日22点25分  '''
# dd   = 2             # degree
# cc   = 28.54217271  # 惩罚系数
# qq   = 0.62311917   # 混合核函数内的比例系数
# tt   = 0.33070032
# gg   = 0.31855279    # gamma
# coco = 25.0922963   # coef0
# ee   = 0.1           # epsilon 判断是否进行惩罚

filename = "E:\linBox\data\gaozhan.txt"

# bd_low1 = [0, 0.2, 0, 0, 0]
# bd_up1 = [10, 1.0, 1, 10, 1]
# v_low1 = [0.001, 0, 0, 0, 0]
# v_up1 = [1, 0.2, 0, 1, 1]
quick = [2, 0.199219146556476, 0.7603566551467286, 105.91960701561483, 0.7214001877527543, 0.23994180874780635]

dd   = quick[0]   # degree
cc   = quick[3]   # 惩罚系数
qq   = quick[1]   # 混合核函数内的比例系数
tt   = quick[2]
gg   = quick[4]   # gamma
coco = quick[5]   # coef0
ee   = 0.1        # epsilon 判断是否进行惩罚




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
zu.set_size(70)         # 总样本数
zu.set_numNeed(14)

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



