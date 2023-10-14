"""寻找参数"""
from playsound import playsound
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

import jihe
import myCore
from main import get_rmse

'''主要参数'''
kk = myCore.MyKernelClass()

zu = jihe.jihe()
max_score = -1000   # 最好分数
count = 0           # 循环次数

# 需要调节的参数
filename = '/8yinWu.txt'
dd = 1          # degree
cc = 4.368421052631579       # 惩罚系数
gg = 0.3      # gamma
# ee = 0.1        # epsilon 判断是否进行惩罚
# coco = -0.8     # coef0


# 设置模型
# 设置最优参数
kk.set_degree(dd)
kk.set_cc(cc)
kk.set_gamma(gg)
kk.set_coef(0)

# 获取训练与测试集合
zu.set_file(filename)   # 文件名
zu.set_mode(0)          # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(85)         # 当mode =0 ，为总样本数， 当 =1， 为需要的测试集数目  当 =2 ，为留一法取相应的值作为测试集

# 分组并获取训练与测试集合
zu.fenZu()
x_tra = zu.get_x()
y_tra = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()

# 查看结果
kk.set_core_kind(4)
final_svr = kk.do_svr(x_tra, y_tra)
print("自写rbf")
print(final_svr)
print('R2(训练集)      ', final_svr.score(x_tra, y_tra))
print('R2(测试集)      ', final_svr.score(x_test, y_test))
print('RMSE(训练集)    ', get_rmse(y_tra, final_svr.predict(x_tra)))
print('RMSE(测试集)    ', get_rmse(y_test, final_svr.predict(x_test)))
print("dd=", dd)
print("cc=", cc)
print("gg=", gg)

print("===========================================================")

kk.set_core_kind(5)
final_svr = kk.do_svr(x_tra, y_tra)
print("自带rbf")
print(final_svr)
print('R2(训练集)      ', final_svr.score(x_tra, y_tra))
print('R2(测试集)      ', final_svr.score(x_test, y_test))
print('RMSE(训练集)    ', get_rmse(y_tra, final_svr.predict(x_tra)))
print('RMSE(测试集)    ', get_rmse(y_test, final_svr.predict(x_test)))
print("dd=", dd)
print("cc=", cc)
print("gg=", gg)







