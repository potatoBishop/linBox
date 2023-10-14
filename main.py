from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import linData
import linSVR
import linTools

dd = 1          # degree
cc = 110.01377655 # 惩罚系数
qq = 0.8  # 混合核函数内的比例系数
tt = 0.17575
gg = 100.01377655    # gamma
ee = 0.1        # epsilon 判断是否进行惩罚
coco = 107.35968878  # coef0

kk = linSVR.LinKernel()     # 混合核函数
kk.set_core_kind(3)
kk.input_settings(dd, qq, tt, cc, gg, coco)

filename = 'E:\linBox\data\\7yinWu.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
zu = linData.dataHandle()        # 分组
zu.set_file(filename)   # 文件名
zu.set_mode(1)          # 0:均匀选择4:1   1:随机选择   2: 留1法
zu.set_size(85)         # 数据总量
zu.set_numNeed(17)      # 当mode = 1 时，需要的测试集数量
zu.fenZu()              # 分组并获取训练与测试集合
x_train = zu.get_x()
y_train = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()
x_all = zu.get_x_all()
y_all = zu.get_y_all()

svr = kk.do_svr(x_train, y_train)
y_preditc = svr.predict( x_test )

print(linTools.get_R2(y_test, svr.predict(x_test)))      # sklean内置 r2——score
print(linTools.get_LOO_R2(y_test, svr.predict(x_test)))  # 用第二个公式计算
print(svr.score(x_test, y_test))
print( 1- (mean_squared_error(y_test,y_preditc)/ np.var(y_test)) )  # 第一个公式



