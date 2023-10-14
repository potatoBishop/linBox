import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from ALO import ALO

import jihe

zu = jihe.jihe()
filename = '/8yinWu.txt'
zu.set_file(filename)   # 文件名
zu.set_mode(1)          # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(17)         # 当mode =0 ，为总样本数， 当 =1， 为需要的测试集数目  当 =2 ，为留一法取相应的值作为测试集
zu.fenZu()  # 分组并获取训练与测试集合

x_tra = zu.get_x()
y_tra = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()

# 加载数据
# X = np.loadtxt('data.txt', delimiter=',', usecols=(0, 1, 2))
# y = np.loadtxt('data.txt', delimiter=',', usecols=(3,))

# 定义SVR模型
svr = SVR(kernel='rbf')

# 定义参数范围
param_grid = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}

# 定义ALO蚁狮优化算法
alo = ALO(svr, param_grid)

# 训练模型
alo.fit(x_tra, y_tra)

# 输出最优参数值
print('最优参数值:', alo.best_params_)
