import numpy as np
import random
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

import linData
import linSVR
import linTools

# filename = 'E:\linBox\data\\7yinWu.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
filename = 'E:\linBox\data\\gaozhan.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
zu = linData.dataHandle()        # 分组
kk = linSVR.LinKernel()     # 混合核函数


index = 0
X_tra = []
X_te = []
y_tra = []
y_te = []
pos = []
res = []
x_min = 0
v_min = 0
vp_min = 0
x_max = 500
v_max = 200
vp_max = 1

w_max = 0.8
w_min = 0.2

qq = 0.1
tt = 0.1
dd = 1

zu.set_file(filename)   # 文件名
zu.set_mode(1)          # 0:均匀选择4:1   1:随机选择   2: 留1法
zu.set_size(70)         # 数据总量
zu.set_numNeed(14)      # 当mode = 1 时，需要的测试集数量
zu.fenZu()              # 分组并获取训练与测试集合

kk.set_core_kind(3)

X      = zu.get_x()
y      = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()
x_all  = zu.get_x_all()
y_all  = zu.get_y_all()

# #############################################################################
class PSO():
    def __init__(self, pN, dim, max_iter):
        # 定义所需变量
        self.w = 0.8
        self.c1 = 2         # 学习因子
        self.c2 = 2

        self.r1 = 0.6       # 超参数
        self.r2 = 0.3

        self.pN = pN                # 粒子数量
        self.dim = dim              # 搜索维度
        self.max_iter = max_iter    # 迭代次数

        # 定义各个矩阵大小
        self.X = np.zeros((self.pN, self.dim))      # 所有粒子的位置和速度矩阵
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置矩阵
        self.gbest = np.ones(self.dim)
        self.p_fit = np.zeros(self.pN)              # 每个个体的历史最佳适应值
        self.fit = 0                                # 全局最佳适应值       把1e10改成0，让R2做适应值，越高越好

        self.init_Population()


    # 目标函数，根据使用场景进行设置
    def function(self, x):
        cc = x[0]
        gg = x[1]
        coco = x[2]

        kk.input_settings(dd, qq, tt, cc, gg, coco)
        svr = kk.do_svr(X, y)
        # print( linTools.get_R2( y_test, svr.predict(X_test) ) )         # 正确
        # print( linTools.get_LOO_R2( y_test, svr.predict(X_test)) )
        # print("========================================================")
        # score1 = svr.score(X_test, y_test)
        # score1 = cross_val_score(svr, x_all, y_all, cv=5, scoring='r2')  # 交叉验证计算r方
        # return score1.mean()
        return svr.score(x_test, y_test)


    # 初始化粒子群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 200)
                self.V[i][j] = random.uniform(0, 20)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            #    R2越大越好
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = self.X[i]

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            print("=======================================================================================第 ", t, "次")
            for i in range(self.pN):                        # 更新gbest 和 pbest
                temp = self.function( self.X[i] )           # 获取当前粒子对应参数的 score ，即 训练集R^2
                if float(temp) > float(self.p_fit[i]):      # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]               # 更新个体最优对应的的矩阵
                if self.p_fit[i] > self.fit:                # 更新全局最优
                    print("全局最优更新")
                    print(self.gbest)
                    self.gbest = self.X[i].copy()           # 更新全局最优对应的参数矩阵
                    self.fit = self.p_fit[i]

            self.w = w_max - (w_max - w_min) * (t / self.max_iter) ** 2     # 更新惯性权重

            for i in range(self.pN):
                # 粒子群算法公式   更新每一个粒子的速度和位置
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                # 边界条件，粒子的速度和位置不能超过边界值
                for j in range(self.dim):
                    if(self.V[i][j] > v_max) or (self.V[i][j] < v_min):
                        self.V[i][j] = v_min + random.uniform(0, 1) * (v_max - v_min)
                    if (self.X[i][j] > x_max) or (self.X[i][j] < x_min):
                        self.X[i][j] = x_min + random.uniform(0, 1) * (x_max - x_min)
            fitness.append(self.fit)
            kk.input_settings(dd, qq, tt, self.gbest[0], self.gbest[1], self.gbest[2])               # 设置每次迭代的最优参数
            svr = kk.do_svr(X, y)

            # 每一代的最佳测试集R2及最佳参数
            print('R2(训练集)      ', svr.score( X, y ) )
            print('R2(测试集)      ', svr.score(x_test, y_test) )
            print('RMSE(训练集)    ', linTools.get_RMSE(y, svr.predict(X)))
            print('RMSE(测试集)    ', linTools.get_RMSE(y_test, svr.predict(x_test)))
            print('最优值R2(测试集) ', self.fit)  # 输出最优值
            print('最佳参数为       ', self.gbest)
        return fitness

if __name__ == '__main__':
    # q_can = np.linspace(0.4, 0.6, 10)  # qq
    # t_can = np.linspace(0.001, 0.6, 10)
    num = 1
    # my_pso = PSO(pN=30, dim=5, max_iter=100)
    qq = 0.75
    tt = 0.2
    #

    my_pso = PSO(pN=60, dim=3, max_iter=50)
    fitness = my_pso.iterator()
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=50)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 200)])
    fitness = np.array(fitness)
    print(len(t),len(fitness))
    plt.plot(t, fitness, color='b', linewidth=3)
    plt.show()