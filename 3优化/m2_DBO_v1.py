import numpy
import numpy as np
import random
import math

from sklearn.svm import SVR

import linData
import linSVR
import decimal

# filename = 'E:\linBox\data\\7引物ci2.txt'  # 读取数据
filename = 'E:\linBox\data\\7yinWu.txt'     # 读取数据
# filename = 'E:\linBox\data\\gaozhan.txt'  # 读取数据
zu = linData.dataHandle()  # 分组
kk = linSVR.LinKernel()    # 混合核函数
dd = 3

zu.set_file(filename)   # 文件名
zu.set_mode(0)          # 0:均匀选择4:1   1:随机选择    2: 留1法    3: 后被隐藏能源
zu.set_dataSize(85)         # 数据总量
zu.set_numNeed(17)      # 当mode = 1 时，需要的测试集数量
zu.fenZu()              # 分组并获取训练、测试、全集合
x_train = zu.get_x()
y_train = zu.get_y()
x_test  = zu.get_x_test()
y_test  = zu.get_y_test()
x_all   = zu.get_x_all()
y_all   = zu.get_y_all()
# F2

class funtion():
    def __init__(self):
        print("starting DBO")


# 根据F的值来获取一个参数集合，用来支撑后面的训练    没什么大用
def Parameters(F):
    """根据F的值来设定 边界 和 粒子群数目 """
    # 默认值
    fobj = F1
    lb = -100
    ub = 100
    dim = 30

    # 判断
    if F == 'F1':
        # ParaValue=[-100,100,30] [-100,100]代表初始范围，30代表dim维度
        fobj = F1
        lb   = -100
        ub   = 100
        dim  = 30
    elif F == 'm3':
        fobj = m3
        # c, g, co , q
        # lb = [  10,    0,  -1,   0.2 ]
        # ub = [   150,  0.8,   60,   0.55 ]
        # lb = [  190,    0.05,  -130,   0.6 ]
        # ub = [   215,   0.08,   -90,   0.85 ]
        # lb = [  0.0001,    0,  5,   0.2 ]
        # ub = [   0.1,     30,  30,   0.65 ]
        # lb = [  80,    0,     10,   0.2 ]           # best dd=2 data = 2
        # ub = [   100,   0.2,  75,   0.55 ]
        lb = [  0.001,     0.001,     -50,   0.15 ]
        ub = [  40,     50,        50,   0.85 ]
        dim = 4
        kk.set_core_kind("m3")

    return fobj, lb, ub, dim


# F1
def F1(x):
    o = np.sum(np.square(x))
    return o

def m3(x):
    kk.input_settings2(x[0], x[1], dd, x[2], x[3], 1)
    svr = kk.do_svr(x_train, y_train)
    return svr.score(x_test, y_test)



def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[i]:
            temp[i] = Lb[i]   + (Ub[i] - Lb[i]) * random.uniform(0, 0.1)
        elif temp[i] > Ub[i]:
            temp[i] = Ub[i]   - (Ub[i] - Lb[i]) * random.uniform(0, 0.1)
    return temp


def DBO(pop, M, c, d, dim):
    """
    :param fun: 适应度函数
    :param pop: 种群数量
    :param M: 迭代次数
    :param c: 迭代范围下界
    :param d: 迭代范围上界
    :param dim: 优化参数的个数
    :return: 适应度值最小的值 对应得位置
    """

    P_percent = 0.2
    pNum = round(pop * P_percent)               # 跟种群数量有关的什么东西
    # lb   = c * np.ones((1, dim))                # 初始化每个维度的下边界  数组
    # ub   = d * np.ones((1, dim))                # 初始化每个维度的上边界  数组
    lb   = c
    ub   = d
    X    = np.zeros((pop, dim))                 # 初始化结果所需的参数    矩阵  pop * dim
    fit  = np.zeros((pop, 1))                   # 初始化最优参数

    for i in range(pop):
        print("初始化 个体", i)
        for j in range(dim):
            X[i, j] = lb[j] + (ub[j] - lb[j]) * random.uniform(0, 1)       # 随机变化每个粒子的边界
        # fit[i, 0] = fun(X[i, :])                                 # 根据适应度来获取个体fit最有参数标志
        fit[i, 0] = m3(X[i, :])                                 # 根据适应度来获取个体fit最有参数标志

    pFit  = fit                                 # 全局最优参数标志  随机初始化
    pX    = X                                   # 全局最优位置参数  随机初始化
    XX    = pX                                  # 往下没看懂
    fMin  = np.max(fit[:, 0])                   # 最优值
    bestI = np.argmax(fit[:, 0])                # 展平 并 显示最小值的下标
    bestX = X[bestI, :]                         # 最有变量

    Convergence_curve = np.zeros((1, M))        # 不知道是什么东西

    for t in range(M):                          # 开始迭代
        # sortIndex = np.argsort(pFit.T)
        # fmax = np.max(pFit[:, 0])
        B = np.argmin(pFit[:, 0])
        worse = X[B, :]  #
        r2 = np.random.rand(1)
        # v0 = 0.5
        # v = v0 + t/(3*M)
        for i in range(pNum):
            if r2 < 0.9:
                r1 = np.random.rand(1)
                a = np.random.rand(1)
                if a > 0.1:
                    a = 1
                else:
                    a = -1
                X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])  # Equation(1)
            else:
                aaa = np.random.randint(180, size=1)
                if aaa == 0 or aaa == 90 or aaa == 180:
                    X[i, :] = pX[i, :]
                theta = aaa * math.pi / 180
                X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])  # Equation(2)
            X[i, :]   = Bounds(X[i, :], lb, ub)
            # fit[i, 0] = fun(X[i, :])
            fit[i, 0] = m3(X[i, :])
        bestII = np.argmax(fit[:, 0])
        bestXX = X[bestII, :]

        R = 1 - t / M

        Xnew1  = bestXX * (1 - R)
        Xnew2  = bestXX * (1 + R)
        Xnew1  = Bounds(Xnew1, lb, ub)  # Equation(3)
        Xnew2  = Bounds(Xnew2, lb, ub)
        Xnew11 = bestX * (1 - R)
        Xnew22 = bestX * (1 + R)  # Equation(5)
        Xnew11 = Bounds(Xnew11, lb, ub)
        Xnew22 = Bounds(Xnew22, lb, ub)
        # xLB = swapfun(Xnew1)
        # xUB = swapfun(Xnew2)
        xLB = Xnew1
        xUB = Xnew2

        for i in range(pNum + 1, 12):   # Equation(4)    当前迭代最优
            X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
            X[i, :] = Bounds(X[i, :], xLB, xUB)
            # fit[i, 0] = fun(X[i, :])
            fit[i, 0] = m3(X[i, :])
        for i in range(13, 19):         # Equation(6)   全局最优   过大
            X[i, :] = (pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) +
                                       ((np.random.rand(1, dim)) * (pX[i, :] - Xnew22))))
            X[i, :] = Bounds(X[i, :], lb, ub)
            # fit[i, 0] = fun(X[i, :])
            fit[i, 0] = m3(X[i, :])
        for j in range(20, pop):        # Equation(7)
            X[j, :] = bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
            X[j, :] = Bounds(X[j, :], lb, ub)
            # fit[j, 0] = fun(X[j, :])
            fit[j, 0] = m3(X[j, :])

        # Update the individual's best fitness vlaue and the global best fitness value
        XX = pX
        for i in range(pop):
            print("个体", i, "R^2 = ", fit[i, 0])
            if fit[i, 0] > pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
            if pFit[i, 0] > fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]
                print("全局最优更新")
                print(bestX)

        print("第", t, "次迭代, 全体最优R^2 = ", fMin, )
        print(bestX)
        Convergence_curve[0, t] = fMin

    return fMin, bestX, Convergence_curve
