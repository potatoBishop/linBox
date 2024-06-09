import numpy
import numpy as np
import random
import math
from sklearn.svm import SVR
import linData
import linSVR
import decimal



class funtion():
    def __init__(self):
        print("starting DBO")

def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[i]:
            # temp[i] = Lb[i]   + (Ub[i] - Lb[i]) * random.uniform(0, 0.1)
            temp[i] = Lb[i]
        elif temp[i] > Ub[i]:
            # temp[i] = Ub[i]   - (Ub[i] - Lb[i]) * random.uniform(0, 0.1)
            temp[i] = Ub[i]
    return temp


def DBO(pop, M, filename, dim, lb, ub, function_name, dd ):
    """
    :param dd:              degree
    :param function_name:   适应度函数
    :param filename:        文件名称
    :param pop:             种群数量
    :param M:               迭代次数
    :param lb:              迭代范围下界
    :param ub:              迭代范围上界
    :param dim:             优化参数的个数
    :return:                适应度值最小的值 对应得位置
    """

    kk = linSVR.LinKernel()         # 混合核函数
    kk.set_core_kind(function_name)

    zu = linData.dataHandle()  # 分组
    zu.set_file(filename)            # 文件名
    zu.set_mode(0)  # 0:均匀选择4:1   1:随机选择    2: 留1法    3: 后被隐藏能源
    zu.set_dataSize(38)  # 数据总量
    zu.set_numNeed(8)  # 当mode = 1 时，需要的测试集数量
    # zu.set_dataSize(85)  # 数据总量
    # zu.set_numNeed(17)  # 当mode = 1 时，需要的测试集数量
    zu.fenZu()  # 分组并获取训练、测试、全集合
    x_train = zu.get_x()
    y_train = zu.get_y()
    x_test  = zu.get_x_test()
    y_test  = zu.get_y_test()
    x_all   = zu.get_x_all()
    y_all   = zu.get_y_all()


    P_percent = 0.2
    pNum = round(pop * P_percent)               # 跟种群数量有关的什么东西
    X    = np.zeros((pop, dim))                 # 初始化结果所需的参数    矩阵  pop * dim
    fit  = np.zeros((pop, 1))                   # 初始化R^2结果

    for i in range(pop):
        print("初始化 个体", i)
        for j in range(dim):
            X[i, j] = (lb[j]
                       + (ub[j] - lb[j]) * random.uniform(0, 1))       # 随机变化每个粒子的边界
        kk.input_settings2(X[i][0], X[i][1], dd, X[i][2], X[i][3], 1)
        svr = kk.do_svr(x_train, y_train)
        fit[i, 0] = svr.score(x_test, y_test)                           # 根据适应度来获取个体fit最有参数标志

    pFit  = fit                                 # 全局最优R^2 随机初始化
    pX    = X                                   # 全局最优位置参数  随机初始化
    XX    = pX                                  # 个体最优位置参数
    fMin  = np.max(fit[:, 0])                   # 最优值 实际上是改为求最大值
    bestI = np.argmax(fit[:, 0])                # 展平 并 显示最大值的下标
    bestX = X[bestI, :]                         # 寻得的最优参数
    Convergence_curve = np.zeros((1, M))        # 返回值


    for t in range(M):                          # 开始迭代
        B = np.argmin(pFit[:, 0])               # 最差位置的id
        worse = X[B, :]                         # 全局最差位置
        r2 = np.random.rand(1)                  #

        # 滚球蜣螂=================================
        for i in range(pNum):
            if r2 < 0.9:                # 没遇到障碍物
                a = np.random.rand(1)   # 是否偏离原来的方向 1 or -1
                if a > 0.1:
                    a = 1
                else:
                    a = -1
                X[i, :] = (pX[i, :]
                           + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :]))  # Equation(1)
            else:                       # 遇到障碍物
                aaa = np.random.randint(180, size=1)
                if aaa == 0 or aaa == 90 or aaa == 180:
                    X[i, :] = pX[i, :]
                theta = aaa * math.pi / 180
                X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])  # Equation(2)
            X[i, :]   = Bounds(X[i, :], lb, ub)
            kk.input_settings2(X[i][0], X[i][1], dd, X[i][2], X[i][3], 1)
            svr = kk.do_svr(x_train, y_train)
            fit[i, 0] = svr.score(x_test, y_test)  # 根据适应度来获取个体fit最有参数标志
        bestII = np.argmax(fit[:, 0])
        bestXX = X[bestII, :]
        R = 1 - t / M

        # 育雏球 定义新边界=================================
        Xnew1  = bestXX * (1 - R)
        Xnew2  = bestXX * (1 + R)
        Xnew1  = Bounds(Xnew1, lb, ub)  # Equation(3)
        Xnew2  = Bounds(Xnew2, lb, ub)
        Xnew11 = bestX * (1 - R)
        Xnew22 = bestX * (1 + R)  # Equation(5)
        Xnew11 = Bounds(Xnew11, lb, ub)
        Xnew22 = Bounds(Xnew22, lb, ub)
        xLB = Xnew1
        xUB = Xnew2

        # 小蜣螂        =================================
        # 局域最优觅食
        for i in range(pNum + 1, pNum + 5):   # Equation(4)    当前迭代最优
            X[i, :] = (bestXX
                       + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1)
                       + (np.random.rand(1, dim)) * (pX[i, :] - Xnew2))
            X[i, :] = Bounds(X[i, :], xLB, xUB)
            # fit[i, 0] = fun(X[i, :]) ========================================
            kk.input_settings2(X[i][0], X[i][1], dd, X[i][2], X[i][3], 1)
            svr = kk.do_svr(x_train, y_train)
            fit[i, 0] = svr.score(x_test, y_test)  # 根据适应度来获取个体fit最有参数标志

        # 全局最优觅食
        for i in range(pNum + 6, pNum + 12):         # Equation(6)   全局最优   过大
            X[i, :] = (pX[i, :]
                       + ((np.random.randn(1)) * (pX[i, :] - Xnew11)
                       + ((np.random.rand(1, dim)) * (pX[i, :] - Xnew22))))
            X[i, :] = Bounds(X[i, :], lb, ub)
            # fit[i, 0] = fun(X[i, :])========================================
            kk.input_settings2(X[i][0], X[i][1], dd, X[i][2], X[i][3], 1)
            svr = kk.do_svr(x_train, y_train)
            fit[i, 0] = svr.score(x_test, y_test)  # 根据适应度来获取个体fit最有参数标志

        # 小偷蜣螂=================================
        for i in range( pNum + 13, pop):        # Equation(7)
            X[i, :] = (bestX +
                       np.random.randn(1, dim) * (np.abs(pX[i, :] - bestXX) + np.abs(pX[i, :] - bestX)) / 2)
            X[i, :] = Bounds(X[i, :], lb, ub)
            # fit[j, 0] = fun(X[j, :])========================================
            kk.input_settings2(X[i][0], X[i][1], dd, X[i][2], X[i][3], 1)
            svr = kk.do_svr(x_train, y_train)
            fit[i, 0] = svr.score(x_test, y_test)  # 根据适应度来获取个体fit最有参数标志

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
        print("dd=", dd, bestX)
        Convergence_curve[0, t] = fMin

    return fMin, bestX, Convergence_curve
