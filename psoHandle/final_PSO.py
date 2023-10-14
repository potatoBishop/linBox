import numpy as np
import random
import matplotlib.pyplot as plt
import linData
import linSVR
import linTools

filename = 'E:\linBox\data\\7yinWu.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
# filename = 'E:\linBox\data\\gaozhan.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1

zu = linData.dataHandle()  # 分组
kk = linSVR.LinKernel()  # 混合核函数

w_max = 0.8
w_min = 0.3

dd = 2
cc = 1.0
qq = 0.1
tt = 0.1
gg = 0.1
coco = 0.1

zu.set_file(filename)  # 文件名
zu.set_mode(0)  # 0:均匀选择4:1   1:随机选择   2: 留1法
zu.set_size(85)  # 数据总量
zu.set_numNeed(17)  # 当mode = 1 时，需要的测试集数量
zu.fenZu()  # 分组并获取训练与测试集合

kk.set_core_kind(3)

x_train = zu.get_x()
y_train = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()
x_all = zu.get_x_all()
y_all = zu.get_y_all()


# #############################################################################
class PSO:
    def __init__(self, pN, dim, max_iter, bd_low, bd_up, v_low, v_up):
        # 定义所需变量
        self.w = 0.8              # 速度变化权重系数
        self.c1 = 2               # 学习因子1
        self.c2 = 2               # 学习因子2
        self.r1 = 0.6             # 超参数1
        self.r2 = 0.3             # 超参数2
        self.pN = pN              # 粒子数量
        self.dim = dim            # 搜索维度
        self.max_iter = max_iter  # 迭代次数

        # 定义各个矩阵大小
        self.X = np.zeros((self.pN, self.dim))      # 所有粒子的     位置矩阵
        self.V = np.zeros((self.pN, self.dim))      # 所有粒子的     速度矩阵
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的     最佳位置矩阵
        self.gbest = np.ones(self.dim)              # 个体经历的     全局最佳位置矩阵
        self.p_fit = np.zeros(self.pN)              # 每个个体的     历史最佳适应值
        self.fit = 0                                # 全局最佳适应值  让R2_score做适应值，越高越好

        # 初始话粒子群
        self.init_Population(bd_low=bd_low, bd_up=bd_up, v_low=v_low, v_up=v_up)

    # 目标函数，根据使用场景进行设置
    @staticmethod
    def function(x):
        # kk.input_settings(dd, qq, tt, cc, gg, coco)              输入参数比对
        kk.input_settings(dd, x[1], x[2], x[0], x[3], x[4])
        svr = kk.do_svr(x_train, y_train)
        return svr.score(x_test, y_test)

    # 初始化粒子群
    def init_Population(self, bd_low, bd_up, v_low, v_up):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(bd_low[j], bd_up[j])
                self.V[i][j] = random.uniform(v_low[j], v_up[j])
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])   # 返回r2
            self.p_fit[i] = tmp
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = self.X[i]

    def iterator(self, bd_low, bd_up, v_low, v_up):
        fitness = []
        for t in range(self.max_iter):
            print("==================================================================第 ", t, "次迭代")
            for i in range(self.pN):  # 更新gbest 和 pbest
                temp = self.function(self.X[i])  # 获取当前粒子对应参数的 score ，即 训练集R^2
                if float(temp) > float(self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]  # 更新个体最优对应的的矩阵
                if self.p_fit[i] > self.fit:  # 更新全局最优
                    self.gbest = self.X[i].copy()  # 更新全局最优对应的参数矩阵
                    self.fit = self.p_fit[i]
                    print("=====全局最优更新======")
                    print(self.gbest)

            if self.fit >= 0.75:
                self.w = w_max - (w_max - w_min) * (t / self.max_iter) ** 2
            else:
                self.w = 0.3 + random.uniform(0, 0.5)

            # self.w = w_max - (w_max - w_min) * (t / self.max_iter) ** 2  # 更新惯性权重  随迭代次数的增大，X的变化减小   stand
            # self.w = w_min + (w_max - w_min) * (1.0 / (t+1) ) ** 2  # 更新惯性权重  随迭代次数的增大，X的变化减小

            for i in range(self.pN):
                # 粒子群算法公式   更新每一个粒子的速度和位置
                self.V[i] = (self.w * self.V[i] +
                             self.c1 * self.r1 * (self.pbest[i] - self.X[i]) +
                             self.c2 * self.r2 * (self.gbest - self.X[i]))
                self.X[i] = self.X[i] + self.V[i]
                # 边界条件，粒子的速度和位置不能超过边界值
                for j in range(self.dim):
                    # print("====维度:", j, "速度:", self.V[i][j])
                    if (self.V[i][j] > v_up[j]) or (self.V[i][j] < v_low[j]):
                        self.V[i][j] = v_low[j] + random.uniform(0, 1) * (v_up[j] - v_low[j])
                    if (self.X[i][j] > bd_up[j]) or (self.X[i][j] < bd_low[j]):
                        self.X[i][j] = bd_low[j] + random.uniform(0, 1) * (bd_up[j] - bd_low[j])
            fitness.append(self.fit)

            kk.input_settings(dd, self.gbest[1], self.gbest[2], self.gbest[0], self.gbest[3], self.gbest[4])  # 设置每次迭代的最优参数
            svr = kk.do_svr(x_train, y_train)

            # 每一代的最佳测试集R2及最佳参数
            print('R2(训练集)      ', svr.score( x_train, y_train ) )
            print('R2(测试集)      ', svr.score(  x_test, y_test ))
            print('RMSE(训练集)    ', linTools.get_RMSE( y_train, svr.predict( x_train )))
            print('RMSE(测试集)    ', linTools.get_RMSE( y_test, svr.predict( x_test )))
            print('最优值R2(测试集) ', self.fit)  # 输出最优值
            print('最佳参数为       ', self.gbest)
        return fitness


if __name__ == '__main__':
    #            C,       q,     t,   gamma, coef0
    bd_low1 = [   0,     0.2,    0,    0,    0  ]
    bd_up1  = [ 200,     1.0,  0.75,   100,    1  ]
    v_low1  = [ 0.001,     0,    0,     0,    0  ]
    v_up1   = [  20,     0.2,   0.2,     1,    1  ]

    iter1 = 100

    my_pso = PSO(pN=60, dim=5, max_iter=iter1, bd_low=bd_low1, bd_up=bd_up1, v_low=v_low1, v_up=v_up1)
    fitness = my_pso.iterator( bd_low=bd_low1, bd_up=bd_up1, v_low=v_low1, v_up=v_up1 )

    '''图像部分'''
    plt.figure(1)
    plt.title("linSVR")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    tt = np.array([t for t in range(1, iter1)])
    fitness = np.array(fitness)
    print(len(tt), len(fitness))
    plt.plot(tt, fitness, color='b', linewidth=3)
    plt.show()
