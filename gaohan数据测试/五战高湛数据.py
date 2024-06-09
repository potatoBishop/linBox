import numpy as np
import random
import matplotlib.pyplot as plt
import linData
import linSVR
import linTools


# filename = 'E:\linBox\data\\7yinWu.txt'     # 读取数据
filename = 'E:\linBox\data\\gaozhan.txt'  # 读取数据

# lin程序
zu = linData.dataHandle()  # 分组
kk = linSVR.LinKernel()    # 混合核函数

# 核函数参数
dd = 2
cc = 1.0
qq = 0.1
tt = 0.1
gg = 0.1
coco = 0.1

# PSO参数
w_max = 0.8
w_min = 0.2
guoNiHe = 0.93

zu.set_file(filename)   # 文件名
zu.set_mode(0)          # 0:均匀选择4:1   1:随机选择    2: 留1法
zu.set_size(70)         # 数据总量
zu.set_numNeed(14)      # 当mode = 1 时，需要的测试集数量


zu.fenZu()              # 分组并获取训练、测试、全集合
x_train = zu.get_x()
y_train = zu.get_y()
x_test  = zu.get_x_test()
y_test  = zu.get_y_test()
x_all   = zu.get_x_all()
y_all   = zu.get_y_all()

# #############################################################################                     PSO主程序
class PSO:
    def __init__(self, pN, dim, max_iter, bd_low, bd_up, v_low, v_up):  # 初始化PSO参数
        # 定义所需变量
        self.w   = 0.8            # 速度变化权重系数
        self.c1  = 2              # 学习因子1
        self.c2  = 2              # 学习因子2
        self.r1  = 0.6            # 超参数1
        self.r2  = 0.3            # 超参数2
        self.pN  = pN             # 粒子数量
        self.dim = dim            # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.bd_low   = bd_low    # pos边界
        self.bd_up    = bd_up
        self.v_low    = v_low     # 速度边界
        self.v_up     = v_up

        # 定义各个矩阵大小
        self.pos   = np.zeros((self.pN, self.dim))      # 所有粒子的     位置矩阵
        self.v     = np.zeros((self.pN, self.dim))      # 所有粒子的     速度矩阵
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的     最佳位置矩阵
        self.gbest = np.ones(self.dim)              # 个体经历的     全局最佳位置矩阵
        self.p_fit = np.zeros(self.pN)              # 每个个体的     历史最佳适应值
        self.fit = 0                                # 全局最佳适应值  让R2_score做适应值，越高越好

        # 初始化粒子群
        self.init_Population()

    def init_Population(self):  # 初始化粒子群
        for i in range(self.pN):
            for j in range(self.dim):
                self.pos[i][j] = random.uniform( self.bd_low[j], self.bd_up[j] )
                self.v[i][j]   = random.uniform(  self.v_low[j],  self.v_up[j] )

    @staticmethod
    def function(pos):    # 返回score
        # kk.input_settings(dd, qq, tt, cc, gg, coco)              输入参数比对
        kk.input_settings(dd, pos[1], pos[2], pos[0], pos[3], pos[4])
        svr = kk.do_svr(x_train, y_train)
        return svr.score(x_test, y_test), svr.score(x_train, y_train)


    def iterator(self):
        fitness = []

        for t in range(self.max_iter):
            can = t / self.max_iter
            self.c1 = 2 - can
            self.c2 = 1 + can
            self.r1 = random.uniform(0, 1)
            self.r2 = random.uniform(0, 1)
            # self.w = w_max - (w_max - w_min) * can  # 趋于稳定开始收敛
            print("==================================================================第 ", t, "次迭代==")

            for i in range(self.pN):
                temp_r2_test, temp_r2_train = self.function(self.pos[i])                        # 获取当前粒子对应参数的 score ，即 训练集R^2
                print("个体", i, "  r2:", temp_r2_test)
                if float(temp_r2_test) > float(self.p_fit[i]) and (temp_r2_train < guoNiHe):    # 更新个体最优 and 防止过拟合
                    self.p_fit[i] = temp_r2_test
                    self.pbest[i] = self.pos[i]                                                 # 更新个体最优对应的的矩阵
                    # print("==================================================================个体", i, "最优更新==")
                if self.p_fit[i] > self.fit:                    # 更新全局最优
                    self.gbest = self.pos[i].copy()                                               # 更新全局最优对应的参数矩阵
                    self.fit = self.p_fit[i]
                    print("==================================================================全局最优更新==")
                    print(self.gbest)

                # 更新每一代的权重系数w
                if 0.5 <= temp_r2_test < 0.94:
                    self.w = w_max - (w_max - w_min) * (can ** 2)         # 趋于稳定开始收敛
                else:
                    self.w = 0.9
                    # self.w = random.uniform( max(0.5, (t/self.max_iter)), 0.9)                                                        # 随着迭代次数增加，如果一直不到0.7，则w增大，变暴躁
                # self.w = w_max - (w_max - w_min) * (t / self.max_iter) ** 2  # 趋于稳定开始收敛

                # 更新 每一代的 不同粒子的 速度
                self.v[i] = (self.w * self.v[i] +
                             self.c1 * self.r1 * (self.pbest[i] - self.pos[i]) +
                             self.c2 * self.r2 * (self.gbest    - self.pos[i]) )


                # 更新 每一代的 不同粒子的 位置参数
                self.pos[i] = self.pos[i] + self.v[i]

                # 边界条件，粒子的速度和位置不能超过边界值
                for j in range(self.dim):
                    if (self.v[i][j] > self.v_up[j]) or (self.v[i][j] < self.v_low[j]):
                        self.v[i][j] = self.v_low[j] + random.uniform(0, 1) * (self.v_up[j] - self.v_low[j])
                    if (self.pos[i][j] > self.bd_up[j]) or (self.pos[i][j] < self.bd_low[j]):
                        self.pos[i][j] = self.bd_low[j] + random.uniform(0, 1) * (self.bd_up[j] - self.bd_low[j])

                # 处理 qq + tt > 1 问题
                while (self.pos[i][1] + self.pos[i][2]) > 1:
                    self.pos[i][1] = self.bd_low[1] + random.uniform(0, 1) * (self.bd_up[1] - self.bd_low[1])
                    self.pos[i][2] = self.bd_low[2] + random.uniform(0, 1) * (self.bd_up[2] - self.bd_low[2])

            # 收集每次迭代完成的最优score， 存储用于生成图像
            fitness.append(self.fit)

            # 设置混合核函数 查看
            kk.input_settings(dd, self.gbest[1], self.gbest[2], self.gbest[0], self.gbest[3], self.gbest[4])  # 设置每次迭代的最优参数
            svr = kk.do_svr(x_train, y_train)

            # 每一代的最佳测试集R2及最佳参数
            r2_train   = svr.score( x_train, y_train )
            r2_test    = svr.score(  x_test,  y_test )
            RMSE_train = linTools.get_RMSE( y_train, svr.predict( x_train ))
            RMSE_test  = linTools.get_RMSE(  y_test, svr.predict(  x_test ))
            if r2_train > 0.94:
                flag_guo = 1
                print("==================================================================本次迭代发生过拟合==")
            print('R2(训练集)      ', r2_train  )
            print('R2(测试集)      ', r2_test   )
            print('RMSE(训练集)    ', RMSE_train)
            print('RMSE(测试集)    ', RMSE_test )
            print('最优值R2(测试集) ', self.fit  )      # 输出最优值
            print('最佳参数为  dd qq tt cc gg coco', )
            print(dd, ",", self.gbest[1], ",", self.gbest[2], ",", self.gbest[0], ",", self.gbest[3], ",", self.gbest[4] )
        return fitness


# #############################################################################                     main区域

if __name__ == '__main__':
    #           C,   q,   t,  gamma, coef0
    bd_low1 = [ 0,    0,   0,     0,    0 ]
    bd_up1  = [170,   1,   1,     10,     1 ]
    v_low1  = [ 0,   0,  0,     0,    0]
    v_up1   = [ 1,    1,   1,     1,   1 ]
    dd = 3
    kk.set_core_kind(5)  # 设置核函数

    # bd_low1 = [   0,   0.4,      0,      0,    0 ]
    # bd_up1  = [   2,   1,      0.6,    7,      1 ]
    # v_low1  = [   0,     0,      0,     0,     0  ]
    # v_up1   = [   1,   0.5,    0.5,     1,    1  ]
    # dd  = 2

    # bd_low1 = [   80,    0,      0,      0,    80 ]
    # bd_up1  = [   120,   1,       1,    10,    120 ]
    # v_low1  = [   0,     0,      0,      0,     0  ]
    # v_up1   = [   10,   0.5,    0.5,     1,    10  ]
    # dd  = 2
    iter1 = 50

    my_pso = PSO(pN=50, dim=5, max_iter=iter1, bd_low=bd_low1, bd_up=bd_up1, v_low=v_low1, v_up=v_up1)
    fitness = my_pso.iterator()

    '''图像部分'''
    plt.figure(1)
    plt.title("linSVR")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    tt = np.array([t for t in range(0, iter1)])
    fitness = np.array(fitness)
    print(len(tt), len(fitness))
    plt.plot(tt, fitness, color='b', linewidth=3)
    plt.show()
