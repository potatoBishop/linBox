import numpy as np
import random
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVR

# filename = '7yinWu.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
filename = 'E:\linBox\data\\7yinWu.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
index=0
X_tra=[]
X_te=[]
y_tra=[]
y_te=[]
pos = []
res = []
x_min = 0
v_min = 0
x_max = 10000
v_max = 10000
T = 200
w_max = 0.8
w_min = 0.2

a = random.randint(1, 85)
b = random.randint(1, 85)
c = random.randint(1, 85)
d = random.randint(1, 85)
e = random.randint(1, 85)
f = random.randint(1, 85)
g = random.randint(1, 85)

with open(filename, 'r') as file_to_read:
  while True:
    lines = file_to_read.readline()  # 整行读取数据
    if not lines:
      break
      pass
    index=index+1

    print(a,b,c,d,e,f,g);
    if(index!=a and index!=b and index!=c  and index!=d and
index!=e and index!=f and index!=g):#选取测试集2
      strs = lines.split()
      X_tmp = []
      for x in range(0, len(strs) - 1):
       X_tmp.append(float(strs[x]))
      y_tmp = float(strs[len(strs) - 1])


      X_tra.append(X_tmp)
      y_tra.append(y_tmp)
    else:
      strs = lines.split()
      X_tmp = []
      for x in range(0, len(strs) - 1):
        X_tmp.append(float(strs[x]))
      y_tmp = float(strs[len(strs) - 1])
      X_te.append(X_tmp)
      y_te.append(y_tmp)

X = np.array(X_tra)

X_test=np.array(X_te)
y=np.array(y_tra)
y_test=np.array(y_te)

# #############################################################################
# Add noise to targets
def get_rmse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


class PSO():
    def __init__(self, pN, dim, max_iter):
        #定义所需变量
        self.w = 0.8
        self.c1 = 2#学习因子
        self.c2 = 2

        self.r1 = 0.6#超参数
        self.r2 = 0.3

        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数

        #定义各个矩阵大小
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度矩阵
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置矩阵
        self.gbest = np.ones((self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 0 # 全局最佳适应值       把1e10改成0，让R2做适应值，越高越好

        self.init_Population()

    #目标函数，根据使用场景进行设置
    def function(self, x):
        cc = x[0]
        gg = x[1]
        # print(cc,gg)
        svr = SVR(C=cc, gamma=gg, kernel='rbf', epsilon=0.1)
        svr.fit(X,y)
        score = svr.score(X,y)
        score1 = svr.score(X_test,y_test)
        # print("kernel='rbf'", 'C=', cc, 'gamma=', gg, 'epsilon=', 0.1)
        # print('R2(训练集)      ', svr.score(X, y))
        # print('R2(测试集)      ', svr.score(X_test, y_test))
        # 最下方的代码注释掉，sum作为适应度值返回，即R2
        if(score > 0.7):
            return score1
        # sum = 0
        # length = len(x)
        # for i in range(length):
        #     sum += (4*x[i]**3-5*x[i]**2+x[i]+6)**2-

    #初始化粒子群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 10000)

                self.V[i][j] = random.uniform(0, 10000)
            # print(self.X[i])
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            # if (tmp < self.fit):    注释掉，R2应该是越大越好
            if(tmp > self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):      # 迭代多少次
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])     # 得到一个粒子的score
                # if (temp < self.p_fit[i]):  # 更新个体最优  注释掉
                if (float(temp) > float(self.p_fit[i])):
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    # if (self.p_fit[i] < self.fit):  # 更新全局最优
                if(self.p_fit[i] > self.fit):
                    print("我被更新了")
                    self.gbest = self.X[i].copy()
                    self.fit = self.p_fit[i]
                self.w = w_max - (w_max - w_min) * i / T
            for i in range(self.pN):
                #粒子群算法公式   更新每一个粒子的速度和位置
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                # 边界条件，粒子的速度和位置不能超过边界值
                for j in range(self.dim):
                    if(self.V[i][j] > v_max) or (self.V[i][j] < v_min):
                        self.V[i][j] = v_min + random.uniform(0,1) * (v_max - v_min)
                    if (self.X[i][j] > x_max) or (self.X[i][j] < x_min):
                        self.X[i][j] = x_min + random.uniform(0,1) * (x_max - x_min)
            fitness.append(self.fit)
            # print('R2(测试集)      ',self.fit)  # 输出最优值
            # print('最佳参数为',self.gbest)
            svr = SVR(C=self.gbest[0],gamma=self.gbest[1],kernel='rbf',epsilon=0.1)
            svr.fit(X,y)
            # 每一代的最佳测试集R2及最佳参数
            print('R2(训练集)      ', svr.score(X, y))
            print('R2(测试集)      ', svr.score(X_test, y_test))
            print('RMSE(训练集)      ', get_rmse(y, svr.predict(X)))
            print('RMSE(测试集)      ', get_rmse(y_test, svr.predict(X_test)))
            print('最优值R2(测试集)      ',self.fit)  # 输出最优值
            print('最佳参数为',self.gbest)
        return fitness

if __name__ == '__main__':
    # my_pso = PSO(pN=30, dim=5, max_iter=100)
    my_pso = PSO(pN=100,dim=2,max_iter=300)
    fitness = my_pso.iterator()
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 200)])
    fitness = np.array(fitness)
    print(len(t),len(fitness))
    plt.plot(t, fitness, color='b', linewidth=3)
    plt.show()