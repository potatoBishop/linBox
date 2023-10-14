import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel 
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime 
from sklearn.metrics import explained_variance_score
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error # 平方绝对误差
import random

x1=0
x2=0
x3=0
x4=0



a = random.randint(1, 38);
b = random.randint(1, 38);
c = random.randint(1, 38);
d = random.randint(1, 38)
e = random.randint(1, 38)
f = random.randint(1, 38)
g = random.randint(1, 38)
#x5=0
# rbf核函数
def rbf_kernel(X, Y=None, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)
    return K

# poly核函数
def safe_sparse_dot(a, b, *, dense_output=False):
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sparse.issparse(a) and sparse.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret

def polynomial_kernel(X, Y=None, degree=2, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    K **= degree
    return K
def mix_kernel(X, Y=None, degree=3, gamma=x3, coef0=0):
    print(x1,x2,x3,x4)
    return rbf_kernel(X, Y, gamma) * x4 + polynomial_kernel(X, Y, degree, gamma, coef0) * (1-x4)

filename = '/data/gaozhan.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
index=0
X_tra=[]
X_te=[]
y_tra=[]
y_te=[]
pos = []
res = []
y_pre=[]
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()  # 整行读取数据
        if not lines:
            break
            pass
        index=index+1
        print(a, b, c, d, e, f, g);
        if (index != a and index != b and index != c and index != d and
                index != e and index != f and index != g):  # 选取测试集2
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
            print(index,y_tmp)
            X_te.append(X_tmp)
            y_te.append(y_tmp)
X = np.array(X_tra)
X_test=np.array(X_te)
y=np.array(y_tra)
y_test=np.array(y_te)


feature_train, feature_test, target_train, target_test = X, X_test, y, y_test

class PSO:
    def __init__(self, parameters):
        # 初始化
        self.NGEN = parameters[0]    # 迭代的代数
        self.pop_size = parameters[1]    # 种群大小
        self.var_num = len(parameters[2])     # 变量个数
        self.bound = []                 # 变量的约束范围
        self.bound.append(parameters[2]) #parameters[2] 变量下限
        self.bound.append(parameters[3]) #parameters[3] 变量上限
        print(1)
        print(self.bound[0])
        print(self.bound[1])
        self.pop_x = np.zeros((self.pop_size, self.var_num))    # 所有粒子的位置       每个粒子的5个维度设0
        self.pop_v = np.zeros((self.pop_size, self.var_num))    # 所有粒子的速度       每个粒子的5个维度设0
        self.p_best = np.zeros((self.pop_size, self.var_num))   # 每个粒子最优的位置     
        self.g_best = np.zeros((1, self.var_num))               # 全局最优的位置 
        print(self.p_best)
        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pop_size):              #第一层循环，遍历每一个粒子
            for j in range(self.var_num):           #第二层循环，遍历每一个粒子随机赋值
                self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]      # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit
    
    #计算个体适应值
    def fitness(self,ind_var):
        X = feature_train
        y = target_train
        """
        个体适应值计算
        """#五个维度依次 为C，e, g, w, d
        global x1 
        x1= ind_var[0]
        global x2 
        x2= ind_var[1]
        global x3 
        x3= ind_var[2]
        global x4
        x4 = ind_var[3]

        if x1==0:x1=0.01
        if x2==0:x2=0.01
        if x3==0:x3=0.01
        if x4==0:x4=0.1   
        #if x5==0:x5=1
        def mix_kernel(X, Y=None, degree=2, gamma=x3, coef0=x2):
            return rbf_kernel(X, Y, gamma) * x4 + polynomial_kernel(X, Y, degree, gamma, coef0) * (1 -x4)
        clf = SVR(C = x1,gamma=x3,kernel=mix_kernel,epsilon=0.1)
        clf.fit(X, y)
        t=clf.score(X,y)
        predictval=clf.predict(feature_test)
        predictval_train=clf.predict(feature_train)
        print("训练集R2 = ",metrics.r2_score(target_train,predictval_train),"  测试集R2 = ",metrics.r2_score(target_test,predictval))
        # print("训练集R2 = ",metrics.r2_score(target_train,predictval_train),"  测试集R2 = ",metrics.r2_score(target_test,predictval)) # R2
        return  metrics.r2_score(target_test,predictval)

    def update_operator(self, pop_size):
        """
        更新算子：更新下一时刻的位置和速度
        """
        c1 = 2     # 学习因子，一般为2
        c2 = 2
        w = 0.5    # 自身权重因子
        for i in range(pop_size): #遍历每一个粒子
            # 更新速度
            self.pop_v[i] = (w * self.pop_v[i] +
                             c1 * random.uniform(0, 1) * (self.p_best[i] - self.pop_x[i]) +
                             c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i]))
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = 0.1
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
 
    def main(self):
        popobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        for gen in range(self.NGEN):   #每一代依次增大
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()
            print('最好的位置：{}'.format(self.ng_best))
            print('最大的函数值：{}'.format(self.fitness(self.ng_best)))
        '''
        bestc = self.ng_best[0]
        besteplison = self.ng_best[1]
        bestg = self.ng_best[2]
        '''
        print("---- End of (successful) Searching ----")
 
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        #plt.title("Figure1")
        plt.xlabel("iterators", size=10)
        plt.ylabel("fitness", size=10)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()
if __name__ == '__main__':
    NGEN = 100
    popsize = 50
    #五个维度依次 为C，e, g, w, d
    low = [0,   0,  0,   0]
    up  = [170, 1, 10, 1.0]
    parameters = [NGEN, popsize, low, up]
    pso = PSO(parameters)
    pso.main()
