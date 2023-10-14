import math

import numpy
import numpy as np
from sklearn.svm import SVR
# from numba import jit

class LinKernel:
    """ 存放核函数方法 """
    # 参数
    core_kind = 0   # 核模型选择

    gamma = 0.5     # 惩罚间隔
    degree = 4      # 多项式次数
    cc = 0          # 惩罚系数
    v = 1.0         # 忘了
    q = 0.5         # 三核混合比例 and 双核混合比例
    t = 0           # 三核混合比例
    # a = 1           # sigmoid 参数
    # b = 0           # sigmoid 参数
    coef = 0




    # get函数和set函数
    def set_tt(self, tt):
        self.t = tt

    def set_qq(self, qq):
        self.q = qq

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_degree(self, degree):
        self.degree = degree

    def set_v(self, v):
        self.v = v

    def set_cc(self, c):
        self.cc = c

    def set_coef(self, coef):
        self.coef = coef

    def input_settings(self, d, qq, tt, c, g, co):  # dd qq tt cc gg coco
        self.degree = d
        self.cc = c
        self.q = qq
        self.t = tt
        self.gamma = g
        self.coef = co

    def set_core_kind(self, kind):
        self.core_kind = kind

    # ============================================================================================ 经典核函数方法
    # @jit
    def my_rbf(self, X1, X2):                                                           # 高斯核 已核对完成，并提高计算速度
        m, n = X1.shape[0], X2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for i in range(m):
            for j in range(n):
                K[i][j] = math.exp(-self.gamma * (np.linalg.norm(X1[i] - X2[j])) ** 2)
        return K

    @staticmethod
    def my_linear(X1, X2):                                                                # 线性 已核对完成
        m, n = X1.shape[0], X2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for i in range(m):
            for j in range(n):
                K[i][j] = numpy.dot(X1[i], X2[j])
        return K

    def my_sigmoid(self, X1, X2):                                                        # sigmoid 已核对完成
        m, n = X1.shape[0], X2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for j in range(m):
            for i in range(n):
                K[j][i] = math.tanh(self.gamma * numpy.dot(X1[j].T, X2[i]) + self.coef)
        return K


    def my_poly(self, x1, x2):                                                           # 多项式 已核对完成
        # coef0 = -1 / (2 * self.gamma ** 2)
        m, n = x1.shape[0], x2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for i in range(m):
            for j in range(n):
                K[i][j] = ( self.gamma * numpy.dot(x1[i], x2[j]) + self.coef ) ** self.degree
        return K

# ============================================================================================混合核函数区域、
    def my_kernel1(self, X1, X2):   # rbf + poly
        return self.q * self.my_rbf(X1, X2) + (1 - self.q) * self.my_poly(X1, X2)

    def my_kernel2(self, X1, X2):   # rbf + linear
        return self.q * self.my_rbf(X1, X2) + (1 - self.q) * self.my_poly(X1, X2)

    def my_kernel3(self, X1, X2):   # rbf + linear + poly
        return self.q * self.my_rbf(X1, X2) \
               + self.t * self.my_linear(X1, X2) \
               + (1 - self.q - self.t) * self.my_poly(X1, X2)

    def my_kernel4(self, X1, X2):   # rbf + sigmoid + poly
        return self.q * self.my_rbf(X1, X2) \
               + self.t * self.my_sigmoid(X1, X2) \
               + (1 - self.q - self.t) * self.my_poly(X1, X2)

    def my_kernel5(self, X1, X2):   # rbf + sigmoid + poly
        return self.q * self.my_rbf(X1, X2) \
               + self.t * self.my_sigmoid(X1, X2) \
               + (1 - self.q - self.t) * self.my_poly(X1, X2)

    def do_svr(self, x_tra, y_tra):
        """训练集x， 测试集x, 训练集y, 测试集y, 惩罚系数, 阶数, gamma, 不知名参数q """
        # x = np.array(x_tra)  # 处理完成的训练集
        # y = np.array(y_tra)
        svr_hunhe = SVR

        if self.core_kind == 0:     # rbf + sigmoid
            svr_hunhe = SVR(kernel=self.my_kernel1, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 1:   # rbf + poly
            svr_hunhe = SVR(kernel=self.my_kernel2, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 2:   # rbf + linear + poly
            svr_hunhe = SVR(kernel=self.my_kernel3, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 3:   # rbf + sigmoid + poly
            svr_hunhe = SVR(kernel=self.my_kernel4, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 4:
            svr_hunhe = SVR(kernel='sigmoid', C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 5:
            svr_hunhe = SVR(kernel=self.my_kernel5, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "rbf":
            svr_hunhe = SVR(kernel=self.my_rbf, C=self.cc, gamma=self.gamma)
        else:
            pass



        svr_hunhe.fit(x_tra, y_tra)
        return svr_hunhe


    def just_svr(self):
        """训练集x， 测试集x, 训练集y, 测试集y, 惩罚系数, 阶数, gamma, 不知名参数q """
        # x = np.array(x_tra)  # 处理完成的训练集
        # y = np.array(y_tra)
        svr_hunhe = SVR

        if self.core_kind == 0:     # rbf + sigmoid
            svr_hunhe = SVR(kernel=self.my_kernel1, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 1:   # rbf + poly
            svr_hunhe = SVR(kernel=self.my_kernel2, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 2:   # rbf + linear + poly
            svr_hunhe = SVR(kernel=self.my_kernel3, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 3:   # rbf + sigmoid + poly
            svr_hunhe = SVR(kernel=self.my_kernel4, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 4:
            svr_hunhe = SVR(kernel='sigmoid', C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 5:
            svr_hunhe = SVR(kernel=self.my_kernel5, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "rbf":
            svr_hunhe = SVR(kernel=self.my_rbf, C=self.cc, gamma=self.gamma)
        else:
            pass

        return svr_hunhe
