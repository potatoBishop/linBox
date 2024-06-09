import math

import numpy
import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from sklearn.svm import SVR


# from numba import jit
def rbf_kernel(X, Y=None, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)
    return K
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
class LinKernel:
    """ 存放核函数方法 """
    # 参数
    core_kind = 0  # 核模型选择

    gamma = 0.5  # 惩罚间隔
    degree = 2  # 多项式次数
    cc = 0  # 惩罚系数
    v = 1.0  # 忘了
    q = 0.5  # 三核混合比例 and 双核混合比例
    t = 0  # 三核混合比例
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

    def input_settings(self, d=1, qq=1, tt=1, c=1, g=1, co=0):  # dd qq tt cc gg coco
        self.degree = d
        self.cc     = c
        self.q      = qq
        self.t      = tt
        self.gamma  = g
        self.coef   = co

    def input_settings2(self, c=1, g=1, d=1, co=1, q=1, t=1):  # cc gamma degree coef0 qq tt
        self.cc     = c
        self.gamma  = g
        self.degree = d
        self.coef   = co
        self.q      = q
        self.t      = t


    # ============================================================================================ 经典核函数方法
    # @jit

    def my_rbf(self, X1, X2):  # 高斯核 已核对完成，并提高计算速度
        m, n = X1.shape[0], X2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for i in range(m):
            for j in range(n):
                K[i][j] = math.exp(-self.gamma * (np.linalg.norm(X1[i] - X2[j])) ** 2)
        return K

    @staticmethod
    def my_linear(X1, X2):  # 线性 已核对完成
        m, n = X1.shape[0], X2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for i in range(m):
            for j in range(n):
                K[i][j] = numpy.dot(X1[i], X2[j])
        return K

    def my_sigmoid(self, X1, X2):  # sigmoid 已核对完成
        m, n = X1.shape[0], X2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for j in range(m):
            for i in range(n):
                K[j][i] = math.tanh(self.gamma * numpy.dot(X1[j].T, X2[i]) + self.coef)
        return K

    def my_poly(self, x1, x2):  # 多项式 已核对完成
        # coef0 = -1 / (2 * self.gamma ** 2)
        m, n = x1.shape[0], x2.shape[0]  # 获取行数
        K = np.zeros((m, n), dtype=float)  # 全零核矩阵
        for i in range(m):
            for j in range(n):
                K[i][j] = (self.gamma * numpy.dot(x1[i], x2[j]) + self.coef) ** self.degree
        return K

    # ============================================================================================混合核函数区域、
    def my_kernel1(self, X1, X2):  # rbf + poly
        return self.q * self.my_rbf(X1, X2) + (1 - self.q) * self.my_poly(X1, X2)

    def my_kernel2(self, X1, X2):  # rbf + poly
        return self.q * self.my_rbf(X1, X2) + (1 - self.q) * self.my_poly(X1, X2)

    def my_kernel3(self, X1, X2):  # rbf + linear + poly
        return self.q * self.my_rbf(X1, X2) \
            + self.t * self.my_linear(X1, X2) \
            + (1 - self.q - self.t) * self.my_poly(X1, X2)

    def my_kernel4(self, X1, X2):  # rbf + sigmoid + poly
        return self.q * self.my_rbf(X1, X2) \
            + self.t * self.my_sigmoid(X1, X2) \
            + (1 - self.q - self.t) * self.my_poly(X1, X2)

    def my_k5(self, X1, X2):
        return self.q * self.my_rbf(X1, X2) \
            + (1 - self.q) * self.my_poly(X1, X2) \
            # + (1 - self.q) * self.my_poly(X1, X2)

    def my_m3(self, X1, X2):
        return self.q * self.my_sigmoid(X1, X2) \
            + (1 - self.q) * self.my_poly(X1, X2) \
            # + (1 - self.q) * self.my_poly(X1, X2)

    def my_m4(self, X1, X2):
        return self.q * self.my_rbf(X1, X2) \
            + (1 - self.q) * self.my_sigmoid(X1, X2)

    def my_m5(self, X1, X2):
        return self.q * self.my_rbf(X1, X2) \
            + (1 - self.q) * self.my_poly(X1, X2) \
            # + (1 - self.q) * self.my_poly(X1, X2)

    def mix_kernel(self, X, Y=None):

        return rbf_kernel(X, Y, self.gamma) * self.q + polynomial_kernel(X, Y, self.degree, self.gamma, self.coef) * (1 - self.q)

    def my_k6(self, X1, X2):
        return self.my_rbf(X1, X2) * self.my_poly(X1, X2)

    def set_core_kind(self, kind):
        self.core_kind = kind
        print("当前核心kind= ", self.core_kind)

    def just_svr(self):
        """训练集x， 测试集x, 训练集y, 测试集y, 惩罚系数, 阶数, gamma, 不知名参数q """
        svr_hunhe = SVR
        temp_function = self.my_k6

        if self.core_kind == "m1" or "rbf+sigmoid":  # rbf + sigmoid
            svr_hunhe = SVR(kernel=self.my_kernel1, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "m2" or "rbf+poly":  # rbf + poly
            svr_hunhe = SVR(kernel=self.my_kernel2, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "m3":                # sigmoid + poly
            svr_hunhe = SVR(kernel=self.my_m3, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "m4":                # sigmoid + rbf
            svr_hunhe = SVR(kernel=self.my_m4, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "m5":                # sigmoid + rbf
            svr_hunhe = SVR(kernel=self.my_m5, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "gaozhan":                # sigmoid + rbf
            svr_hunhe = SVR(kernel=self.mix_kernel, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        # 其他组合
        elif self.core_kind == 3 or "rbf+linear+poly":  # rbf + linear + poly
            svr_hunhe = SVR(kernel=self.my_kernel3, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 4 or "rbf+sigmoid+poly":  # rbf + sigmoid + poly
            svr_hunhe = SVR(kernel=self.my_kernel4, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 5:
            svr_hunhe = SVR(kernel=self.my_k5, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == 6:
            svr_hunhe = SVR(kernel=temp_function, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
         #     经典核心
        elif self.core_kind == "t3" or "rbf":
            svr_hunhe = SVR(kernel=self.my_rbf, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "t4" or "sigmoid":
            svr_hunhe = SVR(kernel=self.my_sigmoid, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "t2" or "poly":
            svr_hunhe = SVR(kernel=self.my_poly, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        elif self.core_kind == "t1" or "linear":
            svr_hunhe = SVR(kernel=self.my_linear, C=self.cc, gamma=self.gamma, degree=self.degree, coef0=self.coef)
        else:
            print("核函数匹配失败")
            pass

        return svr_hunhe

    def do_svr(self, x_tra, y_tra):
        svr_hunhe = self.just_svr()
        svr_hunhe.fit(x_tra, y_tra)
        return svr_hunhe

    def chaKan(self):
        print("dd , qq, tt, cc, gg, coco")
        print(self.degree, self.q, self.t, self.cc, self.gamma, self.coef)
