import numpy as np
# import function as fun
import m2_DBO_v1 as fun1
import sys
import matplotlib.pyplot as plt

# 前置知识点
# [-1]：   列表最后一项
# [:-1]：  从第一项到最后一项
# [::-1]： 代表从全列表倒序取
# np.argmin 展平 并 输出最小值的下标

# x[:,0]
# #　二维数组取第１维所有数据
# x[:,1]
# # 第２列
# x[0,:]
# # 第１行
# x[3,:]
# # 第4行
# x[1:4,:]
# # 第一二三行

def main(argv):
    SearchAgents_no = 30    # 种群粒子数   不要小于30
    Function_name = 'm3'        # 名字
    Max_iteration = 200    # 最大迭代次数

    # 根据名字获取所对应的  适应度函数、下边界、上边界、维度数量
    [fobj, lb, ub, dim] = fun1.Parameters(Function_name)

    # 粒子群数目，最大迭代次数   ，下界，上界，维度数，适应度函数
    [fMin, bestX, DBO_curve] = fun1.DBO(SearchAgents_no, Max_iteration, lb, ub, dim)
    print(['最优值为：  ', fMin])
    print(['最优变量为：', bestX])

    # 画图部分
    thr1 = np.arange(len(DBO_curve[0, :]))
    plt.plot(thr1, DBO_curve[0, :])
    plt.xlabel('num')
    plt.ylabel('object value')
    plt.title('line')
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
