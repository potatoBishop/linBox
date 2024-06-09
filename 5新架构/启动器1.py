import numpy as np
import DBO as fun
import sys
import matplotlib.pyplot as plt
import linSVR
import linData
import linTools

def main(argv):
    # filename = 'E:\linBox\data\\7引物ci2.txt'  # 读取数据
    # filename = 'E:\linBox\data\\7yinWu.txt'  # 读取数据
    # filename = 'E:\linBox\data\\gaozhan.txt'  # 读取数据
    filename = 'E:\\1Python\\linBox\\data\\result_guiyihua.txt'  # 读取数据
    searchAgents_no = 50                       # 种群粒子数   不要小于30
    function_name = 'm2'                        # 核函数 名字 or 代号
    # function_name = 'gaozhan'                        # 核函数 名字 or 代号

    max_iteration = 100                         # 最大迭代次数
    dim = 4                                     # 维度数量
    dd  = 2                                     # degree
    # lb  = [0.001, 0.001, -200, 0.15]            # 下界
    # ub  = [180,   100,    300, 0.85]            # 上界
    lb  = [0.001, 0.00001, -20, 0.15]            # 下界
    ub  = [500,   200,    30, 0.85]            # 上界
    #      c      g     co   q

    [fMin, bestX, DBO_curve] = fun.DBO(searchAgents_no,
                                       max_iteration,
                                       filename,
                                       dim,
                                       lb,
                                       ub,
                                       function_name,
                                       dd)
    print(['最优值为：  ', fMin])
    print(['最优变量为：', "dd=", dd, bestX])

    # 画图部分
    thr1 = np.arange(len(DBO_curve[0, :]))
    plt.plot(thr1, DBO_curve[0, :])
    plt.xlabel('num')
    plt.ylabel('object value')
    plt.title('line')
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
