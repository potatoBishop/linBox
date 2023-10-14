#!/usr/bin/env python
# coding: utf-8

"""
作者：胖哥
微信公众号：胖哥真不错
微信号: zy10178083

为了防止大家在运行项目时报错(项目都是运行好的，报错基本都是版本不一致 导致的)，
胖哥把项目中用到的库文件版本在这里说明：

pandas == 1.1.5
matplotlib == 3.3.4
scikit-learn == 0.24.1
numpy == 1.19.5

"""

# 导入第三方库
from numpy.random import random as rand
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 进度条设置
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score


# =======定义目标函数=====
def calc_f(X):
    """计算粒子的的适应度值，也就是目标函数值 """
    A = 10
    pi = np.pi
    x = X[0]
    y = X[1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)


# ====定义惩罚项函数======
def calc_e(X):
    """计算蚂蚁的惩罚项，X 的维度是 size * 2 """
    ee = 0
    """计算第一个约束的惩罚项"""
    e1 = X[0] + X[1] - 6
    ee += max(0, e1)
    """计算第二个约束的惩罚项"""
    e2 = 3 * X[0] - 2 * X[1] - 5
    ee += max(0, e2)
    return ee


# ===定义子代和父辈之间的选择操作函数====
def update_best(parent, parent_fitness, parent_e, child, child_fitness, child_e, X_train, X_test, y_train, y_test):
    """
        针对不同问题，合理选择惩罚项的阈值。本例中阈值为0.1
        :param parent: 父辈个体
        :param parent_fitness:父辈适应度值
        :param parent_e    ：父辈惩罚项
        :param child:  子代个体
        :param child_fitness 子代适应度值
        :param child_e  ：子代惩罚项
        :return: 父辈 和子代中较优者、适应度、惩罚项
        """

    svr = SVR(kernel='linear', C=abs(parent[0]), gamma=abs(parent[1]) * 10).fit(X_train, y_train)  # 训练
    cv_accuracies = cross_val_score(svr, X_test, y_test, cv=3,
                                    scoring='r2')  # 交叉验证计算r方

    # 使错误率降到最低
    accuracies = cv_accuracies.mean()  # 取交叉验证均值
    fitness_value = 1 - accuracies  # 错误率 赋值 适应度函数值

    parent_e = parent_e + fitness_value  # 父辈适应度值
    child_e = child_e + fitness_value  # 子代适应度值

    # 规则1，如果 parent 和 child 都没有违反约束，则取适应度小的
    if parent_e <= 0.1 and child_e <= 0.1:
        if parent_fitness <= child_fitness:
            return parent, parent_fitness, parent_e
        else:
            return child, child_fitness, child_e
    # 规则2，如果child违反约束而parent没有违反约束，则取parent
    if parent_e < 0.1 and child_e >= 0.1:
        return parent, parent_fitness, parent_e
    # 规则3，如果parent违反约束而child没有违反约束，则取child
    if parent_e >= 0.1 and child_e < 0.1:
        return child, child_fitness, child_e
    # 规则4，如果两个都违反约束，则取适应度值小的
    if parent_fitness <= child_fitness:
        return parent, parent_fitness, parent_e
    else:
        return child, child_fitness, child_e


# =======================初始化参数==========================
m = 20  # 蚂蚁个数
G_max = 10  # 最大迭代次数
Rho = 0.9  # 信息素蒸发系数
P0 = 0.2  # 转移概率常数
XMAX = 2  # 搜索变量x最大值
XMIN = 1  # 搜索变量x最小值
YMAX = 0  # 搜索变量y最大值
YMIN = -1  # 搜索变量y最小值
step = 0.1  # 局部搜索步长
P = np.zeros(shape=(G_max, m))  # 状态转移矩阵
fitneess_value_list = []  # 迭代记录最优目标函数值


# =======================定义初始化蚂蚁群体位置和信息素函数==========================
def initialization():
    """
    :return: 初始化蚁群和初始信息素
    """
    X = np.zeros(shape=(m, 2))  # 蚁群 shape=(20, 2)
    Tau = np.zeros(shape=(m,))  # 信息素
    for i in range(m):  # 遍历每一个蚂蚁
        X[i, 0] = np.random.uniform(XMIN, XMAX, 1)[0]  # 初始化x
        X[i, 1] = np.random.uniform(YMIN, YMAX, 1)[0]  # 初始化y
        Tau[i] = calc_f(X[i])  # 计算信息素
    return X, Tau


# ===定义位置更新函数====
def position_update(NC, P, X, X_train, X_test, y_train, y_test):
    """
    :param NC: 当前迭代次数
    :param P: 状态转移矩阵
    :param X: 蚁群
    :return: 蚁群X
    """
    lamda = 1 / (NC + 1)
    # =======位置更新==========
    for i in range(m):  # 遍历每一个蚂蚁
        # ===局部搜索===
        if P[NC, i] < P0:
            temp1 = X[i, 0] + (2 * np.random.random() - 1) * step * lamda  # x转换到【-1,1】区间
            temp2 = X[i, 1] + (2 * np.random.random() - 1) * step * lamda  # y
        # ===全局搜索===
        else:
            temp1 = X[i, 0] + (XMAX - XMIN) * (np.random.random() - 0.5)
            temp2 = X[i, 0] + (YMAX - YMIN) * (np.random.random() - 0.5)

        # =====边界处理=====
        if (temp1 < XMIN) or (temp1 > XMAX):  # 判断
            temp1 = np.random.uniform(XMIN, XMAX, 1)[0]  # 初始化x
        if (temp2 < YMIN) or (temp2 > YMAX):  # 判断
            temp2 = np.random.uniform(YMIN, YMAX, 1)[0]  # 初始化y

        # =====判断蚂蚁是否移动(选更优）=====
        # ==子代蚂蚁==
        children = np.array([temp1, temp2])  # 子代个体蚂蚁
        children_fit = calc_f(children)  # 子代目标函数值
        children_e = calc_e(children)  # 子代惩罚项
        parent = X[i]  # 父辈个体蚂蚁
        parent_fit = calc_f(parent)  # 父辈目标函数值
        parent_e = calc_e(parent)  # 父辈惩罚项
        # 调用适用度函数
        pbesti, pbest_fitness, pbest_e = update_best(parent, parent_fit, parent_e, children, children_fit, children_e,
                                                     X_train, X_test, y_train, y_test)
        X[i] = pbesti
    return X  # 返回数据


# ======信息素更新============
def Update_information(Tau, X):
    """
    :param Tau: 信息素
    :param X: 蚂蚁群
    :return: Tau信息素
    """
    for i in range(m):  # 遍历每一个蚂蚁
        Tau[i] = (1 - Rho) * Tau[i] + calc_f(X[i])  # (1 - Rho) * Tau[i] 信息蒸发后保留的
    return Tau


# =============定义蚁群优化算法主函数======================
def aco(X_train, X_test, y_train, y_test):
    X, Tau = initialization()  # 初始化蚂蚁群X 和信息素 Tau
    for NC in tqdm(range(G_max)):  # 遍历每一代
        BestIndex = np.argmin(Tau)  # 最优索引
        Tau_best = Tau[BestIndex]  # 最优信息素
        # 计算状态转移概率
        for i in range(m):  # 遍历每一个蚂蚁
            P[NC, i] = np.abs((Tau_best - Tau[i])) / np.abs(Tau_best) + 0.01  # 即离最优信息素的距离
        # =======位置更新==========
        X = position_update(NC, P, X, X_train, X_test, y_train, y_test)  # 位置

        # =====更新信息素========
        Tau = Update_information(Tau, X)

        # =====记录最优目标函数值========
        index = np.argmin(Tau)  # 最小值索引
        value = Tau[index]  # 最小值
        fitneess_value_list.append(calc_f(X[index]))  # 记录最优目标函数值

    # =====打印结果=======
    min_index = np.argmin(Tau)  # 最优值索引
    minX = X[min_index, 0]  # 最优变量x
    minY = X[min_index, 1]  # 最优变量y
    minValue = calc_f(X[min_index])  # 最优目标函数值

    best_C = minX  # 赋值
    best_gamma = minY  # 赋值

    # =====可视化=======
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(fitneess_value_list, label='迭代曲线')
    plt.legend()  # 显示图例
    plt.show()  # 展示图片

    return best_C, best_gamma  # 返回C和gamma


if __name__ == '__main__':
    # 读取数据
    file_name = 'E:\Python\linSVR\8yin.xlsx'
    df = pd.read_excel(file_name)

    # 查看数据前5行
    print('*************查看数据前5行*****************')
    print(df.head())

    # 数据缺失值统计
    print('**************数据缺失值统计****************')
    print(df.info())

    # 描述性统计分析
    print(df.describe())
    print('******************************')

    # y变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = df['y']  # 过滤出y变量的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    plt.xlabel('y')
    plt.ylabel('数量')
    plt.title('y变量分布直方图')
    plt.show()

    # 数据的相关性分析
    import seaborn as sns

    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('相关性分析热力图')
    plt.show()

    # 提取特征变量和标签变量
    y = df.y
    X = df.drop('y', axis=1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 调用蚁群优化算法ACO
    best_C, best_gamma = aco(X_train, X_test, y_train, y_test)

    print('----------------蚁群优化算法ACO优化支持向量机回归模型-最优结果展示-----------------')
    print("The best C is " + str(abs(best_C)))
    print("The best gamma is " + str(abs(best_gamma) * 10))

    print('----------------应用优化后的最优参数值构建支持向量机回归模型-----------------')
    # 应用优化后的最优参数值构建支持向量机回归模型
    svr = SVR(kernel='linear', C=abs(best_C), gamma=abs(best_gamma) * 10)  # 建模
    svr.fit(X_train, y_train)  # 拟合
    y_pred = svr.predict(X_test)  # 预测

    print('----------------模型评估-----------------')
    # 模型评估
    print('**************************输出测试集的模型评估指标结果*******************************')

    print('支持向量机回归模型-最优参数-R^2：', round(r2_score(y_test, y_pred), 4))
    print('支持向量机回归模型-最优参数-均方误差:', round(mean_squared_error(y_test, y_pred), 4))
    print('支持向量机回归模型-最优参数-解释方差分:', round(explained_variance_score(y_test, y_pred), 4))
    print('支持向量机回归模型-最优参数-绝对误差:', round(mean_absolute_error(y_test, y_pred), 4))

    # 真实值与预测值比对图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")
    plt.plot(range(len(y_pred)), y_pred, color="red", linewidth=1.5, linestyle="-.")
    plt.legend(['真实值', '预测值'])
    plt.title("ACO蚁群优化算法优化支持向量机回归模型真实值与预测值比对图")
    plt.show()  # 显示图片