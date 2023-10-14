import geatpy as ea
import numpy as np

# 构建问题
r = 1  # 目标函数需要用到的额外数据
# from sklearn.svm import SVR


@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    f = np.sum((Vars - r) ** 2)  # 计算目标函数值
    x1 = Vars[0]
    x2 = Vars[1]
    CV = np.array([(x1 - 0.5) ** 2 - 0.25,
                   (x2 - 1) ** 2 - 1])  # 计算违反约束程度
    return f, CV


problem = ea.Problem(name='soea quick start demo',
                     M=1,  # 目标维数
                     maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                     Dim=5,  # 决策变量维数
                     varTypes=[0, 0, 1, 1, 1],  # 决策变量的类型列表，0：实数；1：整数
                     lb=[-1, 1, 2, 1, 0],  # 决策变量下界
                     ub=[1, 4, 5, 2, 1],  # 决策变量上界
                     evalVars=evalVars)
# 构建算法
algorithm = ea.moea_NSGA2_templet(problem,
                                  ea.Population(Encoding='RI', NIND=50),
                                  MAXGEN=200,  # 最大进化代数
                                  logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
# 求解
res = ea.optimize(algorithm,
                  seed=1,
                  verbose=False,
                  drawing=1,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=False,
                  dirName='result')
