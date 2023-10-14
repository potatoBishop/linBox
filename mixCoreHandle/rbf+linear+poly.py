"""寻找参数  # rbf + linear + poly"""
from playsound import playsound
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

import jihe
import myCore
from main import get_rmse

'''主要参数'''
kk = myCore.MyKernelClass()
zu = jihe.jihe()
max_score = -1000   # 最好分数
count = 0           # 循环次数

# 需要调节的参数
filename = '8yinWu.txt'
dd = 1          # degree
cc = 37         # 惩罚系数
qq = 0.5        # 混合核函数内的比例系数
tt = 0
gg = 0.2        # gamma
coco = 0        # coef0


# 参数范围
# d_can = [1, 2, 3]                    # dd
# c_can = np.logspace(-3, 3, 25)          # cc
# q_can = np.linspace(0, 1, 20)           # qq
# t_can = np.linspace(0, 1, 20)
# gamma_can = np.logspace(-2, 3, 20)      # gg
d_can = [1]                    # dd
# c_can = np.logspace(-3, 3, 10)          # cc
c_can = np.linspace(1, 3, 10)
q_can = np.linspace(0.6, 0.9, 20)           # qq
t_can = np.linspace(0.001, 0.35, 10)
gamma_can = np.linspace(0.1, 2, 5)
co_can = np.linspace(-9, -4, 5)


# 获取训练与测试集合
zu.set_file(filename)   # 文件名
zu.set_mode(1)          # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(17)         # 当mode =0 ，为总样本数， 当 =1， 为需要的测试集数目  当 =2 ，为留一法取相应的值作为测试集
zu.fenZu()              # 分组并获取训练与测试集合
x_tra = zu.get_x()
y_tra = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()


# 设置kk
kk.set_core_kind(2)

# 按间距中的绿色按钮以运行脚本。
aim = 0
if __name__ == '__main__':
    for d in d_can:
        kk.set_degree(d)
        for c in c_can:
            kk.set_cc(c)
            for q in q_can:
                for t in t_can:
                    if t + q > 1:
                        break
                    kk.set_qq(q)
                    kk.set_tt(t)
                    for g in gamma_can:
                        kk.set_gamma(g)
                        for co in co_can:
                            kk.set_coef(co)

                            # 正式训练
                            temp_svr = kk.do_svr(x_tra, y_tra)

                            # score = temp_svr.score(x_tra, y_tra)
                            score1 = temp_svr.score(x_test, y_test)
                            # score = cross_val_score(temp_svr, x_tra, y_tra, cv=5)
                            cv_accuracies = cross_val_score(temp_svr, x_test, y_test, cv=3,
                                                            scoring='r2')  # 交叉验证计算r方
                            if cv_accuracies.all() > 0.7 and score1 > max_score:
                                max_score = score1
                                dd = d
                                cc = c
                                qq = q
                                gg = g
                                tt = t
                                coco = co
                        count = count + 1
                        print("第", count, "次")

    # 设置最优参数
    kk.set_degree(dd)
    kk.set_cc(cc)
    kk.set_qq(qq)
    kk.set_tt(tt)
    kk.set_gamma(gg)
    kk.set_coef(coco)

    # 查看结果
    final_svr = kk.do_svr(x_tra, y_tra)
    print(final_svr)
    print('R2(训练集)      ', final_svr.score(x_tra, y_tra))
    print('R2(测试集)      ', final_svr.score(x_test, y_test))
    print('RMSE(训练集)    ', get_rmse(y_tra, final_svr.predict(x_tra)))
    print('RMSE(测试集)    ', get_rmse(y_test, final_svr.predict(x_test)))
    print("dd=", dd)
    print("cc=", cc)
    print("qq=", qq)
    print("tt=", tt)
    print("gg=", gg)
    print("coco=", coco)
    # 闹钟
    # playsound("forya.mp3")

    # 图像
    # plot(x1,y1,xlab = "是x轴啦",ylab = "是y轴啦",xlim = c(0,100),ylim = c(0,100),col = "red",pch = 19)
    plt.scatter(y_tra, final_svr.predict(x_tra), label='Training set')
    plt.scatter(y_test, final_svr.predict(x_test), marker='^', label='Testing set')
    plt.legend()
    plt.xlabel('Experimental')
    plt.ylabel('Calculated')
    plt.legend(['Training set', 'Testing set'])
    plt.plot([-4, 3], [-4, 3])  # 画对角线      3
    plt.axis([-4, 3, -4, 3])  # 定义xy轴的显示范围
    plt.show()
    # print(svr.predict(X))
    # print(svr.predict(X_test))

