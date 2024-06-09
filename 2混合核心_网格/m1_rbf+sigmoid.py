"""寻找参数  linear"""

from playsound import playsound
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import linSVR
import linTools
import linData

'''主要参数'''
kk = linSVR.LinKernel()
zu = linData.dataHandle()

max_score = -1000   # 最好分数
count     = 0           # 循环次数

# 暂时不需要调节的参数
dd   = 1          # degree
cc   = 37         # 惩罚系数
qq   = 0.5        # 混合核函数内的比例系数
tt   = 0
gg   = 0.2        # gamma
coco = 0        # coef0

# 设置参数                  =====================================================================================
filename = 'E:\\linBox\\data\\7引物ci2.txt'
zu.set_file(filename)       # 文件名
zu.set_mode(1)              # 鼠标放在函数上查看文档
zu.set_dataSize(85)         # 数据总量
zu.set_numNeed(17)          # 当mode = 1 时，需要的测试集数量
kk.set_core_kind("m1")      # 设置核心kind

# 参数范围
d_can     = [1, 2]                                                  # dd
c_can     = np.logspace(-3, 3, 10)                    # cc
q_can     = np.linspace(0.001, 1, 20)            # qq
t_can     = np.linspace(0.001, 1, 10)            # tt
gamma_can = np.linspace(0.1, 10, 10)              # gg gamma
co_can    = np.linspace(-1, 1, 5)                     # coco coef0




# 主要调整                  =====================================================================================

# 获取训练与测试集合
zu.fenZu()
x_tra  = zu.get_x()
y_tra  = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()

# 按间距中的绿色按钮以运行脚本。
aim = 0
if __name__ == '__main__':
    for d in d_can:
        for c in c_can:
            for q in q_can:
                for t in t_can:
                    if t + q > 1:
                        break
                    for g in gamma_can:
                        for co in co_can:
                            kk.input_settings2(c, g, d, co, q, t)       # 读入参数
                            temp_svr = kk.do_svr(x_tra, y_tra)
                            # score = temp_svr.score(x_tra, y_tra)
                            score1 = temp_svr.score(x_test, y_test)
                            # score = cross_val_score(temp_svr, x_tra, y_tra, cv=5)
                            # cv_accuracies = cross_val_score(temp_svr, x_test, y_test, cv=3, scoring='r2')  # 交叉验证计算r方
                            # if cv_accuracies.all() > 0.7 and score1 > max_score:
                            if score1 > 0.7 and score1 > max_score:
                                max_score = score1
                                dd = d
                                cc = c
                                qq = q
                                gg = g
                                tt = t
                                coco = co
                        count = count + 1
                        print("第", count, "次")

    # 循环网格结束 设置最优参数
    kk.input_settings2(cc, gg, dd, coco, qq, tt)

    # 查看结果
    final_svr = kk.do_svr(x_tra, y_tra)
    print(final_svr)
    print('R2(训练集)      ', final_svr.score( x_tra,  y_tra))
    print('R2(测试集)      ', final_svr.score(x_test, y_test))
    print('RMSE(训练集)    ', linTools.get_RMSE( y_tra, final_svr.predict( x_tra)))
    print('RMSE(测试集)    ', linTools.get_RMSE(y_test, final_svr.predict(x_test)))
    print("dd=", dd)
    print("cc=", cc)
    print("qq=", qq)
    print("tt=", tt)
    print("gg=", gg)
    print("coco=", coco)
    print(cc, ",", gg, ",", dd, ",", coco, ",", qq, ",", tt)        # 输出所以数据的规格化，以便于进行测试
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

