import sys
import linData
import linSVR
import linTools


def test1(quick, core_kind, filename, dataSize, numNeed):
    kk = linSVR.LinKernel()
    zu = linData.dataHandle()

    kk.set_core_kind(core_kind)
    dd   = quick[0]   # degree
    cc   = quick[1]   # 惩罚系数
    gg   = quick[2]
    coco = quick[3]
    qq   = quick[4]   # 混合核函数内的比例系数
    tt   = quick[5]
    ee   = 0.1        # epsilon 判断是否进行惩罚

    # 设置模型
    # 设置最优参数
    kk.set_degree(dd)
    kk.set_cc(cc)
    kk.set_qq(qq)
    kk.set_tt(tt)
    kk.set_gamma(gg)
    kk.set_coef(coco)

    # 获取训练与测试集合
    zu.set_file(filename)   # 文件名
    zu.set_mode(1)          # 0:均匀选择4b1   1:随机选择   2: 留1法
    zu.set_dataSize(dataSize)         # 总样本数
    zu.set_numNeed(numNeed)

    sum_r2_train = 0
    sum_r2_test = 0
    sum_RMSE_train = 0
    sum_RMSE_test = 0

    max_score = -1000   # 最好分数
    count = 0           # 循环次数
    ci = 80
    num = 0

    while num < ci:
        num = num + 1
        zu.fenZu()              # 分组并获取训练与测试集合
        x_tra = zu.get_x()
        y_tra = zu.get_y()
        x_test = zu.get_x_test()
        y_test = zu.get_y_test()

        # 查看结果
        print("第", num, "次随机训练")
        final_svr = kk.do_svr(x_tra, y_tra)

        r2_train   = final_svr.score(x_tra, y_tra)
        r2_test    = final_svr.score(x_test, y_test)
        RMSE_train = linTools.get_RMSE(y_tra, final_svr.predict(x_tra))
        RMsE_test   = linTools.get_RMSE(y_test, final_svr.predict(x_test))

        print('训练集R2        ', r2_train)
        print('测试集R2        ', r2_test)
        print('RMSE(训练集)    ', RMSE_train)
        print('RMSE(测试集)    ', RMsE_test)

        sum_r2_train   = sum_r2_train   + r2_train
        sum_r2_test    = sum_r2_test    + r2_test
        sum_RMSE_train = sum_RMSE_train + RMSE_train
        sum_RMSE_test  = sum_RMSE_test  + RMsE_test

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("参数")
    print("degree:", dd, " C:", cc, " qq:", qq, " tt:", tt, " gamma:", gg, " coef0:", coco)
    print("训练集 R2 均值  ", sum_r2_train/ci)
    print("测试集 R2 均值  ", sum_r2_test/ci)
    print("训练集 RMSE 均值  ", sum_RMSE_train/ci)
    print("测试集 RMSE 均值  ", sum_RMSE_test/ci)

