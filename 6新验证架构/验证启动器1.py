import sys
import 验证myData as t1
import 验证myData_k折 as t2
import 验证myData_留一 as t3



def main(argv):
    # quick = [1, 10.23774114,   2.15705083, -93.49955121,   0.70530699, 1]
    # quick = [2, 0.3886434 ,   0.41908116, 120.1688884 ,   0.77651648, 1]   dd = 2   m5
    # quick = [2, 4.24089776,   0.80674889, 427.04708595,   0.81608511, 1]    #  0.6

    # quick = [1, 0.70901997, 1.94672489, 0.70673982, 0.54556722, 1]  # 0.8  0.7   111
    # quick = [3, 1.45088386e+02,  8.74734174e-02, -3.91465284e+00,  7.84727746e-01, 1]  # 0.8  0.7   111
    quick = [3,9.75883169e+01, 5.78003558e-02, 3.00000000e+01, 5.07383384e-01, 1]  # 0.8  0.7   111

    # core_kind = "m3"
    # core_kind = "m4"
    # core_kind = 'm2'                        # 核函数 名字 or 代号
    core_kind = 'gaozhan'                        # 核函数 名字 or 代号
    # filename = 'E:\linBox\data\\7引物ci2.txt'  # 读取数据
    # filename = 'E:\linBox\data\\7yinWu.txt'  # 读取数据
    filename = 'E:\linBox\data\\gaozhan.txt'  # 读取数据

    # ===========================================
    select_kind = 2 
    dataSize = 70
    numNeed = 14
    # ===========================================

    if select_kind == 1:        # 1:随机数据多次验证
        t1.test1(quick, core_kind, filename, dataSize, numNeed)
    elif select_kind == 2:      # 2：k折叠验证  k
        k = 5
        t2.test2(quick, core_kind, filename, k, dataSize, numNeed)
    elif select_kind == 3:      # 3：留一验证法
        t3.test3(quick, core_kind, filename, dataSize, numNeed)
    else:
        print("发生了某种错误，很可能与select_kind有关")

if __name__ == '__main__':
    main(sys.argv)