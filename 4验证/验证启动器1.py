import sys
import 验证myData as test1
import 验证myData_k折 as test2
import 验证myData_留一 as test3


class test:
    def __init__(self):
        print("starting 检测参数正确性")


def main(argv):
    quick = [3, 1.44715268, 1.45220254, -0.22055727, 0.61210873, 1]
    select_kind = 0      # 1:随机数据多次验证
                         # 2：k折叠验证
                         # 3：留一验证法
    X = []
    if select_kind == 1:
        ...
    elif select_kind == 2:
        ...
    elif select_kind == 3:
        ...
    else:
        print("发生了某种错误，很可能与select_kind有关")

if __name__ == '__main__':
    main(sys.argv)