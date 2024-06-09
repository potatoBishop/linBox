import numpy as np
from numpy import random


class dataHandle:
    filename = str
    mode = 0
    dataSize = 0
    testNumNeed = 0
    idNeed = 0

    x = []
    y = []
    x_test = []
    y_test = []
    x_all = []
    y_all = []
    selected_id_train = []
    selected_id_test = []
    flag = np.zeros(dataSize + 10)
    """ 
    0: 默认4 : 1 均匀分组
    1: 随机留k分组 
    2: 留1分组
    3: 分类X 与 Y ，存储在训练集中
    """

    def fenZu(self):
        X_tra = []
        X_te = []
        px_all = []
        y_tra = []
        y_te = []
        py_all = []
        self.flag = np.zeros(self.dataSize + 10)

        num = 0

        # 创建训练与测试集数组     1:代表选中为测试集
        if self.mode == 0:                                  # mode = 0 进行 训练：测试 == 4：1的分组，顺序固定，分组均匀
            num = 1
            self.flag[num] = 1
            while num <= self.dataSize:
                num = num + 5
                self.flag[num] = 1
        elif self.mode == 1:                                # mode = 1 随机选择 need 个数据作为测试集
            while num < self.numNeed:
                temp = random.randint(1, self.dataSize)
                if self.flag[temp] == 0:
                    self.flag[temp] = 1
                    num = num + 1
        elif self.mode == 2:                                # mode = 2 留一法  根据idNeed选出唯一的测试集
            self.flag[ self.idNeed ] = 1
        elif self.mode == 3:                                # mode = 3 按照给定的flagSet选择参数 前提：已完成set
            pass  # def set_flagSet(self, pflag):
        elif self.mode == 4:                                # mode = 4 返回设置的x， y值
            for i in self.selected_id_test:
                self.flag[i+1] = 1
        else:
            pass

        # 进行分组
        index = 0
        with open(self.filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()         # 整行读取数据
                if not lines:                           # 已经读取完毕
                    break
                index = index + 1                       # 还有数据需要处理
                strs = lines.split()
                X_tmp = []
                for x in range(0, len(strs) - 1):       # 左闭右开
                    X_tmp.append(float(strs[x]))
                y_tmp = float(strs[len(strs) - 1])

                if self.flag[index] == 0:               # flag[index] == 0 选训练集
                    X_tra.append(X_tmp)
                    y_tra.append(y_tmp)
                elif self.flag[index] == 1:             # flag[index] == 1 选测试集
                    X_te.append(X_tmp)
                    y_te.append(y_tmp)

                # all集合获取
                px_all.append(X_tmp)
                py_all.append(y_tmp)

        self.x = np.array(X_tra)                        # 处理完成的训练集和测试集
        self.x_test = np.array(X_te)
        self.y = np.array(y_tra)
        self.y_test = np.array(y_te)
        self.x_all = np.array(px_all)
        self.y_all = np.array(py_all)

    def get_x_all(self):
        # print("x_all:", self.x_all)
        return self.x_all

    def get_y_all(self):
        # print("y_all:", self.y_all)
        return self.y_all

    def set_file(self, filename):
        self.filename = filename

    def set_mode(self, mode):
        """"
        mode = 0 进行 训练：测试 == 4：1的分组，顺序固定，分组均匀\n
        mode = 1 随机选择 need 个数据作为测试集\n
        mode = 2 留一法  根据idNeed选出唯一的测试集\n
        mode = 3 按照给定的flagSet选择参数 前提：已完成set\n
        mode = 4 返回设置的x， y值   未完善\n
        """
        self.mode = mode

    def set_dataSize(self, dataSize):
        self.dataSize = dataSize

    def get_x(self):
        # print("x_train:", self.x)
        return self.x

    def get_y(self):
        # print("y_train:", self.y)
        return self.y

    def get_x_test(self):
        # print("x_test:", self.x_test)
        return self.x_test

    def get_y_test(self):
        # print("y_test:", self.y_test)
        return self.y_test

    def get_mod(self):
        return self.mode

    def set_numNeed(self, numNeed):
        self.numNeed = numNeed

    def get_numNeed(self):
        return self.numNeed

    def set_idNeed(self, idNeed):
        self.idNeed = idNeed

    def get_idNeed(self):
        return self.idNeed

    def set_selected_id(self, selected_id_train, selected_id_test):
        self.selected_id_train = selected_id_train
        self.selected_id_test = selected_id_test

    def set_flagSet(self, pflag):
        self.flag = pflag


