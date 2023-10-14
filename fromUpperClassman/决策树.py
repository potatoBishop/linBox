import sklearn.tree as tree

# 直接使用交叉网格搜索来优化决策树模型，边训练边优化
from sklearn.model_selection import GridSearchCV
# 网格搜索的参数：正常决策树建模中的参数 - 评估指标，树的深度，
# 最小拆分的叶子样本数与树的深度
import jihe

filename = '8yinWu.txt'
zu = jihe.jihe()
zu.set_file(filename)   # 文件名
zu.set_mode(1)          # 0:均匀选择4b1   1:随机选择   2: 留1法
zu.set_size(21)         # 当mode =0 ，为总样本数， 当 =1， 为需要的测试集数目  当 =2 ，为留一法取相应的值作为测试集

zu.fenZu()              # 分组并获取训练与测试集合
x_tra = zu.get_x()
y_tra = zu.get_y()
x_test = zu.get_x_test()
y_test = zu.get_y_test()

param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]}
                # 通常来说，十几层的树已经是比较深了

clf = tree.DecisionTreeClassifier()  # 定义一棵树
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=4)
    # 传入模型，网格搜索的参数，评估指标，cv交叉验证的次数
    # 这里也只是定义，还没有开始训练模型

clfcv.fit(x_tra, y_tra)

# 使用模型来对测试集进行预测
test_est = clfcv.predict(x_test)

# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test,test_est))
        # 该矩阵表格其实作用不大
print("决策树 AUC:")
fpr_test, tpr_test, th_test = metrics.roc_curve(y_test, test_est)
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))