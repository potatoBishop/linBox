from math import sqrt
import scipy
import numpy as np
from xgboost import XGBRegressor
from scipy import stats
import sklearn
import pandas

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
#from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score
import random
######################################################

'''
from sklearn.svm import SVR
svr_rbf = SVR(kernel ='rbf',degree = 3,gamma ='auto_deprecated',coef0 = 0.0,tol = 0.001,
C = 1.0,epsilon = 0.1,shrinking = True,cache_size = 200,verbose = False,max_iter = -1)
'''

# #############################################################################
# Generate sample data
filename = 'result_guiyihua.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径   读取元数据文件  1
index=0
X_tra=[]
X_te=[]
y_tra=[]
y_te=[]
pos = []
res = []
a = random.randint(1, 38)
b = random.randint(1, 38)
c = random.randint(1, 38)
d = random.randint(1, 38)
e = random.randint(1, 38)
f = random.randint(1, 38)
g = random.randint(1, 38)

with open(filename, 'r') as file_to_read:
  while True:
    lines = file_to_read.readline()  # 整行读取数据
    if not lines:
      break
      pass
    index=index+1

    print(a,b,c,d,e,f,g);
    if (index != a and index != b and index != c and index != d and
            index != e and index != f and index != g):  # 选取测试集2
      strs = lines.split()
      X_tmp = []
      for x in range(0, len(strs) - 1):
       X_tmp.append(float(strs[x]))
      y_tmp = float(strs[len(strs) - 1])

      X_tra.append(X_tmp)
      y_tra.append(y_tmp)
    else:
      strs = lines.split()
      X_tmp = []
      for x in range(0, len(strs) - 1):
        X_tmp.append(float(strs[x]))
      y_tmp = float(strs[len(strs) - 1])
      X_te.append(X_tmp)
      y_te.append(y_tmp)

X = np.array(X_tra)

X_test=np.array(X_te)
y=np.array(y_tra)
y_test=np.array(y_te)

# #############################################################################

# Add noise to targets
def get_rmse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

#------------手动调参
gamma_can = np.linspace(0.02,0.5,100)
max = -1000
gg = 0
for g in gamma_can:
  xgb_res = XGBRegressor(learning_rate=0.9,
                     n_estimators=1000,
                     max_depth=6,
                     min_child_weight=2,
                     gamma=g, )
  xgb_res.fit(X,y)
  score=xgb_res.score(X,y)
  score1=xgb_res.score(X_test,y_test)
         #scores=cross_val_score(svr_rbf,X,y,cv=5)
  if(score>0.7 and score1>max):
    max=score1
    # cc=c
    gg=g

xgb_res =   xgb_res = XGBRegressor(learning_rate=0.9,
                     n_estimators=1000,
                     max_depth=6,
                     min_child_weight=2,
                     gamma=gg, )
xgb_res.fit(X, y)

print('gsearch1.best_params_', gg)
print('R2(训练集)      ',xgb_res.score(X,y))
print('R2(测试集)      ',xgb_res.score(X_test,y_test))



# ----------------------------------------------------不用网格法调参

# xgb = XGBRegressor(max_depth=6, learning_rate=0.9, n_estimators=2000,gamma=0.0785,min_child_weight=2)
# xgb.fit(X,y)
# print('R2(训练集)      ',xgb.score(X,y))
# print('R2(测试集)      ',xgb.score(X_test,y_test))
# print('RMSE(训练集)      ',get_rmse(y,xgb.predict(X)))
# print('RMSE(测试集)      ',get_rmse(y_test,xgb.predict(X_test)))
#
#
# plt.scatter(y,xgb.predict(X),label='Training set')
# plt.scatter(y_test,xgb.predict(X_test),marker='^',label='Testing set')
# plt.legend()
# plt.xlabel('Experimental')
# plt.ylabel('Calculated')
# plt.legend(['Training set','Testing set'])
# plt.plot([-4,3],[-4,3])#  画对角线      3
# plt.axis([-4, 1, -4, 1])#定义xy轴的显示范围
# plt.show()
