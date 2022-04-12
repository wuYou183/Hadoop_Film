# 对电影信息进行补全

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston#数据集
from sklearn.impute import SimpleImputer#填补缺失的类
from sklearn.ensemble import RandomForestRegressor
import json
from sklearn.model_selection import cross_val_score


all_data=np.zeros((100000,3))
queshi=[]
queshi_num=np.zeros(1682)
with open('C:/Users/asus/Desktop/大数据平台-Hadoop/ml-100k/ml-100k/u.data') as f:
	for i in range(0,100000):
		data=f.readline()
		all_data[i][0]=int(data.split('\t')[0])
		all_data[i][1]=int(data.split('\t')[1])
		queshi_num[int(data.split('\t')[1])-1]+=1
		if data.split('\t')[2]!='':
			all_data[i][2]=int(json.loads(data.split('\t')[2]))
		else:
			queshi.append(all_data[i][1])
			all_data[i][2]=np.nan
# print(all_data)

mse = []
buquan_result=[]
for m in range(len(queshi)):
    k=0
    # kk=0
    data_queshi=np.zeros((int(queshi_num[int(queshi[m]-1)]),3))
    for v in range(0,100000):
        if all_data[v][1]==queshi[m]:
            data_queshi[k][0]=all_data[v][0]
            data_queshi[k][1]=all_data[v][1]
            data_queshi[k][2]=all_data[v][2]
            k+=1
    
    y_full=[]
    with open('C:/Users/asus/Desktop/大数据平台-Hadoop/12628658_yxj/u.data') as ff:
        for i in range(0,100000):
            data_y=ff.readline()
            if int(data_y.split('\t')[1])==queshi[m]:
                y_full.append(data_y.split('\t')[2])

    data_copy = pd.DataFrame(data_queshi)
    sortindex = np.argsort(data_copy.isnull().sum(axis=0)).values  # [0,1,2] 用argsort 排序可以返回索引位置

    for i in sortindex:
        if data_copy.iloc[:,i].isnull().sum() == 0 :
            continue
        df = data_copy
        fillc = df.iloc[:, i]
        df = df.iloc[:,df.columns!=df.columns[i]]
    
        #在新特征矩阵中，对含有缺失值的列，进行0的填补
        df_0 = SimpleImputer(missing_values=np.nan
                            ,strategy = "constant"
                            ,fill_value = 0
                            ).fit_transform(df)
    
        #找出我们的训练集和测试集
        y_train = fillc[fillc.notnull()]  #是被选中要填充的值，存在的那些值，非空值
        y_test = fillc[fillc.isnull()]  #是被选中的要填充给的值，不存在的那些值，是空值
        x_train = df_0[y_train.index,:]  #在新特征矩阵中，我们需要非空值所对应的的索引
        x_test = df_0[y_test.index,:]  #空值所对应的记录
    
        #用随机森林回归来填补缺失值
        rfc = RandomForestRegressor(n_estimators=100)
        rfc = rfc.fit(x_train,y_train)  #导入训练集去进行训练
        Ypredict = rfc.predict(x_test)  #用oredicr接口将x_TEST,就是我们要填补空值的这些值
    
        #将填补号的特征返回到我们的原始的特征矩阵中
        data_copy.loc[data_copy.iloc[:,i].isnull(),i] = Ypredict
        buquan_result.append(Ypredict[0])

    x = [data_copy]
 
    # mse = []
    for xx in x:
        estimator = RandomForestRegressor(random_state=0, n_estimators=100)
        scores = cross_val_score(estimator,xx,y_full,scoring='neg_mean_squared_error', cv=5).mean()
        mse.append(scores * -1)

print(buquan_result)
print(mse)