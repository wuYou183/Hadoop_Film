#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from hdfs import *
import pymysql

from mrjob.job import MRJob
from mrjob.step import MRStep

import os
import math

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# In[2]:


# 对电影信息进行补全

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def Check_buquan(client,file_pingfen):
    all_data=np.zeros((100000,3))
    queshi=[]
    queshi_num=np.zeros(1682)
    with client.read(file_pingfen) as f:
        for i in range(0,100000):
            data=f.readline()
            all_data[i][0]=int(data.split('\t'.encode(encoding='utf-8'))[0])
            all_data[i][1]=int(data.split('\t'.encode(encoding='utf-8'))[1])
            queshi_num[int(data.split('\t'.encode(encoding='utf-8'))[1])-1]+=1
            if data.split('\t'.encode(encoding='utf-8'))[2]!='':
                all_data[i][2]=int(data.split('\t'.encode(encoding='utf-8'))[2])
            else:
                queshi.append(all_data[i][1])
                all_data[i][2]=np.nan

    buquan_result=[]
    for m in range(len(queshi)):
        k=0
        data_queshi=np.zeros((int(queshi_num[int(queshi[m]-1)]),3))
        for v in range(0,100000):
            if all_data[v][1]==queshi[m]:
                data_queshi[k][0]=all_data[v][0]
                data_queshi[k][1]=all_data[v][1]
                data_queshi[k][2]=all_data[v][2]
                k+=1
            
        data_copy = pd.DataFrame(data_queshi)
        sortindex = np.argsort(data_copy.isnull().sum(axis=0)).values  # [0,1,2]

        for i in sortindex:
            if data_copy.iloc[:,i].isnull().sum() == 0 :
                continue
            df = data_copy
            fillc = df.iloc[:, i]
            df = df.iloc[:,df.columns!=df.columns[i]]
    
            #在新特征矩阵中，对含有缺失值的列，进行0的填补
            df_0 = SimpleImputer(missing_values=np.nan,strategy = "constant",fill_value = 0).fit_transform(df)
    
            #找出我们的训练集和测试集
            y_train = fillc[fillc.notnull()]  #是被选中要填充的值，存在的那些值，非空值
            y_test = fillc[fillc.isnull()]  #是被选中的要填充给的值，不存在的那些值，是空值
            x_train = df_0[y_train.index,:]  #在新特征矩阵中，我们需要非空值所对应的的索引
            x_test = df_0[y_test.index,:]  #空值所对应的记录
    
            #用随机森林回归来填补缺失值
            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(x_train,y_train) 
            Ypredict = rfc.predict(x_test)
    
            #将填补号的特征返回到我们的原始的特征矩阵中
            data_copy.loc[data_copy.iloc[:,i].isnull(),i] = Ypredict
            buquan_result.append(Ypredict)
            
            client.write(file_pingfen,Ypredict,overwrite=False,append=True)


# In[3]:


def ReadData():
    client=Client('http://localhost:9870')
    
    file_movie='/user/hadoop/ml-100k/ml-100k/u.item'
    file_pingfen='/user/hadoop/ml-100k/ml-100k/u.data'
    file_user='/user/hadoop/ml-100k/ml-100k/u.user'
    file_sum='/user/hadoop/ml-100k/ml-100k/u.info'
    
    all_sum = {}
    with client.read(file_sum) as lines_0:
        line_0 = lines_0.readline()
        while line_0!=''.encode(encoding='utf-8'):
            number, name = line_0.split(' '.encode(encoding='utf-8'))
            all_sum.setdefault(name,{})
            all_sum[name]=int(number)
            line_0 = lines_0.readline()
    
#     user_movie = {}  #用来存放用户对电影的评分信息
#     with client.read(file_pingfen) as lines: #逐行读取
#         line = lines.readline()
#         while line!=''.encode(encoding='utf-8'):
#             user, item, score = line.split('\t'.encode(encoding='utf-8'))[0:3]
#             user_movie.setdefault(user,{})
#             user_movie[user][item] = int(score)
#             line=lines.readline()
    Check_buquan(client,file_pingfen)

    movies = {}  #用来存放电影基本信息
    movies_styles = np.zeros((all_sum['items\n'.encode(encoding='utf-8')],19))  #存放电影种类
    with client.read(file_movie) as lines_2:
        line_2=lines_2.readline()
        while line_2!=''.encode(encoding='utf-8'):
            movieId, movieTitle = line_2.split('|'.encode(encoding='utf-8'))[0:2]
            movies[movieId] = movieTitle
            for style in range(0,19):
                movies_styles[int(movieId)-1][style]=int(line_2.split('|'.encode(encoding='utf-8'))[style+5])
            line_2=lines_2.readline()
            
    user_infor = {} #用于存放用户信息
    with client.read(file_user) as lines_3:
        line_3=lines_3.readline()
        while line_3!=''.encode(encoding='utf-8'):
            userId, age, sex, occupation = line_3.split('|'.encode(encoding='utf-8'))[0:4]
            user_infor.setdefault(userId,{})
            user_infor[userId]=int(age),sex,occupation
            line_3=lines_3.readline()

    return movies,movies_styles,all_sum,user_infor


# In[4]:


import json

def ReadData_2():
    os_str='python3.7 /home/hadoop/Desktop/mapreduce1.py /home/hadoop/Desktop/ml-100k/u.data > /home/hadoop/Desktop/result_test1'
    ff=os.system(os_str)
    user_movie={}    #用来存放用户对电影的评分信息
    user_pingfen_sum=np.zeros(all_sum['users\n'.encode(encoding='utf-8')]+1)
    with open('/home/hadoop/Desktop/result_test1','r') as result1:
        while True:
            data1=result1.readline()
            if data1=='':
                break
            user1=data1.split('\t')[0].strip("\"")
            js1=json.loads(data1.split('\t')[1])
            sum=js1[0]
            user_pingfen_sum[int(user1)]=sum
            for i in range(sum):   
                user_movie.setdefault(user1,{})
                item1=js1[1][i][0]
                user_movie[user1][item1]=js1[1][i][1]
    return user_movie,user_pingfen_sum
                


# In[5]:


import json

def read_user_movie_pingfen():
    os_str='python3.7 /home/hadoop/Desktop/mapreduce2.py /home/hadoop/Desktop/result_test1 > /home/hadoop/Desktop/result_test2_2'
    ff=os.system(os_str)
    user_movie_pingfen={}
    with open('/home/hadoop/Desktop/result_test2_2','r') as result2:
        while True:
            data2=result2.readline()
            if data2=='':
                break
            js2_name=json.loads(data2.split('\t')[0])
            js2_value=json.loads(data2.split('\t')[1])
            user_movie_pingfen.setdefault(js2_name[0],{})
            user_movie_pingfen[js2_name[0]][js2_name[1]]=js2_value
        return user_movie_pingfen
    


# In[6]:


def UserSimilarity(user_input,user_infor):
    W_user = {} #计算最终物品余弦相似度矩阵
    for user in user_infor.keys():
        a=1
        b=1
        W_user.setdefault(user,{})
        age,sex,occupation=user_infor[user]
        if user_input[1]==sex:
            a=0
        if user_input[2]==occupation:
            b=0
        W_user[user]=math.sqrt(abs(user_input[0]-age)+a+b)
    return W_user


# In[7]:


#根据用户信息得到推荐信息
def Recommend_1(user_input,user_info,user_movie,user_movie_pingfen):
    #取前10最相似的用户数据
    rank_user=dict(sorted(UserSimilarity(user_input,user_infor).items(),key = lambda x:x[1],reverse=False)[0:10])
    movies_re={}
    for k in rank_user.keys():
        movies_re.setdefault(k,{})
        rank = {}
        action_item = user_movie[str(k,encoding='utf-8')]
        for item, score in action_item.items():
            #取前10最相似的电影数据
            for j, wj  in sorted(user_movie_pingfen[item].items(),key = lambda x: x[1], reverse = True)[0:10]:
                rank.setdefault(j,0)
                #计算相似度
                rank[j] += wj*(100-rank_user[k])
        #挑选出最相似的前10个数据
        movies_re[k]=dict(sorted(rank.items(),key = lambda x:x[1],reverse= True)[0:10])
    #将推荐数据写入文件
    with open('/home/hadoop/Desktop/rec_result','a+',encoding='utf-8') as f:
        #清空文件
        f.seek(0)
        f.truncate()
        for k in movies_re.keys():
            for kk in movies_re[k].keys():
                f.write(str(kk)+'\t'+str(movies_re[k][kk])+'\n')
            


# In[8]:


import json
#不分析电影类别推荐
def Recommend_2(movies,N):
    recom={}
    recom_result={}
    item_name=np.zeros(N)
    kk=0
    with open('/home/hadoop/Desktop/rec_mr_result','r') as f:
        while True:
            data=f.readline()
            if data=='':
                break
            js_name=json.loads(data.split('\t')[0])
            js_value=json.loads(data.split('\t')[1])
            recom.setdefault(js_name,0.0)
            recom[js_name]=float(js_value)
    #计算相似度
    for i,simi in sorted(recom.items(),key = lambda x:x[1],reverse= True)[0:N]:
        for j in movies.keys():
            if i.encode(encoding='utf-8')==j:
                item_name[kk]=int(j)
                kk+=1
                recom_result.setdefault(str(movies[j]),simi)
    #输出结果
    with open('/home/hadoop/Desktop/final_re_result','a+',encoding='utf-8') as f:
        f.seek(0)
        f.truncate()
        for k in recom_result.keys():
            f.write(str(k)+'\n')
    return recom_result,item_name


# In[9]:


import json
#分析电影类别推荐
def Recommend_3(movies,movies_styles,styles,N):
    recom={}
    recom_result={}
    item_name=np.zeros(N)
    kk=0
    with open('/home/hadoop/Desktop/rec_mr_result','r') as f:
        while True:
            data=f.readline()
            if data=='':
                break
            js_name=json.loads(data.split('\t')[0])
            js_value=json.loads(data.split('\t')[1])
            recom.setdefault(js_name,0.0)
            recom[js_name]=float(js_value)
    #分析电影类别
    for m in range(len(styles)):
        for n in recom.keys():
            if movies_styles[int(n)-1][styles[m]]==0:
                recom[n]*=0
    #计算相似度
    for i,simi in sorted(recom.items(),key = lambda x:x[1],reverse= True)[0:N]:
        if simi==0:    #去掉无关数据
            continue
        for j in movies.keys():
            if i.encode(encoding='utf-8')==j:
                item_name[kk]=int(j)
                kk+=1
                recom_result.setdefault(str(movies[j],encoding='utf-8'),simi)
    #输出结果
    with open('/home/hadoop/Desktop/final_re_result_1','a+',encoding='utf-8') as f:
        f.seek(0)
        f.truncate()
        for k in recom_result.keys():
            f.write(str(k)+'\n')
    return recom_result,item_name


# In[10]:


#存放到mysql
def put_in_mysql(user_input,styles,N):
    style_kind=['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fanstasy','Film_Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western']
    #连接mysql
    db=pymysql.Connect(host='localhost',port=3306,user='root',password='123456',db='my_movies')  
    cursor=db.cursor()
    str_user=str(user_input[0])+'_'+str(user_input[1])+'_'+str(user_input[2])+'_'+str(N)
    for i in range(len(styles)):
        str_user+='_'
        str_user+=style_kind[styles[i]]
    #将推荐信息存放到表中
    cursor.execute('CREATE TABLE '+str_user+'(movies_name VARCHAR(255))')
    query='INSERT INTO '+str_user+'(movies_name)VALUES(%s)'
    with open('/home/hadoop/Desktop/final_re_result_1','r') as f:
        while True:
            data=f.readline()
            if data=='':
                break
            value=[data]
            cursor.executemany(query,value)
            db.commit()
    cursor.close()
    db.close()


# In[11]:


import plotly.graph_objs as go
import numpy as np

def pingfen_an(item_name,movies_styles):
    style_kind=['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fanstasy','Film_Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western']
    style=np.zeros(19)
    for i in range(len(item_name)):
        for j in range(19):
            if movies_styles[int(item_name[i])-1][j] ==1:
                style[j]+=1
    trace=go.Bar(x=style_kind[1:],y=style[1:],name='统计')
    data=[trace]
    fig=go.Figure(data)
    fig.update_layout(title='电影种类统计图',xaxis=dict(title='电影种类'),yaxis=dict(title='数量'))
    fig.show()


# In[12]:



#数据处理
movies,movies_styles,all_sum,user_infor = ReadData()   #读取数据
user_movie,user_pinglun_sum=ReadData_2()    #预处理评分数据
user_movie_pingfen=read_user_movie_pingfen()    #电影相似度矩阵


# In[18]:


#UI界面
import PySimpleGUI as sg
def movies_ui():
    layout=[[sg.Text('性别'),sg.InputText(key='MF')],
        [sg.Text('年龄'),sg.InputText(key='age')],
        [sg.Text('职业'),sg.InputText(key='profession')],
        [sg.Text('类别'),sg.InputText(key='kind')],
        [sg.Text('数量'),sg.InputText(key='num')],
        [sg.Button('搜索')],
        [sg.Multiline('',key='result',size=(60,20),background_color='white',text_color='black')]]

    window=sg.Window('电影推荐系统',layout)

    while True:
        event,values=window.read()
        
        if event=='搜索':
            window['result'].update('')
            text1=values['MF']
            text2=values['age']
            text3=values['profession']
            text4=values['kind']
            text5=values['num']

            user_input=[int(text2),text1,text3]
            if text5=='':
                N=10
            else:
                N=int(text5)
            W_user=UserSimilarity(user_input,user_infor)    #用户相似度矩阵
            Recommend_1(user_input,user_infor,user_movie,user_movie_pingfen)    #各个电影推荐系数
            #处理推荐系数
            os_str='python3.7 /home/hadoop/Desktop/mapreduce3.py /home/hadoop/Desktop/rec_result > /home/hadoop/Desktop/rec_mr_result'
            ff=os.system(os_str)
            #得到推荐结果
            if text4=='':
                recom_2,item_name2=Recommend_2(movies,N) 
                pingfen_an(item_name2,movies_styles)
                path='/home/hadoop/Desktop/final_re_result'
                print(user_input)
                print(item_name2)
                styles=[0]
                put_in_mysql(user_input,styles,N)
            else:
                styles=[int(text4)]
                recom_3,item_name3=Recommend_3(movies,movies_styles,styles,N)
                pingfen_an(item_name3,movies_styles)
                path='/home/hadoop/Desktop/final_re_result_1'
                print(user_input,styles)
                print(item_name3)
                put_in_mysql(user_input,styles,N)
            #放入mysql
          #  put_in_mysql(user_input,styles)
            
            movie=''
            with open(path,'r') as f:
                while True:
                    data=f.readline()
                    if data=='':
                        break
                    movie+=str(data)
                        
            window['result'].update(value=movie)

        if event==sg.WIN_CLOSED:
            break
    
    window.close()


# In[21]:


movies_ui()


# #### 
