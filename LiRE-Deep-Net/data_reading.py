# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:53:12 2022

@author: ZESHAN KAHN
"""

def read_all(files_dev):

    train_X=pd.DataFrame()
    train_Y=pd.DataFrame()

    for f in files_dev:
        print(f,train_X.shape)
        if("lbp" in f):
            df=pd.read_csv(f).iloc[:,1:]
            df["file_name"]=df['Yl']+"__"+df['filename'].astype('str')+".jpg"
            df.set_index("file_name",inplace=True)
            train_Y=df['Yl']
            df.drop(['Yl','Y','filename'],axis=1,inplace=True)
            
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X = pd.concat([train_X, df], axis=1)

        elif("densenet" in f):
            df=pd.read_csv(f).iloc[:,1:]
            df[['Y','img']] = df['image_name'].str.split('__',expand=True)
            df.rename(columns={"image_name": "file_name"},inplace=True)
            df.set_index("file_name",inplace=True)
            
            train_Y=df['Y']
            df.drop(['Actual','Pred','Y','img'],axis=1,inplace=True)
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X = pd.concat([train_X, df], axis=1,join="inner")
        elif("mobilenetv2" in f):
            df=pd.read_csv(f).iloc[:,1:-2].set_index('image_name')
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X = pd.concat([train_X, df], axis=1,join="inner")
        else:
            df=pd.read_csv(f).iloc[:,1:]
            df["file_name"]=df['class1']+"__"+df['img'].astype('str')
            df.set_index("file_name",inplace=True)
            train_Y=df['class1']
            df.drop(['class1','img'],axis=1,inplace=True)
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X = pd.concat([train_X, df], axis=1,join="inner")

    print(train_X.shape,train_Y.shape)
    return train_X,train_Y

import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import f1_score,accuracy_score,matthews_corrcoef
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

clfs=[LogisticRegression(random_state=0,solver='liblinear'),
          RandomForestClassifier(n_estimators=50, random_state=1),
          ExtraTreesClassifier(n_estimators=100, random_state=0),
          SVC(gamma='auto'),
          LinearSVC(random_state=0, tol=1e-05),
          KNeighborsClassifier(n_neighbors=1),
          DecisionTreeClassifier(random_state=0),
          SGDClassifier(max_iter=1000, tol=1e-3),
          GaussianNB()
    ]
clf=LogisticRegression(random_state=0,solver='liblinear')
files_dev=[base_path+f for f in os.listdir(base_path) if "dev_" in f and "densenet" not in f]# and "Tamura" not in f]
files_dev.sort()
#files_dev=files_dev[:-7]
train_X,train_Y=read_all(files_dev)
temp=pd.concat([train_X, train_Y], axis=1,join="inner")
train_X=temp.iloc[:,:-1]
train_Y=temp.iloc[:,-1]
temp.to_csv("input1.csv")

files_val=[f.replace("dev_","val_") for f in files_dev]
test_X,test_Y=read_all(files_val)
temp=pd.concat([test_X,test_Y], axis=1,join="inner")
test_X=temp.iloc[:,:-1]
test_Y=temp.iloc[:,-1]
temp.to_csv("input2.csv")
def many(Y):
    Ys=np.zeros((len(Y),len(set(Y))),np.float32)
    for i in range(len(Y)):
        Ys[i,label_map16.index(Y[i])]=1
    return Ys