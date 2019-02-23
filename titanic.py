#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:42:45 2019

@author: danielagorduza
"""

import pandas as pd
import numpy as np
import sklearn 
from sklearn import preprocessing,neighbors,svm, tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

survived = pd.read_csv("gender_submission.csv")


# Part one data manipulation / transformation
##merging testing with survived
df3 = pd.merge(df_test,survived)

## Dropping useless columns
to_drop = ['PassengerId','Name','Ticket','Cabin','Embarked']
df_train.drop(to_drop,1,inplace = True)
df3.drop(to_drop,1,inplace = True)



## change sex to numbers

dico = {"male":1,"female":0}

df_train['Sex'] = [dico[item] for item in df_train['Sex']]
df3['Sex'] = [dico[item] for item in df_test['Sex']]

## Replace empty or unuseable variables

#df_train.replace('nan', -9999999999999999,inplace = True)
df_train.dropna(inplace=True)
df3.dropna(inplace=True)





print("sum of train empty" , df_train.isnull().sum())
print("sum of test empty" , df3.isnull().sum())






#Part two split into features and classifiers
##since the train test split has been done for us lets just split into what we want to predict and predictors


ytrain = df_train['Survived']

xtrain = df_train.drop(['Survived'],1)

ytest = df3['Survived']

xtest = df3.drop(['Survived'],1)


# Part three : defining our models
mod = RandomForestClassifier(n_estimators=300, oob_score=True)

mod.fit(xtrain,ytrain)

accuracy = mod.score(xtest,ytest)

print("mod accuracy", accuracy)


acc  = all_accuracies = cross_val_score(estimator=mod, X=xtrain, y=ytrain, cv=10)
mean = acc.mean()
print ("CVed acc", mean)










