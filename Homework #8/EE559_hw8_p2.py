# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:23:43 2020

@author: user
"""

import csv
import matplotlib as plt
import sklearn
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


#reading csv files
#train data    
train_data=np.genfromtxt('C:\\Users\\user\\Desktop\\USC Classes\\Spring 2020\\EE559- Mathematical Pattern Recognition\\HW\\HW8\\wine_csv\\wine_csv\\feature_train.csv',dtype = float, delimiter= ',')
#train label   
train_label=np.genfromtxt('C:\\Users\\user\\Desktop\\USC Classes\\Spring 2020\\EE559- Mathematical Pattern Recognition\\HW\\HW8\\wine_csv\\wine_csv\\label_train.csv',dtype = float, delimiter= ',')
#test data   
test_data=np.genfromtxt('C:\\Users\\user\\Desktop\\USC Classes\\Spring 2020\\EE559- Mathematical Pattern Recognition\\HW\\HW8\\wine_csv\\wine_csv\\feature_test.csv',dtype = float, delimiter= ',')
#test label    
test_label=np.genfromtxt('C:\\Users\\user\\Desktop\\USC Classes\\Spring 2020\\EE559- Mathematical Pattern Recognition\\HW\\HW8\\wine_csv\\wine_csv\\label_test.csv',dtype = float, delimiter= ',')

#using first 2 features for the problem
train_data_use=train_data[:,0:2]
test_data_use=test_data[:,0:2]



accuracy_store=[]
ACC=[]
DEV=[]
#Split
skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(train_data_use, train_label)


for (train_index, label_index) in skf.split(train_data_use, train_label):
    X_train, X_test = train_data_use[train_index], train_data_use[label_index] 
    Y_train, Y_test = train_label[train_index], train_label[label_index]
    #print(X_train, Y_train)
    #print(X_test, Y_test)
    #SVM
    clf = SVC(C=1, kernel='rbf',gamma=1)
    clf.fit(X_train, Y_train)
    b = (clf.support_vectors_)
    predicted_label=clf.predict(X_train)
    predicted_label_valid=clf.predict(X_test)


    #Accuracy 
    accuracy=sklearn.metrics.accuracy_score(Y_train,predicted_label)
    print("Training Accuracy:",accuracy*100)
    accuracy_valid=sklearn.metrics.accuracy_score(Y_test,predicted_label_valid)
    print("Cross Validation Accuracy:",accuracy_valid*100)
    accuracy_store.append(accuracy_valid)
    a=(np.std(accuracy_store))
    DEV.append(a)
    print('........')

Average_crossvalidation_accuracy=np.mean(accuracy_store)
print('Average_crossvalidation_accuracy:', Average_crossvalidation_accuracy)
Average_crossvalidation_std=np.mean(DEV)
#print('Average_crossvalidation_std:', Average_crossvalidation_std)
    


