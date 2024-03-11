# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:23:43 2020

@author: Sourabh Tirodkar
USC ID-3589406164
Problem 2
The final code has been reported. 
This has extra codes which are not required for part a and b
"""

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


#function
def getaccuracy(train_data_use,train_label,clf,skf):
    accuracy_store=[]
    DEV=[]

    for (train_index, label_index) in skf.split(train_data_use, train_label):
        X_train, X_test = train_data_use[train_index], train_data_use[label_index] 
        Y_train, Y_test = train_label[train_index], train_label[label_index]
        #print(X_train, Y_train)
        #print(X_test, Y_test)
        #SVM
        clf.fit(X_train, Y_train)
        predicted_label_valid=clf.predict(X_test)


        #Accuracy 
        accuracy_valid=sklearn.metrics.accuracy_score(Y_test,predicted_label_valid)
        accuracy_store.append(accuracy_valid)
        a=(np.std(accuracy_store))
        DEV.append(a)

    Average_crossvalidation_accuracy=np.mean(accuracy_store)
    #print('Average_crossvalidation_accuracy:', Average_crossvalidation_accuracy)
    Average_crossvalidation_std=np.mean(DEV)
    #print('Average_crossvalidation_std:', Average_crossvalidation_std)
    
    return Average_crossvalidation_accuracy,Average_crossvalidation_std


#using first 2 features for the problem
train_data_use=train_data[:,0:2]
test_data_use=test_data[:,0:2]

accuracy_store=[]
max_accuracy_store=[]
range_C= np.logspace(-3, 3, 50)
range_gamma= np.logspace(-3, 3, 50)
#ACC=np.zeros([len(range_gamma),len(range_C)])
#DEV=np.zeros([len(range_gamma),len(range_C)])

ACCf=[]
DEVf=[]
Store=[]
Gamma_value=[]
C_value=[]
best_gamma=[]
best_C=[]
Average_ACC_matrix=np.zeros([50,50])
Average_DEV_matrix=np.zeros([50,50])
##Split
#skf=StratifiedKFold(n_splits=5)
#skf.get_n_splits(train_data_use, train_label)

#SVM
for k in range(20):
    ACC=np.zeros([50,50])
    DEV=np.zeros([50,50])
    #Split
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state = k+3)
    skf.get_n_splits(train_data_use, train_label)
    for i in range(len(range_gamma)): #gamma
        for j in range(len(range_C)): #C value
            clf=SVC(C=range_C[j],kernel='rbf',gamma=range_gamma[i])
            a,b=getaccuracy(train_data_use,train_label,clf,skf)
            ACC[i][j]= a
            ACC_matrix=ACC
            DEV[i][j] = b
            DEV_matrix=DEV
    
    ACCf.append(ACC)
    DEVf.append(DEV)
    max_accuracy=np.amax(ACC)
    print(k)
    #print(max_accuracy)
    max_accuracy_store.append(max_accuracy)

    max_index=np.where(ACC==np.max(ACC))
    min_std=np.amin(DEV)
    min_index=np.where(DEV==np.min(DEV))
    a1=max_index[0][0]
    b1=max_index[1][0]
    #print(a1)
    #print(b1)
    aa=range_gamma[a1] 
    bb=range_C[b1]
    Gamma_value.append(aa)
    C_value.append(bb)
    
    
    
result=0
result1=0
for i in range(20): 
    result=result+ ACCf[i] 
    result1=result1+DEVf[i]
    Average_ACC_matrix= result/20
    Average_DEV_matrix= result1/20   
    

max_accuracy=np.amax(Average_ACC_matrix)
#print("Max Accuracy:", max_accuracy)
max_index=np.where(Average_ACC_matrix==np.max(Average_ACC_matrix))
max_index1=(max_index[0][0],max_index[1][0])

min_index=np.where(Average_DEV_matrix==np.min(Average_DEV_matrix))   

#print("Gamma value",range_gamma[max_index1[0]])
#print("C value",range_gamma[max_index1[1]])

Avg_accuracy=np.mean(max_accuracy_store)
print('Avg_cross_validation_accuracy',Avg_accuracy)

print('Best Gamma Value',range_gamma[max_index1[0]])

print('Best C value',range_gamma[max_index1[1]])
best_gamma=range_gamma[max_index1[0]]
best_C=range_gamma[max_index1[1]]
print("Standard Deviation:", DEV[max_index1])


###PART 2D
#SVM
clf = SVC(C=best_C, kernel='rbf',gamma=best_gamma)
clf.fit(train_data_use, train_label)
b = (clf.support_vectors_)
predicted_label=clf.predict(train_data_use)
predicted_label_test=clf.predict(test_data_use)

#Accuracy 
print('Avg_cross_validation_accuracy',Avg_accuracy)
print("Standard Deviation:", DEV[max_index1])
print("1 standard deviation value:",Avg_accuracy-DEV[max_index1])
accuracy1=sklearn.metrics.accuracy_score(test_label,predicted_label_test)
print("Testing Accuracy:",accuracy1)



 