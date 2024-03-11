# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:45:17 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:31:55 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:23:43 2020

@author: user
"""

import matplotlib as plt
import sklearn
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import random

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
def getaccuracy(train_data_use,train_label,clf):
    accuracy_store=[]
    ACC=[]
    DEV=[]
    #Split
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    skf.get_n_splits(train_data_use, train_label)

    
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
range_C= np.logspace(-3, 3, 50)
range_gamma= np.logspace(-3, 3, 50)
ACC=np.zeros([len(range_gamma),len(range_C)])
DEV=np.zeros([len(range_gamma),len(range_C)])


##Split
#skf=StratifiedKFold(n_splits=5)
#skf.get_n_splits(train_data_use, train_label)

#SVM
for i in range(len(range_gamma)): #gamma
    for j in range(len(range_C)): #C value
        clf=SVC(C=range_C[j],kernel='rbf',gamma=range_gamma[i])
        a,b=getaccuracy(train_data_use,train_label,clf)
        ACC[i][j]= a
        ACC_matrix=ACC
        DEV[i][j] = b
        DEV_matrix=DEV
        

imgplot = plt.imshow(ACC)
plt.colorbar()
plt.show()

max_accuracy=np.amax(ACC)
print("Max Accuracy:", max_accuracy)
max_index=np.where(ACC==np.max(ACC))
max_index1=(max_index[0][0],max_index[1][0])
print("Standard Deviation:", DEV[max_index1])
min_index=np.where(DEV==np.min(DEV))   

print("Gamma value",range_gamma[max_index1[0]])
print("C value",range_gamma[max_index1[1]])
#aa=(random.choice(max_index[0])) 
#print(aa)    
##bb=(random.choice(max_index[1]))    
##print(bb)
#a=print(range_gamma[aa]) 
##b=print(range_C[bb])      

#a1=max_index[0][0]
#b1=max_index[1][0]
#print(a1)
#print(b1)
#a=range_gamma[a1]
#b=range_C[b1] 
#print(a)
#print(b)

#dictionary= {max_index[0][0]: max_index[1][0]}
#aa,bb=(random.choice(max_index[0], max_index[1])) 
#print(aa)   