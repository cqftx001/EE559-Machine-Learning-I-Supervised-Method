# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:27:40 2020

@author: Sourabh Tirodkar
USC ID- 3589406164
HW8
Problem 1
## Changing parameters for each sub question
"""

import matplotlib as plt
import sklearn
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def plotSVMBoundaries(training, label_train, classifier, support_vectors = []):

    #Plot the decision boundaries and data points for minimum distance to
    #class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # classifier: sklearn classifier model, must have a predict() function
    #
    # Total number of classes
    nclass =  max(np.unique(label_train))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 0.01
    min_x = np.floor(min(training[:, 0])) - 0.01
    max_y = np.ceil(max(training[:, 1])) + 0.01
    min_y = np.floor(min(training[:, 1])) - 0.01

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    # distance measure evaluations for each (x,y) pair.
    pred_label = classifier.predict(xy)
    
    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    #turn on interactive mode
    plt.figure()
    plt.ion()

    #show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    unique_labels = np.unique(label_train)
    # plot the class training data.
    plt.plot(training[label_train == unique_labels[0], 0],training[label_train == unique_labels[0], 1], 'rx')
    plt.plot(training[label_train == unique_labels[1], 0],training[label_train == unique_labels[1], 1], 'go')
    if nclass == 3:
        plt.plot(training[label_train == unique_labels[2], 0],training[label_train == unique_labels[2], 1], 'b*')

    # include legend for training data
    if nclass == 3:
        l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    else:
        l = plt.legend(('Class 1', 'Class 2'), loc=2)
    plt.gca().add_artist(l)

    # plot support vectors
    if len(support_vectors)>0:
        sv_x = support_vectors[:, 0]
        sv_y = support_vectors[:, 1]
        plt.scatter(sv_x, sv_y, s = 100, c = 'blue')

    plt.show()
    
    
#reading csv files
#train data    
train_data=np.genfromtxt('C:\\Users\\user\\Desktop\\USC Classes\\Spring 2020\\EE559- Mathematical Pattern Recognition\\HW\\HW8\\HW8_1_csv\\HW8_1_csv\\train_x.csv',dtype = float, delimiter= ',')
#train label
train_label=np.genfromtxt('C:\\Users\\user\\Desktop\\USC Classes\\Spring 2020\\EE559- Mathematical Pattern Recognition\\HW\\HW8\\HW8_1_csv\\HW8_1_csv\\train_y.csv',dtype = float, delimiter= ',')

#SVM
C=500
clf = SVC(C, kernel='linear')
clf.fit(train_data, train_label)
b = (clf.support_vectors_)
predicted_label=clf.predict(train_data)

print('C_value:', C)
#Accuracy 
accuracy=sklearn.metrics.accuracy_score(train_label,predicted_label)
print("Accuracy:",accuracy)

#Weights
w0=clf.intercept_ # for w0
w1=clf.coef_  # for all other weights
print("Weights (w0 w1 w2):", w0, w1)
print("Decision Boundary:")
print(w0,"+ X1*",[w1[0][0]],"+ X2*",[w1[0][1]])
print("Support Vectors:",b)

plotSVMBoundaries(train_data, train_label, clf, b)

SV1=b[0]
SV2=b[1]
SV3=b[2]

g=[0,0,0]
for i in range (0,3):
    g[i]=b[i][0]*w1[0][0]+b[i][1]*w1[0][1]+w0[0]
    print("g(X) of support vector",i+1,"when C=500 is",g[i])