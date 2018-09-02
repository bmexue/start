# -*- coding: UTF-8 -*-
#################################################  
# SVM: support vector machine  
# Author : zouxy  
# Date   : 2013-12-12  
# HomePage : http://blog.csdn.net/zouxy09  
# Email  : zouxy09@qq.com  
#################################################  
  
from numpy import *  
from svm import *
import svm as SVM


  
################## test svm #####################  
## step 1: load data  
print ("step 1: load data...")  
dataSet = []  
labels = []  

fileIn = open('testSet_old.txt')  
for line in fileIn.readlines():  
    lineArr = line.strip().split('\t')  
    dataSet.append([float(lineArr[0]), float(lineArr[1])])  
    labels.append(float(lineArr[2]))  

'''
dataSet.append([float(1.0), float(1.0)])  
labels.append(float(1))
dataSet.append([float(0.9), float(0.8)])  
labels.append(float(1))  

dataSet.append([float(2.0), float(2.0)])  
labels.append(float(-1))
'''

'''
dataSet.append([float(3.542485), float(1.977398)])  
labels.append(float(-1))
dataSet.append([float(3.018896), float(2.556416)])  
labels.append(float(-1))  

dataSet.append([float(7.551510), float(-1.580030)])  
labels.append(float(1))
dataSet.append([float(8.127113), float(1.274372)])  
labels.append(float(1))  
'''
#test
'''
dataSet.append([float(0.5), float(1.5)])  
labels.append(float(1))
dataSet.append([float(0.0), float(1.1)])  
labels.append(float(-1)) 
'''

cta = len(dataSet)
ct = cta# - 2
dataSet = mat(dataSet)  
labels = mat(labels).T  
train_x = dataSet[0:ct, :]  
train_y = labels[0:ct, :]  
test_x = dataSet[0:cta, :]  
test_y = labels[0:cta, :]  
  
## step 2: training...  
print ("step 2: training...") 

maxIter = 50

svmClassifier = SVM.trainSVM(train_x, train_y, maxIter, kernelOption = ('linear', 0))

#print('svmClassifier.alphas %s' % (svmClassifier.alphas))
#print('svmClassifier.b: %s' % (svmClassifier.b))
#print('svmClassifier.errorCache: %s' % (svmClassifier.errorCache))

## step 3: testing  
print ("step 3: testing...") 
accuracy = 0
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)  
 

## step 4: show the result  
print ("step 4: show the result...")    
print ('The classify accuracy is: %.3f' % (accuracy * 100))
SVM.showSVM(svmClassifier)  