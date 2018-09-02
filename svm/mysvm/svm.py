# -*- coding: UTF-8 -*-
#################################################  
# SVM: support vector machine  
# Author : zouxy  
# Date   : 2013-12-12  
# HomePage : http://blog.csdn.net/zouxy09  
# Email  : zouxy09@qq.com  
#################################################  
  
from numpy import *  
import time  
import matplotlib.pyplot as plt  

# calulate kernel value  
def calcKernelValue(matrix_x, sample_x, kernelOption):  
    kernelType = kernelOption[0]  
    numSamples = matrix_x.shape[0]  
    kernelValue = mat(zeros((numSamples, 1)))  

    if kernelType == 'linear':  
        kernelValue = matrix_x * sample_x.T  # 矩阵相乘
    elif kernelType == 'rbf':  
        sigma = kernelOption[1]  
        if sigma == 0:  
            sigma = 1.0  
        for i in range(numSamples):  
            diff = matrix_x[i, :] - sample_x  
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))  
    else:  
        raise NameError('Not support kernel type! You can use linear or rbf!')  
    return kernelValue  
  
  
# calculate kernel matrix given train set and kernel type  
def calcKernelMatrix(train_x, kernelOption):  
    numSamples = train_x.shape[0]  
    kernelMatrix = mat(zeros((numSamples, numSamples)))  
    for i in range(numSamples):  
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    print ('==========begin')
    print (train_x)
    print ('==========mid')
    print (train_x[i, :])
    print ('==========end')
    return kernelMatrix  # n*n   Kij
  
  
# define a struct just for storing variables and data  
class SVMStruct:  
    def __init__(self, dataSet, labels, kernelOption):  
        self.train_x = dataSet # each row stands for a sample  
        self.train_y = labels  # corresponding label  
        self.numSamples = dataSet.shape[0] # number of samples  
        self.alphas = mat(zeros((self.numSamples, 1))) # Lagrange factors for all samples 
        self.flag = mat(zeros((self.numSamples, 1))) # Lagrange factors for all samples  
        self.b = 0  
        self.errorCache = mat(zeros((self.numSamples, 2)))  
        self.kernelOpt = kernelOption  
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)
  
          
# calculate the error for alpha k  
def calcError(svm, alpha_k): 
    #print ('alphas %d %d' % (svm.alphas.shape[0],svm.alphas.shape[1]))
    #print ('train_y %d %d' % (svm.train_y.shape[0],svm.train_y.shape[1]))
    t1 = multiply(svm.alphas, svm.train_y).T    # 1 * 81   multiply 是内积 
    t2 = svm.kernelMat[:, alpha_k]              # 81 * 1
    #print ('t1 %d %d' % (t1.shape[0],t1.shape[1]))
    #print ('t2 %d %d' % (t2.shape[0],t2.shape[1]))
    output_k = float(t1 * t2+ svm.b)  
    #print (output_k)
    error_k = output_k - float(svm.train_y[alpha_k])  
    return error_k  
  
  
# update the error cache for alpha k after optimize alpha k  
def updateError(svm, alpha_k):  
    error = calcError(svm, alpha_k)  
    svm.errorCache[alpha_k] = [1, error]  
  
  
# select alpha j which has the biggest step  
def selectAlpha_j(svm, alpha_i, error_i):  
    svm.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)  
    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0] # mat.A return array  
    maxStep = 0; alpha_j = 0; error_j = 0  
    #print ('candidateAlphaList: %s' % (candidateAlphaList))
    # find the alpha with max iterative step  
    if len(candidateAlphaList) > 1: 
        for alpha_k in candidateAlphaList:  
            if alpha_k == alpha_i:   
                continue  
            error_k = calcError(svm, alpha_k)  
            if abs(error_k - error_i) > maxStep and svm.train_y[alpha_i] != svm.train_y[alpha_k]:  
                maxStep = abs(error_k - error_i)  
                alpha_j = alpha_k  
                error_j = error_k  
    # if came in this loop first time, we select alpha j randomly  
    else:
        alpha_j = alpha_i  
        while alpha_j == alpha_i:  
            alpha_j = int(random.uniform(0, svm.numSamples))  
        error_j = calcError(svm, alpha_j)  
      
    return alpha_j, error_j  
  
  

# the inner loop for optimizing alpha i and alpha j  
def innerLoop(svm, alpha_i):  
    error_i = calcError(svm, alpha_i)  
  
    ### check and pick up the alpha who violates the KKT condition  
    ## satisfy KKT condition  
    # 1) yi*f(i) - 1 > 0 and alpha == 0 (outside the boundary)  
    # 2) yi*f(i) - 1 == 0 and alpha > 0 0 (between the boundary)  
    ## violate KKT condition  
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so  
    # violate: y[i]*f(i) >1 and alphas >0    or  y[i]*f(i) < 1
    # => y[i]*E_i >0 and alphas >0    or  y[i]*E_i < 0
    # yi*f(i) - 1 < 0   => y[i]*E_i < 0
    if ( svm.train_y[alpha_i] * error_i > 0 ) and (svm.alphas[alpha_i] >0) or (svm.train_y[alpha_i] * error_i < 0 ) :
        # step 1: select alpha j  
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)  
        alpha_i_old = svm.alphas[alpha_i].copy()  
        alpha_j_old = svm.alphas[alpha_j].copy() 
        print ('Find alpha_i %d  alpha_j %d ' % (alpha_i,alpha_j))
        # step 2: calculate eta (the similarity of sample i and j)  
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] - svm.kernelMat[alpha_j, alpha_j]  
        if eta >= 0:  
            return 0  
  
        # step 3: update alpha j  
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta  
        if svm.alphas[alpha_j] <0:
            svm.alphas[alpha_j] = 0 
  
        # step 4: if alpha j not moving enough, just return       
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updateError(svm, alpha_j)
            print ('PPPPPP  %d  %d' % (alpha_i,alpha_j))
            return 0  

        # step 5: update alpha i after optimizing aipha j  
        # y1*y2 == y1/y2  因为都是 +1 -1
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])  
        if svm.alphas[alpha_i] < 0 :
            svm.alphas[alpha_i] = 0
  
        # step 6: update threshold b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old)* svm.kernelMat[alpha_i, alpha_i]- svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old)* svm.kernelMat[alpha_i, alpha_j]

        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old)* svm.kernelMat[alpha_i, alpha_j]- svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old)* svm.kernelMat[alpha_j, alpha_j]  
        svm.b = (b1 + b2) / 2.0  
  
        # step 7: update error cache for alpha i, j after optimize alpha i, j and b  
        updateError(svm, alpha_j)  
        updateError(svm, alpha_i)  
  
        return 1  
    else:
        #print ('innerLoop no loop %d ' % (alpha_i))
        return 0  
  
  
# the main training procedure  
def trainSVM(train_x, train_y, maxIter, kernelOption = ('rbf', 1.0)):  
    # calculate training time  
    startTime = time.time()  
  
    # init data struct for svm  
    svm = SVMStruct(mat(train_x), mat(train_y),  kernelOption)  
      
    # start training  
    entireSet = True  
    alphaPairsChanged = 0  
    iterCount = 0  
    # Iteration termination condition:  
    #   Condition 1: reach max iteration  
    #   Condition 2: no alpha changed after going through all samples,  
    #                in other words, all alpha (samples) fit KKT condition  
    while (iterCount < maxIter) :#and ((alphaPairsChanged > 0) or entireSet):  
        alphaPairsChanged = 0  
  
        # update alphas over all training examples  
        if entireSet:  
            for i in range(svm.numSamples):  
                alphaPairsChanged += innerLoop(svm, i)  
            print ('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)) 
            iterCount += 1  
        # update alphas over examples where alpha is not 0 & not C (not on boundary)  
        else:  
            #nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            nonBoundAlphasList = nonzero((svm.alphas.A > 0) )[0]  
            for i in nonBoundAlphasList:  
                alphaPairsChanged += innerLoop(svm, i)  
            print ('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged,))
            iterCount += 1  
  
        # alternate loop over all examples and non-boundary examples  
        if entireSet:  
            entireSet = False  
        elif alphaPairsChanged == 0:  
            entireSet = True  
  
    print ('Congratulations, training complete! Took %fs! iterCount %d' % (time.time() - startTime, iterCount) )
    return svm  
  
  
# testing your trained svm model given test set  
def testSVM(svm, test_x, test_y):
    test_x = mat(test_x)  
    test_y = mat(test_y)  
    numTestSamples = test_x.shape[0]  
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]  
    supportVectors      = svm.train_x[supportVectorsIndex]  
    supportVectorLabels = svm.train_y[supportVectorsIndex]  
    supportVectorAlphas = svm.alphas[supportVectorsIndex]  
    matchCount = 0  
    for i in range(numTestSamples):  
        kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)
        #print ('kernelValue %s' % (kernelValue)) # 就是那个f(x) 公式
        predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b  
        #if sign(predict) == sign(test_y[i]):  
        if abs(predict) >= 1:
            matchCount += 1
        else:
            print ('Error %d %f' % (i,predict) )
        if abs(predict) < 1:
            svm.flag[i] = 1
    accuracy = float(matchCount) / numTestSamples  
    return accuracy  
  
  
# show your trained svm model only available with 2-D data  
def showSVM(svm):  
    if svm.train_x.shape[1] != 2:  
        print ("Sorry! I can not draw because the dimension of your data is not 2!" ) 
        return 1  
  
    # draw all samples 
    #plt.plot(-1, -1, 'og')  
    #plt.plot(3, 3, 'og')  

    print ('svm.numSamples: %d' % (svm.numSamples))
    for i in range(svm.numSamples):  
        if svm.flag[i] == 1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')
        else:
            if svm.train_y[i] == -1:  
                plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')  
            elif svm.train_y[i] == 1:  
                plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')  
  
    # mark support vectors  
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]  
    #for i in supportVectorsIndex:  
    #    plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')  
      
    # draw the classify line  
    w = zeros((2, 1))  
    for i in supportVectorsIndex:  
        w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)   
    min_x = 0#min(svm.train_x[:, 0])#[0, 0]  
    max_x = 10#max(svm.train_x[:, 0])#[0, 0]
    #print ('min_x %d max_x %d w[0] %f w[1] %f' % (min_x,max_x,w[0],w[1]))
    #if w[0]==0:

    if w[1]==0:
        #y_min_x = min_x
        #y_max_x = max_x
        plt.axvline((min_x+max_x)/2)
        #plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
    else:
        min_y = float(-svm.b - w[0] * min_x) / w[1]
        t1 = w[0] * min_x
        t2 = -svm.b -t1
        print ('w[0]%f  min_x %f t1 %f   t2 %f' % (w[0],min_x,t1,t2))
        max_y = float(-svm.b - w[0] * max_x) / w[1]  
        plt.plot([min_x, max_x], [min_y, max_y], '-g') 
        print ('min_x:%d max_x：%d min_y:%d  max_y:%d ||w[0] %f w[1]: %f  b: %f' % (min_x,max_x,min_y,max_y,w[0],w[1],svm.b))
        
    plt.show()  