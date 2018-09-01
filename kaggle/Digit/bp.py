# -*- coding: utf-8 -*-
import numpy as np
import math
import operator
import csv
import random
import matplotlib.pyplot as plt
debug = 0

# 采取完全自己的矩阵模式
# 双曲正切函数,该函数为奇函数
def tanh(x):    
    return np.tanh(x)

# tanh导函数性质:f'(t) = 1 - f(x)^2
def tanh_prime(x):      
    #return 1.0 - tanh(x)**2
    return 1.0 - x**2


def ReLU(x):
    return np.tanh(x)

def ReLU_prime(x):
    #return 1.0 - tanh(x)**2
    return 1.0 - x**2

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def softmax_prime(a):
    return a*(1-a)


class NeuralNetwork:
    def __init__(self, layers, activation = 'tanh'):
        """
        :参数layers: 神经网络的结构(输入层-隐含层-输出层包含的结点数列表)
        :参数activation: 激活函数类型
        """
        if activation == 'tanh':    # 也可以用其它的激活函数
            self.activation = tanh
            self.activation_prime = tanh_prime
        else:
            pass

        # 存储权值矩阵
        self.weights = []
        self.biass=[]
        self.dweights = []
        self.dbiass=[]
        self.dweights_last = []
        self.dbiass_last=[]
        self.loss = []
        self.dropout = []

        # range of weight values (-1,1)
        # 初始化输入层和隐含层之间的权值
        for i in range(1, len(layers)):
            r = 2*np.random.random((layers[i], layers[i-1])) -1     # add 1 for bias node
            self.weights.append(r)
            self.dweights.append(r)
            b = 2*np.random.random((layers[i], 1)) -1     # add 1 for bias node
            self.biass.append(b)
            self.dbiass.append(b)
            drop = np.zeros([layers[i],1])
            self.dropout.append(drop)

    def ZeroD(self,A):
        m,n = np.shape(A)
        return np.zeros([m,n])

    def InitDropout(self):
        for k in range(len(self.dropout)):
            m,n = np.shape(self.dropout[k])
            for i in range(m):
                for j in range(n):
                    p = random.random()
                    if p > 0.5:
                        self.dropout[k][i,j] = 1.0
                    else:
                        self.dropout[k][i,j] = 0.0
 
    def GetDrop(self,k,activation):
        m,n = np.shape(self.dropout[k])
        m1,n1= np.shape(activation)
        tmp = self.dropout[k]
        for i in range(n1-1):
            tmp = np.column_stack((tmp,self.dropout[k][:,0]))

        return  tmp * activation
 
    def myfitone(self, X, Y, i , minibatch ):
        lastError = []
        # Return random integers from the discrete uniform distribution in the interval [0, low).
        #if -1 == i:
        #    i = np.random.randint(X.shape[0],high=None)
        a =[]
        X_T = X[ i:i+minibatch, :]
        X_T = X_T.T
        Y_T = Y[ i:i+minibatch]
        Y_T = np.atleast_2d(Y_T).T
        a.append(X_T)   # 从m个输入样本中随机选一组 这样迭代的快一点
        self.InitDropout()
        for l in range(len(self.weights)):
            t1 = np.dot(self.weights[l],a[l])
            t2 = self.biass[l]
            dot_value = np.dot(self.weights[l],a[l]) + self.biass[l]  # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
            if l < (len(self.weights) - 1):
                activation = ReLU(dot_value)
                activation = self.GetDrop(l,activation)
                a.append(activation) # activation 还是2维数组
            else:
                activation = softmax(dot_value)
                a.append(activation) # activation 还是2维数组
            # so a 里面保存了每一层的输出
        error =  Y_T - a[-1]    # 计算输出层delta  是二维向量
        m,n = np.shape(error)
        assert(m==10)
        assert(n==minibatch)
        lastError = error
        deltas = []
        sL = error * softmax_prime(a[-1])
        deltas.append(sL)   # 内积,传递a[-1] 是否错误的？

        # 从倒数第2层开始反向计算delta
        index = len(self.weights) - 2
        for l in range(len(a) - 2, 0, -1):                
            left = np.dot(self.weights[l].T,deltas[-1])  #[q,1]
            delta = left*ReLU_prime(a[l])
            delta = self.GetDrop(index,delta)
            deltas.append(delta)
            index = index-1

        # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
        deltas.reverse()    # 逆转列表中的元素

        # backpropagation
        # 1. Multiply its output delta and input activation to get the gradient of the weight.
        # 2. Subtract a ratio (percentage) of the gradient from the weight.
        for i in range(len(self.weights)):  # 逐层调整权值
            aa = a[i]
            de = deltas[i]
            tmpw = np.dot(de,aa.T)
            m,n = np.shape(self.dweights[i])
            m1,n1 = np.shape(tmpw)
            assert(m==m1)
            assert(n==n1)
            self.dweights[i] += tmpw / minibatch
            tmpb = de

            m,n = np.shape(tmpb)            
            m1,n1 = np.shape(self.dbiass[i])
            assert(m==m1)
            assert(n==minibatch)
            bb = np.sum(tmpb,axis=1)
            bb = np.atleast_2d(bb).T
            m,n = np.shape(bb)
            assert(m==m1)
            self.dbiass[i] += bb/minibatch

        return error

    def GetError(self,error):
        errora = abs(error)
        return  np.sum(errora)

    def CheckAccuracyOne(self,X,Y):
        m,n=np.shape(X)
        right = 0
        for i in xrange(m):
            res = self.predict(X[i])
            maxf = 0.0
            bestindex = -1
            for k in xrange(10):
                if res[k] > maxf:
                    maxf = res[k]
                    bestindex = k
            if 1 == Y[i][bestindex]:
                right += 1
        return right,m

    def CheckAccuracy(self, X, Y, X_checkData,Y_checkData):
        right_train,total_train = self.CheckAccuracyOne(X,Y)
        right_check,total_check = self.CheckAccuracyOne(X_checkData,Y_checkData)
        print "CheckAccuracy:right_train %d,total_train %d  %f ;right_check %d,total_check %d %f" % (right_train,total_train,float(right_train)/float(total_train),right_check,total_check,float(right_check)/float(total_check))
        return float(right_train)/float(total_train),float(right_check)/float(total_check)

    def GetCostJ(self,A):
       
        m,n = np.shape(A)
        print "GetCostJ %d %d" % (m,n)
        R = np.zeros([m,n])
        for i in xrange(m):
            for j in xrange(n):
                R[i,j] = A[i,j]*A[i,j]
        return R
        

    def GetDerR(self,dd):
        return np.sum(dd)

    def PrintWBAbs(self):
        allw = 0.0
        allb = 0.0
        for i in range(len(self.weights)):
            wi =abs( self.weights[i])
            allw += np.sum(wi)
            bi=abs( self.biass[i])
            allb += np.sum(bi)
        print "abs w and b %f  %f" % (allw,allb)  
 
        
# X,Y 第一维表示样例
    def myfit(self, X, Y, X_checkData,Y_checkData,minibatch=32,learning_rate=0.01, epochs=400,checkcount=10000):
        # 0.01  -2 ===>  0.001 -3   ==> 0.0001  -4
        # 指数递减  38682  历史新高
        step = 1 / epochs    #  dropout 需要把这个参数调小点试试   原来是200
        index = -2  
        m,n = np.shape(X)
        times_mini = m/minibatch #最后几个数据放弃学习
        Losslist = []
        AccuracyTrain = []
        AccuracyCheck = []
        momentum = 0.9
        lasttotalloss = -1.0
        for k in range(epochs):     # 训练固定次数
            learning_rate = math.pow(10,index)
            index -= step

            totalloss = 0.0
            
            for i in xrange (times_mini):
                for j in range(len(self.weights)):  # 逐层调整权值
                    self.dweights[j] = self.ZeroD(self.dweights[j])
                    self.dbiass[j] = self.ZeroD(self.dbiass[j])

                loss = 0.0
                thisbatch = minibatch
                error = self.myfitone(X,Y,i*minibatch,minibatch)
                loss = self.GetError(error)
                totalloss += np.sum(loss/minibatch)
                if i == 900 and k % 100 == 0:
                    self.PrintWBAbs()
                    #print "self.biass[4]"
                    #print self.biass[3]
                    #print "self.dbiass[4]"
                    #print self.dbiass[3]

                L2 = 0.99999
                # 动量加速  有个缺点，就是从破下来，会向 前冲很远
                for j in range(len(self.weights)):  # 逐层调整权值
                   #self.weights[j] = 0.99999 * self.weights[j]  #L2
                   if len(self.dweights_last) > 0:
                       self.dweights[j] = (1-momentum)* self.dweights[j] + momentum*self.dweights_last[j]
                       self.dbiass[j] = (1-momentum) * self.dbiass[j] + momentum * self.dbiass_last[j]
                       self.weights[j]  += learning_rate * self.dweights[j]
                       self.biass[j]  += learning_rate * self.dbiass[j]
                   else:
                       self.weights[j]  += learning_rate *(self.dweights[j] )
                       self.biass[j]  += learning_rate * (self.dbiass[j] )
                #self.dweights_last = self.dweights  #close momentum  dropout + momentum 导致速度太慢..
                self.dbiass_last = self.dbiass
            #if lasttotalloss > 0 and totalloss >lasttotalloss :
            #    momentum = momentum/2
            lasttotalloss = totalloss
            print "epochs  k: %d loss %f momentum %f" % (k,totalloss,momentum)
            if k % 10 ==0 :
                Losslist.append(totalloss/times_mini)
                f1,f2 = self.CheckAccuracy(X, Y, X_checkData,Y_checkData)
                AccuracyTrain.append(f1)
                AccuracyCheck.append(f2)

        plt.figure(1) # 创建图表1
        x = np.linspace(0, epochs-1, epochs-1)
        print "Len Plot %d %d" % (len(Losslist),epochs)
        Losslist = Losslist
        plt.plot(Losslist)
        
        plt.plot(AccuracyTrain)
        plt.plot(AccuracyCheck)
        plt.show()

    def predict(self, x): 
        self.InitDropout()
        a = np.atleast_2d(x).T
        for l in range(0, len(self.weights)):               # 逐层计算输出
            if l<len(self.weights)-1:
                a = ReLU(np.dot(self.weights[l],a ) + self.biass[l])
                a = self.GetDrop(l,a)
            else:
                a = softmax(np.dot(self.weights[l],a ) + self.biass[l])
        return a
# 有时候结果不太理想，可能是落入一个局部最优解了

def toInt(array):
    array=mat(array)
    m,n=np.shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n=np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

    
def readCSVFile(file):
    rawData=[]
    trainFile=open(file,'rb')
    reader=csv.reader(trainFile)
    for line in reader:
        rawData.append(line) # 42001 lines
    rawData.pop(0)
    intData=np.array(rawData).astype(np.int32)
    return intData
    
def loadTrainingData():
    intData=readCSVFile("train.csv")
    label=intData[:,0]
    data=intData[:,1:]
    data=np.where(data>0,1,0)
    return nomalizing(data),label
    
def loadTestData():
    intData=readCSVFile("test.csv")
    data=np.where(intData>0,1,0)
    return nomalizing(data)

def saveResult(result):
    with open('result_my_neural_network.csv','wb') as myFile:
        myWriter=csv.writer(myFile)
        tmpl=['ImageId','Label']
        myWriter.writerow(tmpl)
        line = 1;
        for i in result:
            tmp = []
            tmp.append(int(line))
            tmp.append(int(i))
            line = line + 1
            myWriter.writerow(tmp)
        myFile.close()

def Test101():
    trainData,tranLabel=loadTrainingData()
    testData=loadTestData()
    m,n = np.shape(trainData)
    print m,n
    m,n = np.shape(testData)
    print m,n

    Y = []
    for i in xrange(len(tranLabel)):
        yy = [0,0,0,0,0,0,0,0,0,0]
        num = tranLabel[i]
        yy[num] = 1
        Y.append(yy)
    m,n= np.shape(Y) #一维数组
    print "Y shape:"
    print m,n

    trainnum = 32000
    X_trainData = trainData[0:trainnum,:]
    Y_trainData = Y[0:trainnum]

    X_checkData = trainData[trainnum:,:]
    Y_checkData = Y[trainnum:]
    #  784,100,40,20,10  比784 50  25 15 10 更好
    #  对于784的输入，直接用15个神经元的效果不好，应该是损失了太多信息导致的，增加一下效果的确好了点
    nn = NeuralNetwork([784,200,100,50,10])  # 网络结构: 2输入1输出,1个隐含层(包含2个结点)
    #nn = NeuralNetwork([784,100,40,20,10])
    nn.myfit(X_trainData, Y_trainData,X_checkData,Y_checkData,32,0.01,2001)                    # 训练网络

    resultList=[]
    m,n=np.shape(testData)
    for i in xrange(m):
        tmp = nn.predict(testData[i])
        maxf = 0.0
        bestindex = -1
        for k in xrange(10):
            if tmp[k] > maxf:
                maxf = tmp[k]
                bestindex = k
        resultList.append(bestindex)
    saveResult(resultList)
    
    resultListL = []
    m,n=np.shape(trainData)
    pe = 0
    for i in xrange(m):
        res = nn.predict(trainData[i])
        maxf = 0.0
        bestindex = -1
        for k in xrange(10):
            if res[k] > maxf:
                maxf = res[k]
                bestindex = k
        resultListL.append(bestindex)
        if bestindex != tranLabel[i] and pe < -1:
            print "\n"
            print "real value %d, predict value %d" % (tranLabel[i],bestindex)
            print res.T
            pe += 1
    right = 0
    for i in xrange(m):
        if resultListL[i] ==tranLabel[i]:
            right = right + 1
    print "right %d  totle %d Accuracy: %f" % (right,m,float(right)/float(m))

if __name__ == '__main__':
    Test101()