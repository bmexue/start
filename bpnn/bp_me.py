# -*- coding: utf-8 -*-
import numpy as np

import operator
import csv

debug = 0

# 采取完全自己的矩阵模式
# 双曲正切函数,该函数为奇函数
def tanh(x):    
    return np.tanh(x)

# tanh导函数性质:f'(t) = 1 - f(x)^2
def tanh_prime(x):      
    return 1.0 - tanh(x)**2

def tanh_prime2(fx):
    return 1.0 - fx*fx

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
        self.loss = []

        # range of weight values (-1,1)
        # 初始化输入层和隐含层之间的权值
        for i in range(1, len(layers)):
            r = 2*np.random.random((layers[i], layers[i-1])) -1     # add 1 for bias node
            self.weights.append(r)
            self.dweights.append(r)
            b = 2*np.random.random((layers[i], 1)) -1     # add 1 for bias node
            self.biass.append(b)
            self.dbiass.append(b)

    def ZeroD(self,A):
        m,n = np.shape(A)
        return np.zeros([m,n])
 
    def myfitone(self, X, Y):
        lastError = []
            
        # Return random integers from the discrete uniform distribution in the interval [0, low).
        i = np.random.randint(X.shape[0],high=None)
        a =[]
        a.append(np.atleast_2d([X[i]]).T)   # 从m个输入样本中随机选一组 这样迭代的快一点

        for l in range(len(self.weights)):
            t1 = np.dot(self.weights[l],a[l])
            t2 = self.biass[l]
            dot_value = np.dot(self.weights[l],a[l]) +  self.biass[l]  # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
            activation = self.activation(dot_value)
            a.append(activation) # activation 还是2维数组
            # so a 里面保存了每一层的输出

        error =  np.atleast_2d([Y[i]]).T - a[-1]    # 计算输出层delta  是二维向量
        
        lastError = error
        deltas = []
        sL = error * tanh_prime(a[-1])
        deltas.append(sL)   # 内积,传递a[-1] 是否错误的？

        # 从倒数第2层开始反向计算delta
        for l in range(len(a) - 2, 0, -1):                
            left = np.dot(self.weights[l].T,deltas[-1])  #[q,1]
            tmp = left*tanh_prime(a[l])
            deltas.append(tmp)

        # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
        deltas.reverse()    # 逆转列表中的元素

        # backpropagation
        # 1. Multiply its output delta and input activation to get the gradient of the weight.
        # 2. Subtract a ratio (percentage) of the gradient from the weight.
        for i in range(len(self.weights)):  # 逐层调整权值
            aa = a[i]
            de = deltas[i]
            tmpw = np.dot(de,aa.T)
            self.dweights[i] += tmpw
            tmpb = de
            self.dbiass[i] += tmpb
        return error

# X,Y 第一维表示样例
    def myfitminibatch(self, X, Y, minibatch=8,learning_rate=0.01, epochs=1000):
        for k in range(epochs):     # 训练固定次数
            if k % 1000 == 0: print 'epochs:', k

            for i in range(len(self.weights)):  # 逐层调整权值
                self.dweights[i] = self.ZeroD(self.dweights[i])
                self.dbiass[i] = self.ZeroD(self.dbiass[i])

            loss = 1
            for i in xrange(minibatch):
                error = self.myfitone(X,Y)
                if k % 1000 == 0:
                    print error

            for i in range(len(self.weights)):  # 逐层调整权值
                self.weights[i]  += learning_rate *(self.dweights[i] / minibatch)
                self.biass[i]  += learning_rate * (self.dbiass[i] / minibatch)

        #print self.dweights    

    # X,Y 第一维表示样例
    def myfit(self, X, Y, learning_rate=0.01, epochs=100000):
        lastError = []
        for k in range(epochs):     # 训练固定次数
            if k % 1000 == 0: print 'epochs:', k
            
            # Return random integers from the discrete uniform distribution in the interval [0, low).
            i = np.random.randint(X.shape[0],high=None)
            if k==0 and debug == 1:
                print "--------------------This input:----------------------"
                print np.atleast_2d([X[i]]).T
            a =[]
            a.append(np.atleast_2d([X[i]]).T)   # 从m个输入样本中随机选一组 这样迭代的快一点

            for l in range(len(self.weights)):
                if k==0 and debug == 1:
                    print "self.weights[l]:"
                    print self.weights[l]
                    print "a[l]:"
                    print a[l]
                    print "self.biass[l]:"
                    print self.biass[l]
                t1 = np.dot(self.weights[l],a[l])
                if k==0 and debug == 1:
                    print "t1:"
                    print t1
                t2 = self.biass[l]
                dot_value = np.dot(self.weights[l],a[l]) +  self.biass[l]  # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
                activation = self.activation(dot_value)
                a.append(activation) # activation 还是2维数组
                if k ==0 and debug == 1:
                    print "activation in k %d in l %d" % (k,l)
                    print activation

                    print "t1:"
                    print t1
                    print "t2:"
                    print t2


            # so a 里面保存了每一层的输出
            if k == 0 and debug == 1:
                print "Y[i]:"
                print Y[i]
                print "a[-1]:"
                print a[-1]
            error =  np.atleast_2d([Y[i]]).T - a[-1]    # 计算输出层delta  是二维向量
            lastError = error
            deltas = []
            sL = error * tanh_prime(a[-1])
            if k == 0 and debug == 1:
                print "-----------------------------------tanh_prime(a[-1]  tanh_prime2(a[-1]):"
                print tanh_prime(a[-1])
                print tanh_prime2(a[-1])
            
            if k==0 and debug == 1:
                print "error:"
                print error
                print "tanh_prime2(a[-1]):"
                print tanh_prime2(a[-1])
                print "sL:"
                print sL
            deltas.append(sL)   # 内积,传递a[-1] 是否错误的？
            if k == 0 and debug == 1:
                print "---------k: %d" % (k)
                print "self.activation_prime(a[-1]):"
                print self.activation_prime(a[-1])
                print "error:"
                print error
                print "deltas:"
                print deltas

            # 从倒数第2层开始反向计算delta
            for l in range(len(a) - 2, 0, -1):
                
                left = np.dot(self.weights[l].T,deltas[-1])  #[q,1]
                tmp = left*tanh_prime(a[l])
                if k ==0 and debug == 1:
                    print "Check error AAAA:"
                    print "self.weights[l].T:"
                    print self.weights[l].T
                    print "deltas[-1]:"
                    print deltas[-1]
                    print "left:"
                    print left
                    print "tanh_prime2(a[l]):"
                    print tanh_prime2(a[l])
                    print "deltas.append:"
                    print tmp
                deltas.append(tmp)
                if k==0 and debug == 1:
                    print "Check error:"
                    print self.weights[l]  # why 1 * 2
                    print deltas[-1]

                    print "Update deltas "

            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()    # 逆转列表中的元素

            # backpropagation
            # 1. Multiply its output delta and input activation to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):  # 逐层调整权值
                aa = a[i]
                de = deltas[i]
                tmpw = np.dot(de,aa.T)
                if  debug == 0:
                    print "------aa k %d:" % (k)
                    print aa
                    print "de:"
                    print de
                    print "tmpw:"
                    print tmpw
                self.weights[i]  += learning_rate * tmpw
                tmpb = de
                self.biass[i]  += learning_rate * tmpb

        print "lastError:"
        print lastError

    def fit(self, X, Y, learning_rate=0.2, epochs=10000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        print X
        # 矩阵左右连接起来 np.ones((X.shape[0],1))  +  X
        X = np.hstack([np.ones((X.shape[0],1)),X])
        print X
        lastError
        for k in range(epochs):     # 训练固定次数
            if k % 1000 == 0: print 'epochs:', k

            # Return random integers from the discrete uniform distribution in the interval [0, low).
            i = np.random.randint(X.shape[0],high=None) 
            a = [X[i]]   # 从m个输入样本中随机选一组 这样迭代的快一点
            if k< 5:
                print "--------------a: in k:%d i: %d" % (k,i)
                print a
                print "\n"
                print "Start Loop compute len(weights): %d" % (len(self.weights))
            for l in range(len(self.weights)): 
                dot_value = np.dot(a[l], self.weights[l])   # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
                activation = self.activation(dot_value)
                if k<5:
                    print "dot_value len: %d k: %d" % (len(dot_value),k)
                    print dot_value
                    print "\n"
                    
                a.append(activation)
                    
            # 反向递推计算delta:从输出层开始,先算出该层的delta,再向前计算
            error = Y[i] - a[-1]    # 计算输出层delta
            lastError = error
            deltas = [error * self.activation_prime(a[-1])]  # 内积
            
            # 从倒数第2层开始反向计算delta
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))


            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()    # 逆转列表中的元素


            # backpropagation
            # 1. Multiply its output delta and input activation to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):  # 逐层调整权值
                layer = np.atleast_2d(a[i])     # View inputs as arrays with at least two dimensions
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * np.dot(layer.T, delta) # 每输入一次样本,就更新一次权值
        print "lastError:"
        print lastError

    def predict(self, x): 
        a = np.atleast_2d(x).T
        for l in range(0, len(self.weights)):               # 逐层计算输出
            a = self.activation(np.dot(self.weights[l],a ) + self.biass[l])
        return a
# 有时候结果不太理想，可能是落入一个局部最优解了

def TestCase1():  # 异或
    nn = NeuralNetwork([2,2,1])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)

    X = np.array([[0, 0],           # 输入矩阵(每行代表一个样本,每列代表一个特征)
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0], [1], [1], [0]])      # 期望输出

    nn.myfit(X, Y)                    # 训练网络

    print 'w:', nn.weights          # 调整后的权值列表
    print 'b:', nn.biass            # 调整后的权值列表
    for s in X:
        print(s, nn.predict(s))     # 测试

def TestCase2(): # 只看最高位
    print "TestCase2:"
    X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    Y = np.array([[0,0,0,0,1,1,1,1]]).T
    nn = NeuralNetwork([3,3,2,1])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)
    nn.myfit(X, Y)                    # 训练网络
    for s in X:
        print(s, nn.predict(s))     # 测试

def TestCase3(): # 非线性
    print "TestCase2:"
    X = np.array([[0,0],[0,1],[1,1],[1,0]])
    Y = np.array([[1,0,1,0]]).T
    nn = NeuralNetwork([2,3,2,1])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)
    #nn.myfitminibatch(X, Y)                    # 训练网络
    nn.myfit(X, Y,0.01,10)                    # 训练网络
    print "weights:"
    print nn.weights
    # 最后的值波动比较大，说明没有训练的稳定结果，因为这个区域我们没有数据
    T = np.array([[0,0],[0,1],[1,1],[1,0],[0.5,0.5],[0.1,0.1],[0.99,0.01],[0.99,0.2]])
    for s in T:
        print(s, nn.predict(s))     # 测试

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
    nn = NeuralNetwork([784,15,15,15,15,10])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)
    nn.myfit(trainData, Y,0.01,100000)                    # 训练网络

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
    for i in xrange(m):
        tmp = nn.predict(trainData[i])
        maxf = 0.0
        bestindex = -1
        for k in xrange(10):
            if tmp[k] > maxf:
                maxf = tmp[k]
                bestindex = k
        resultListL.append(bestindex)
    right = 0
    for i in xrange(m):
        if resultListL[i] ==tranLabel[i]:
            right = right + 1
    print "right %d  totle %d" % (right,m)

if __name__ == '__main__':
    #Test101()
    TestCase3()
    #TestCase2()