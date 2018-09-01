# -*- coding: utf-8 -*-
import numpy as np


# 双曲正切函数,该函数为奇函数
def tanh(x):    
    return np.tanh(x)

# tanh导函数性质:f'(t) = 1 - f(x)^2
def tanh_prime(x):      
    return 1.0 - tanh(x)**2


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

        # range of weight values (-1,1)
        # 初始化输入层和隐含层之间的权值
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1     # add 1 for bias node
            self.weights.append(r)
            print "r:"  # 3*3  感觉不够呀..
            print r
            
        # 初始化输出层权值
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1         
        print "r_o:"  # [3,1]
        print r
        self.weights.append(r)


    def fit(self, X, Y, learning_rate=0.2, epochs=10000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        print X
        # 矩阵左右连接起来 np.ones((X.shape[0],1))  +  X
        X = np.hstack([np.ones((X.shape[0],1)),X])
        print X
        
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
            deltas = [error * self.activation_prime(a[-1])]
            
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

    def predict(self, x): 
        a = np.concatenate((np.ones(1), np.array(x)))       # a为输入向量(行向量)
        for l in range(0, len(self.weights)):               # 逐层计算输出
            a = self.activation(np.dot(a, self.weights[l]))
        return a



if __name__ == '__main__':
    nn = NeuralNetwork([2,2,1])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)

    X = np.array([[0, 0],           # 输入矩阵(每行代表一个样本,每列代表一个特征)
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([0, 1, 1, 0])      # 期望输出

    nn.fit(X, Y)                    # 训练网络

    print 'w:', nn.weights          # 调整后的权值列表
    
    for s in X:
        print(s, nn.predict(s))     # 测试