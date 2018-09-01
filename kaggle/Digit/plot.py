# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

# 采取完全自己的矩阵模式
# 双曲正切函数,该函数为奇函数
def tanh(x):    
    return np.tanh(x)

# tanh导函数性质:f'(t) = 1 - f(x)^2
def tanh_prime(x):      
    return 1.0 - tanh(x)**2

def tanh_prime2(x):
    a = tanh(x)
    return 1.0 - tanh(a)**2

def Test2():

    plt.figure(1) # 创建图表1
    x = np.linspace(-10, 10, 100)
    plt.plot(x,tanh(x))
    plt.plot(x,tanh_prime(x))
    plt.plot(x,tanh_prime2(x))
    plt.show()

def Test():
    plt.figure(1) # 创建图表1
    plt.figure(2) # 创建图表2
    ax1 = plt.subplot(211) # 在图表2中创建子图1
    ax2 = plt.subplot(212) # 在图表2中创建子图2
 
    x = np.linspace(0, 3, 100)
    for i in xrange(5):
        plt.figure(1)  #❶ # 选择图表1
        plt.plot(x, np.exp(i*x/3))
        plt.sca(ax1)   #❷ # 选择图表2的子图1
        plt.plot(x, np.sin(i*x))
        plt.sca(ax2)  # 选择图表2的子图2
        plt.plot(x, np.cos(i*x))
 
    plt.show()

def GetGain(x):
    return x*np.log(x) + (1.0-x) * np.log(1-x)

def Gain():
    plt.figure(1) # 创建图表1
    x = np.linspace(0.001, 0.999, 100)
    plt.plot(x,-1*GetGain(x))
    plt.show()

if __name__ == '__main__':
    Gain()