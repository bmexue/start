#coding=utf-8
import operator
import math
from math import log
import time
import numpy as np
import matplotlib.pyplot as plt

#  K聚类算法优点： 实现简单，逻辑清晰，缺点：不好预先确定K，难以跳出局部最优解，复杂度 O(kmn)
def GetDis(left,right):
    #print " left:" 
    #print left
    #print " right:" 
    #print right
    dis = math.pow(left[0]-right[0],2) + math.pow(left[1]-right[1],2)
    dis = math.pow(dis,0.5)
    return dis

def GetColor(i):
    if i == 0:
        return 'red'
    if i == 1:
        return 'blue'
    if i == 2:
        return 'black'
    return 'green'

def DealKmeans(m,data, listu):
    
    k = len(listu)
    u1 = listu[0]
    u2 = listu[1]
    u3 = listu[2]
    su1 = []
    su2 = []
    su3 = []
    for i in range (10000):
        su1 = []
        su2 = []
        su3 = []

        for j in range(m):
            dis1 = GetDis(u1,data[j,:])
            dis2 = GetDis(u2,data[j,:])
            dis3 = GetDis(u3,data[j,:])
            if dis1 <= dis2 and dis1 <=dis3:
                su1.append(data[j,:])
            if dis2 <= dis1 and dis2 <=dis3:
                su2.append(data[j,:])
            if dis3 <= dis1 and dis3 <=dis2:
                su3.append(data[j,:])

        havenew = 0

        newu1 =  sum(su1)/len(su1)
        if  False  == np.allclose(newu1,u1):
            u1 = newu1
            havenew = 1

        newu2 =  sum(su2)/len(su2)
        if  False  == np.allclose(newu2,u2):
            u2 = newu2
            havenew = 1

        newu3 =  sum(su3)/len(su3)
        if  False  ==np.allclose(newu3,u3):
            u3 = newu3
            havenew = 1
        if havenew == 1:
            break


    plt.figure(1) 
    for i in range(len(su1)):
        plt.scatter(su1[i][0],su1[i][1],color='red')

    for i in range(len(su2)):
        plt.scatter(su2[i][0],su2[i][1],color='black')

    for i in range(len(su3)):
        plt.scatter(su3[i][0],su3[i][1],color='blue')

    plt.scatter(u1[0],u1[1],color='green')
    plt.scatter(u2[0],u2[1],color='green')
    plt.scatter(u3[0],u3[1],color='green')

    plt.show()

def main():
    m = 100
    #print data
    data = np.random.rand(m,2)
    off = 33
    for i in range(33):
        data[i+off,0] += 1 
    off = 66
    for i in range(33):
        data[i+off,0] += 2

    #print data
    k = 3
    u1 = data[1,:]
    u2 = data[40,:]
    u3 = data[80,:]
    su1 = []
    su2 = []
    su3 = []
    for i in range (10000):
        su1 = []
        su2 = []
        su3 = []

        for j in range(m):
            dis1 = GetDis(u1,data[j,:])
            dis2 = GetDis(u2,data[j,:])
            dis3 = GetDis(u3,data[j,:])
            if dis1 <= dis2 and dis1 <=dis3:
                su1.append(data[j,:])
            if dis2 <= dis1 and dis2 <=dis3:
                su2.append(data[j,:])
            if dis3 <= dis1 and dis3 <=dis2:
                su3.append(data[j,:])

        #get means
        havenew = 0

        newu1 =  sum(su1)/len(su1)
        #print newu1
        #print u1
        if  False  == np.allclose(newu1,u1):
            u1 = newu1
            havenew = 1

        newu2 =  sum(su2)/len(su2)
        if  False  == np.allclose(newu2,u2):
            u2 = newu2
            havenew = 1

        newu3 =  sum(su3)/len(su3)
        if  False  ==np.allclose(newu3,u3):
            u3 = newu3
            havenew = 1
        if havenew == 1:
            break


    plt.figure(1) 
    for i in range(len(su1)):
        plt.scatter(su1[i][0],su1[i][1],color='red')

    for i in range(len(su2)):
        plt.scatter(su2[i][0],su2[i][1],color='black')

    for i in range(len(su3)):
        plt.scatter(su3[i][0],su3[i][1],color='blue')

    plt.scatter(u1[0],u1[1],color='green')
    plt.scatter(u2[0],u2[1],color='green')
    plt.scatter(u3[0],u3[1],color='green')

    plt.show()

def DBSCAN ():
    m = 200
    data = np.random.rand(m,2)
    off = 50
    for i in range(10):
        data[i+off,0] += 1 
    off = 60
    for i in range(60):
        data[i+off,0] += 2 
    off = 120
    for i in range(10):
        data[i+off,0] += 3
    off = 130
    for i in range(60):
        data[i+off,0] += 4
    #step1
    flag = np.zeros(m)
    mindis = 0.5
    midnum = 30
    # 0.8这个参数计算出的族，可能计算多出来,需要合并
    for i in range(m):
        count = 0
        for j in range(m):
            if j != i :
                dis = GetDis(data[i,:],data[j,:])
                if dis < mindis:
                    count += 1
        if count >= midnum:
            flag[i] = 1
    #step2
    #print flag
    #print flag
    setcore = []
    for i in range(m):
        if flag[i] == 1:
            findpre =[]
            for j in range(len(setcore)):
                lines = setcore[j]
                for k in range(len(lines)):
                    index = lines[k]
                    node = data[index,:]
                    #print "error:"
                    #print lines
                    #print data[index,:]
                    #print index
                    #print data[i,:]
                    #print i
                    dis = GetDis(node,data[i,:])
                    if dis<mindis:
                        
                        #lines.append(i)
                        #setcore[j] = lines
                        findpre.append(j)
                        break
            
            if len(findpre) == 0:
                lines = []
                lines.append(i)
                setcore.append(lines)
                #print "setcore"
                #print setcore
            else: #合并
                #print "merge;"
                lines = []
                for j in range(len(findpre)):
                    line = setcore[findpre[j]]
                    for l in range(len(line)):
                        lines.append(line[l])  #????
                lines.append(i)
                setcoretmp = []
                setcoretmp.append(lines)
                for j in range(len(setcore)):
                    find = 0
                    for k in range(len(findpre)):
                        if findpre[k] == j:
                            find = 1
                    if find ==0:
                        lines = setcore[j]
                        setcoretmp.append(lines)
                setcore = setcoretmp
                
    #print len(setcore)

    plt.figure(2) 
    for i in range(len(setcore)):
        lines = setcore[i]
        for j in range(len(lines)):
            plt.scatter(data[lines[j],0],data[lines[j],1],color=GetColor(i))

    plt.show()

    su=[]
    k = len(setcore)
    for i in range(k):
        lines = setcore[i]
        index = lines[0]
        node = data[index,:]
        su.append(node)
    
    DealKmeans(m,data,su)

def SVM():
    m1 = 10
    data1 = np.random.rand(m1,2)
    print (data1[5,0])
    m2 = 10
    data2 = np.random.rand(m2,2)
    
    data2 = data2 + 1
    
    plt.figure(1) 
    for i in range(len(data1)):
        plt.scatter(data1[i,0],data1[i,1],color='red')

    for i in range(len(data2)):
        plt.scatter(data2[i][0],data2[i][1],color='black')

    x = np.arange(0,3.0)
    y=-x+2
    plt.plot(x,y)
    plt.show()

if __name__=='__main__':
    SVM()