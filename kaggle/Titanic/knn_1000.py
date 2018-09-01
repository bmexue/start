'''
0.76555  female 
0.67943  female and cls != 3
'''
#!/usr/bin/python
#-*-coding:utf-8-*-
import sys
from numpy import *
import operator
import csv
def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def MyInt(datas):
    count = len(datas)
    newArray=zeros(count,int)
    index = 0
    for data in datas:
        if data=='1':
            newArray[index] = int(1)
        elif data =='2':
            newArray[index] = int(2)
        elif data =='3':
            newArray[index] = int(3)
        else :
            newArray[index] = int(0)
        index = index + 1;
    return newArray

def SexInt(sexs):
    count = len(sexs)
    newArray=zeros(count,int)
    index = 0
    for sex in sexs:
        if sex=='female':
            newArray[index] = int(1)
        else:
            newArray[index] = int(0)
        index = index + 1;
    return newArray

def loadTrainData():
    l=[]
    with open('train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
    m,n = np.shape(l)
    print "train.csv %d %d" % (m,n)
    l.remove(l[0])
    l=array(l)
   
    surived=l[:,1]
    surived = MyInt(surived)
    sex=l[:,4]
    sex = SexInt(sex)
    cls = l[:,2]
    cls = MyInt(cls)
    return surived,sex,cls
    #return nomalizing(toInt(data)),toInt(label)  #label 1*42000  data 42000*784
    #return data,label
    
def loadTestData():
    l=[]
    with open('test.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
     #28001*784
    l.remove(l[0])
    data=array(l)
    sex=data[:,3]
    sex =SexInt(sex)
    pid=data[:,0]
    return pid,sex
 
    #return nomalizing(toInt(data))  #  data 28000*784

def loadTestResult():
    l=[]
    with open('knn_benchmark.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
     #28001*2
    l.remove(l[0])
    label=array(l)
    return toInt(label[:,1])  #  label 28000*1

#dataSet:m*n   labels:m*1  inX:1*n
def classify(inX, dataSet, labels, k):
    inX=mat(inX)
    dataSet=mat(dataSet)
    labels=mat(labels)
    dataSetSize = dataSet.shape[0]                  
    diffMat = tile(inX, (dataSetSize,1)) - dataSet   
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)                  
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()            
    classCount={}                                      
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def saveResult(pid,result):
    with open('myresult.csv','wb') as myFile:    
        myWriter=csv.writer(myFile)
        tmpl=['PassengerId','Survived']
        myWriter.writerow(tmpl)
        line = 0;
        for i in result:
            tmp = []
            tmp.append(pid[line])
            tmp.append(int(i))
            line = line + 1
            myWriter.writerow(tmp)
        myFile.close()

def handwritingClassTest():
    tr_surived,tr_sex, tr_cls = loadTrainData()
    totalp = len(tr_surived)
    surived_c = 0


    i = 0
    cls_s = [0,0,0,0]

    for s in tr_surived:
        if s == 1:
            c_cls = tr_cls[i]
            cls_s[c_cls]= cls_s[c_cls] + 1
        i = i + 1
    print "surived in cls1 %d cls2 %d cls3 %d" % (cls_s[1],cls_s[2],cls_s[3])
    cls_count=[0,0,0,0]
    for s in tr_cls:
        cls_count[s] = cls_count[s] + 1
    print "pasenger in cls1 %d cls2 %d cls3 %d" % (cls_count[1],cls_count[2],cls_count[3])
    for k in tr_surived:
        if k == 1:
           surived_c = surived_c+ 1
    famele=0
    famele_sv = 0
    m =0
    for s in tr_sex:
        if s == 1:
            famele = famele+ 1
            if tr_surived[m]==1:
                famele_sv = famele_sv+1
        m= m +1
    print "total %d surived %d female %d famele_sv %d" % (totalp,surived_c,famele,famele_sv)
    pid,sex = loadTestData()
    m=len(sex)
    resultList=[]
    
    test_s = 0
    for i in range(m):
        if sex[i]==1:
             resultList.append(1)
             test_s = test_s +1
        elif tr_cls[i] == 1:
             resultList.append(1)
             test_s = test_s +1
        else:
             resultList.append(0)

    print "test total %d surived %d" % (len(resultList),test_s)
    saveResult(pid,resultList)


if __name__ == '__main__':
   handwritingClassTest()