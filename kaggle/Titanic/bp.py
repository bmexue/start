# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import operator
import csv
import re
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

        # range of weight values (-1,1)
        # 初始化输入层和隐含层之间的权值
        for i in range(1, len(layers)):
            r = 2*np.random.random((layers[i], layers[i-1])) -1     # add 1 for bias node
            self.weights.append(r)
            b = 2*np.random.random((layers[i], 1)) -1     # add 1 for bias node
            self.biass.append(b)

    # X,Y 第一维表示样例
    def myfit(self, X, Y, learning_rate=0.2, epochs=10000):
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
                if k==0 and debug == 1:
                    print "aa:"
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
    nn.myfit(X, Y)                    # 训练网络
    print "weights:"
    print nn.weights
    # 最后的值波动比较大，说明没有训练的稳定结果，因为这个区域我们没有数据
    T = np.array([[0,0],[0,1],[1,1],[1,0],[0.5,0.5],[0.1,0.1],[0.99,0.01],[0.99,0.2]])
    for s in T:
        print(s, nn.predict(s))     # 测试

def toInt(array):
    array=mat(array)
    m,n=bp.shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n=bp.shape(array)
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
    intData=np.array(rawData)
    return intData
    
def loadTrainData():
    l=[]
    with open('train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
    m,n = np.shape(l)
    print "train.csv %d %d" % (m,n)
    l.remove(l[0])
    m,n = np.shape(l)
    print "train.csv %d %d" % (m,n)
    l.remove(l[:,int(10)])
    m,n = np.shape(l)
    print "train.csv %d %d" % (m,n)

    surived=l[:,1]
    surived = MyInt(surived)
    sex=l[:,4]
    sex = SexInt(sex)
    cls = l[:,2]
    cls = MyInt(cls)
    return surived,sex,cls
 
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

def loadTestData():
    intData=readCSVFile("test.csv")
    data=np.where(intData>0,1,0)
    return data

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

def saveResult(result):
    with open('myresult.csv','wb') as myFile:    
        myWriter=csv.writer(myFile)
        tmpl=['PassengerId','Survived']
        myWriter.writerow(tmpl)
        pid = 892
        for i in result:
            tmp = []
            tmp.append(pid)
            tmp.append(int(i))
            pid = pid + 1
            myWriter.writerow(tmp)
        myFile.close()

    # Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def TestTanic():
    # Load in the train and test datasets
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # Store our passenger ID for easy access
    PassengerId = test['PassengerId']
    print train.head(3)

    full_data = [train, test]

    #  Some features of my own that I have added in
    # Gives the length of the name
    train['Name_length'] = train['Name'].apply(len)
    test['Name_length'] = test['Name'].apply(len)
    # Feature that tells whether a passenger had a cabin on the Titanic
    train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Feature engineering steps taken from Sina
    # Create new feature FamilySize as a combination of SibSp and Parch
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # Create new feature IsAlone from FamilySize
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # Remove all NULLS in the Embarked column
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    # Create a New feature CategoricalAge
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    train['CategoricalAge'] = pd.cut(train['Age'], 5)

    # Create a new feature Title, containing the titles of passenger names
    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    
        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
        # Mapping Fare
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    
        # Mapping Age
        dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
    # Feature selection
    
    
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train = train.drop(drop_elements, axis = 1)
    train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
    test  = test.drop(drop_elements, axis = 1)
    print "--------------------------------------------- Init Data"
    # Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
    keys = train.keys
    y_train = train['Survived'].ravel()
    train = train.drop(['Survived'], axis=1)
    x_train = train.values # Creates an array of the train data
    x_test = test.values # Creats an array of the test data
    
    m,n = np.shape(x_train)
    print m,n
    y_train = np.atleast_2d(y_train).T
    m,n = np.shape(y_train)
    print "y_train shape:"
    print m,n
    m,n = np.shape(x_test)
    print m,n
    # 11,10,1   729
    # 11,10,10,1   549
    nn = NeuralNetwork([11,10,5,1])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)
    nn.myfit(x_train, y_train,0.001,200000)                    # 训练网络

    resultList=[]
    m,n=np.shape(x_test)
    for i in xrange(m):
        tmp = nn.predict(x_test[i])
        if tmp>0.5:
            resultList.append(1)
        else:
            resultList.append(0)
    saveResult(resultList)
   
    
    resultListL = []
    m,n=np.shape(x_train)
    rightcount = 0
    print "keys:"
    #print keys
    for i in xrange(m):
        tmp = nn.predict(x_train[i])
        resultListL.append(tmp)
        s = 0
        if tmp>0.5:
            s =  1
        else:
            s = 0
        if s == y_train[i]:
            rightcount = rightcount + 1
        print "pid %d y_hat  and y : %f  %f " % (i+1,tmp,y_train[i])
        print x_train[i]
    print "keys:"
    #print keys
    #print resultListL
    #print nn.weights
    print "right %d  totle %d" % (rightcount,m)


if __name__ == '__main__':
    TestTanic()
