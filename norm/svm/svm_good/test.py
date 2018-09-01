from numpy import *  
from svm import *
import svm as SVM
import numpy as np


aa = [[1,2,3],[4,5,6]]
#aa = np.array(a)
bb = [[1,2],[4,5],[3,6]]
#bb= np.array(b)
print(aa)
aa = mat(aa)
print(aa)
bb = mat(bb)
cc = np.dot(aa, bb)
dd = aa*bb

a = [1, 2, 3]
d = [1, 2, 3]

e = []
f = []
e.append([1, 2])
e.append([3, 4])
e.append([5, 6])
e = mat(e)
f = e[0]
print ('e')
print (e)
print ('f')
print (f)
g =e*(f.T)
print (g)
