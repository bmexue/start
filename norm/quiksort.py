# -*- coding: utf-8 -*-
import numpy as np
import math
import operator
import csv
import random
import matplotlib.pyplot as plt
debug = 0



#终于搞明白快排了
def QuikSort(s,left,right):
    if left >= right:
        return  
    i = left
    j = right
    key = s[i]

    while i<j:
        
        while i<j and s[j]>= key:
            j=j-1
        s[i] = s[j]
        while i<j and s[i] <=key:
            i = i+ 1
        s[j] = s[i]
    s[i] = key
    QuikSort(s,left,i-1)
    QuikSort(s,i+1,right)

def TestQuikSort():
    s = []
    for i in xrange(100):
        s.append(random.randint(0, 100))
    QuikSort(s,0,len(s)-1)
    print s


if __name__ == '__main__':
    TestQuikSort()