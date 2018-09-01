import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


def handwritingClassTest():
    #bring in the six packs
    df_train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print df_train.columns
    print df_train['SalePrice'].describe()
    sns.distplot(df_train['SalePrice'])
    #skewness and kurtosis
    print("Skewness: %f" % df_train['SalePrice'].skew())
    print("Kurtosis: %f" % df_train['SalePrice'].kurt())

    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    f1= plt.figure(1)

if __name__ == '__main__':
   handwritingClassTest()