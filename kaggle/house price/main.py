import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

#set 'png' here when working on notebook
#%config InlineBackend.figure_format = 'retina' 
#%matplotlib inline




def handwritingClassTest():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print ("read data finished")
    train.head()
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
        test.loc[:,'MSSubClass':'SaleCondition']))
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
    prices.hist()
   
    #log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data = pd.get_dummies(all_data)
    #filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    #creating matrices for sklearn:
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

if __name__ == '__main__':
   handwritingClassTest()