# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model



# Function to show the resutls of linear fit model
def show_linear_line(X_parameters,Y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()

# Function for Fitting our data to Linear model
def linear_model_main(X_parameters,Y_parameters,predict_value):

    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

def handLineRegregssionssion():
    x_p = []
    y_p = []
    x_p.append([150])
    x_p.append([200])
    x_p.append([250])
    x_p.append([300])
    x_p.append([350])
    x_p.append([400])
    x_p.append([600])

    y_p.append(6450)
    y_p.append(7450)
    y_p.append(8450)
    y_p.append(9450)
    y_p.append(11450)
    y_p.append(15450)
    y_p.append(18450)

    show_linear_line(x_p,y_p)

    predictvalue = 700
    result = linear_model_main(x_p,y_p,predictvalue)
    print "Intercept value " , result['intercept']
    print "coefficient" , result['coefficient']
    print "Predicted value: ",result['predicted_value']

if __name__ == '__main__':
   handLineRegregssionssion()