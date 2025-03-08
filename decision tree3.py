# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:21:06 2025
@author: Huzur Bilgisayar
"""
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np

X=np.sort(5*np.random.rand(80,1),axis=0)
y=np.sin(X).ravel()

y[::5]+=5*(0.5-np.random.rand(16))#görültü(kısacası tahmini zorlaştırmak için yapılır)
#plt.scatter(X,y,color="r")

regr_1=DecisionTreeRegressor(max_depth=3)
regr_2=DecisionTreeRegressor(max_depth=15)
regr_1.fit(X,y)
regr_2.fit(X,y)
X_test=np.arange(0.5,5,0.075)[:,np.newaxis]
y_pred_1=regr_1.predict(X_test)
y_pred_2=regr_2.predict(X_test)

plt.figure()
plt.scatter(X, y,color="red",label="data")
plt.plot(X, y,color="red",label="data")
plt.plot(X_test,y_pred_1,label="max_depth=3",color="blue",linewidth=2)
plt.plot(X_test,y_pred_2,label="max_depth=15",color="green",linewidth=2)
plt.legend()
plt.xlabel("data")
plt.ylabel("target")



