# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:11:45 2025

@author: Huzur Bilgisayar
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np

california_haues=fetch_california_housing()
#test aşamasi
X=california_haues.data
y=california_haues.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
#tahin ve eğitim aşaması
rf_reng=RandomForestRegressor(random_state=42)
rf_reng.fit(X_train,y_train)
y_pred=rf_reng.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
print("rmse=",rmse)

