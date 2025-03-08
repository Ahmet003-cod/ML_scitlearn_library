# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:51:35 2025
@author: Huzur Bilgisayar
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error#ortama uzaklık
import numpy as np
diyabet=load_diabetes()
X=diyabet.data
y=diyabet.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train,y_train)
y_pred=tree_reg.predict(X_test)

mse=mean_squared_error(y_test, y_pred)#ortalama uzaklık değerlerini buluyor ama net değil
print("mse=",mse)

nmse=np.sqrt(mse)#karekökünü al ve tahmin doğruluğunu bana daha net gösterir(77 puan ara farklı yaklaşmışızz)
print("nmse=",nmse)