# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 21:31:28 2025
@author: Huzur Bilgisayar
"""
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
oli=fetch_olivetti_faces()
plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(oli.images[i+375],cmap="gray")
    plt.axis("off")#yan kenardai dereceleri off sayesinde kaldırır
plt.show()

X=oli.data
y=oli.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42 )
n_values=[]
accuraccy_values=[]
for n in [15,75,150,200]:
    rf_clf=RandomForestClassifier(n_estimators=n,random_state=42)#n_estimators bu ne kadar artırırsan doğruluk oranı artıyor
    rf_clf.fit(X_train,y_train)
    y_pred=rf_clf.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    accuraccy_values.append(accuracy)
    n_values.append(n)
   
plt.plot(n_values,accuraccy_values,marker="o",linestyle="-",color="red")
plt.title("n değerine göre doğruluk değeri")
plt.xlabel("n değeri")
plt.ylabel("Doğruluk")
plt.xticks(n_values)
plt.grid(True)
plt.show()
    



