# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:31:35 2025

@author: Huzur Bilgisayar
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import  load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
from matplotlib import pyplot as plt
iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

tree_clf=DecisionTreeClassifier(criterion="entropy",max_depth=8,random_state=42)#criterion="entropy"
tree_clf.fit(X_train,y_train)

y_pred=tree_clf.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("iris datasets veri doğruluğu=",accuracy)

conf_metrix=confusion_matrix(y_test, y_pred)
print("confsion metrix=")
print(conf_metrix)
plt.figure(figsize=(10,15))
plot_tree(tree_clf,filled=True,feature_names=iris.feature_names,class_names=list(iris.target_names))
plt.show()

feature_importances=tree_clf.feature_importances_ #¿en önemli featureleri bize gösteriyor
feature_names=iris.feature_names
feature_importest_sort=sorted(zip(feature_importances,feature_names),reverse=True)
for importance ,feature_names in feature_importest_sort:
    print(f"{feature_names}:{importance}")





















