# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:51:55 2025

@author: Huzur Bilgisayar
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib import pyplot as plt
import numpy as np

iris=load_iris()#veri seti oluşturma

import warnings
warnings.filterwarnings("ignore")#terminaldeki uyarıları gizler
n_clases=len(iris.feature_names)
plot_colors="ryb"#renk seçimmi red-yellow-blue(üç farklı çiçek için)

for pairinx,pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    X=iris.data[:,pair]
    y=iris.target
    clf=DecisionTreeClassifier().fit(X, y)
    ax=plt.subplot(2,3,pairinx+1)#2 satır ve 3 sütündan olşan bir grafiklet dizisi
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)#h_pad =yatay boşluk,w_pad=dikey boşluk,had=dolgu_miktarı
    DecisionBoundaryDisplay.from_estimator(clf,
                                           X,
                                           cmap=plt.cm.RdYlBu,
                                           response_method="predict",
                                           ax=ax,
                                           xlabel=iris.feature_names[pair[0]],
                                           ylabel=iris.feature_names[pair[1]],)
    for i,color in zip(range(n_clases),plot_colors):
        idx=np.where(y==i)
        plt.scatter(X[idx,0],X[idx,1],c=color,label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu,
                    edgecolors="black")
plt.legend()

    
    
    
    
    
    
    
    
    
    
    
    
        