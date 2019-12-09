import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures #Polinomik Regresyon için gerekli kütüphaneyi dahil ediyoruz.

datas=pd.read_csv("datasets/dollar.csv") #Veri setimin yolunu belirliyorum

X = datas["Day"] #Veri setimizden sütunlarımızı alıyoruz
Y = datas["Price"]

plt.plot(X,Y,"ob")  #Verilerimizi nokta grafiği şeklinde çiziyoruz.


Q = np.matrix([[len(X),X.sum(),(X**2).sum()],[X.sum(),(X**2).sum(),(X**3).sum()],[(X**2).sum(),(X**3).sum(),(X**4).sum()]])

W = np.matrix([[Y.sum()],[(Y*X).sum()],[(Y*X**2).sum()]])

"""

Burada tamamen matematik formulünden yola çıkarak 2 tane matres oluşturduk ve bu matrisleri çözdürdük.
Burada değerlerden ilki bias değeridir ondan dolayı a[0] değeri bias değeri olarak kullanılmıştır.

"""

a = np.array(np.linalg.solve(Q,W))


plt.plot(X, a[2]*X**2+a[1]*X+a[0])
