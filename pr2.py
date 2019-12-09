import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures #Polinomik Regresyon için gerekli kütüphaneyi dahil ediyoruz.

datas=pd.read_csv("datasets/dollar.csv") #Veri setimin yolunu belirliyorum

X = datas["Day"] #Veri setimizden sütunlarımızı alıyoruz
Y = datas["Price"]

plt.plot(X,Y,"ob")  #Verilerimizi nokta grafiği şeklinde çiziyoruz.

a,b,c = np.polyfit(X,Y,deg = 2) 
"""
2.Dereceden bir denklem y = ax²+bx+c olacağı için a b ve c yazıyoruz. Ardından derecemizi belirtiyoruz.
"""

plt.plot(X,a*X**2+b*X+c,color = "red") # Burada da grafiğimizi çizdiriyoruz.