import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures #Polinomik Regresyon için gerekli kütüphaneyi dahil ediyoruz.

datas=pd.read_csv("datasets/dollar.csv") #Veri setimin yolunu belirliyorum

X=np.array(datas["Day"]).reshape(-1,1)  #Veri setimizden sütunlarımızı alıyoruz
Y=np.array(datas["Price"]).reshape(-1,1)
"""
Verilerimize reshape fonksiyonunu uyguladık. Bunun sebebi verileri
direk aldığımızda 1xN boyutunda bir matris olarak almasındandır.
Yani verilerin yatay olmasıdır. [1,2,3,4,5] gibi
Eğer biz reshape(-1,1) yaparsak verilerimiz
[1
 2
 3
 4
 5]  gibi olur. Bunu yapmamızın sebebi scikitlearn kütüphanesinin verileri
böyle istemesindendir.

"""

plt.plot(X,Y,"ob")  #Verilerimizi nokta grafiği şeklinde çiziyoruz.

polinom = PolynomialFeatures(degree = 2) # Polinom derecesini belirtiyoruz

X_ = polinom.fit_transform(X,Y) # X ve Y yi polinom fonksiyonuna göre yeniden şekillendiriyoruz.

lr = LinearRegression() 
"""

Her ne kadar polinomik regresyon ile ilgilensek de burada lineer regresyon ile ilgileniyor gibi durabilir.
Burada lineer regresyon sınıfını kullanmamızın sebebi matris çözümünü yapmasındandır. Özel bir sebebi yok. 

"""

lr.fit(X_,Y) #Verilerimizi fit ediyoruz

plt.plot(X,lr.predict(X_),color = "red") #Polinomumuzun grafiğini çizdiriyoruz.