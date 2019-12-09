import numpy as np
import pandas as pd

"""

Multiple lineer regresyon: Birden fazla nicel özellikler arasında ilişki
kurmaya yarayan metotdur.

"""

datas = pd.read_csv("datasets/fish.csv")

X1 = datas["Weight"]
X2 = datas["Width"]
Y = datas["Price"]
"""

Verilerimizi aldık. Burada multiple lineer regresyon kullanacağımız için denklemimiz

Y = a*X1+b*X2+c olacaktır. İki tane özelliğimiz olduğu için X1 ve X2 diye ayırdık.

"""

lr = 0.000001  #learning rate degeri denenerek bulunmuştur. Cost fonksiyonu yardımıyla da bulunabilir.
epoch = 100000

a = 0
b = 0
c = 0

for i in range(epoch):
    Y_ = a*X1+b*X2+c
    
    Ja = ((Y-Y_)*X1).sum()*(-2)/len(X1) #Tamamen lineer regresyon gibi düşünüyoruz.
    Jb = ((Y-Y_)*X2).sum()*(-2)/len(X2)
    Jc = (Y-Y_).sum()*(-2)/len(X1)
    
    a = a-lr*Ja #Gradient Descent ile katsayıları güncelliyoruz.
    b = b-lr*Jb
    c = c-lr*Jc
    
print("Y = {}X1+{}X2+{}".format(a,b,c))