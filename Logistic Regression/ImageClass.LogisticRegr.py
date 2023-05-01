import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2 
import os 
from random import shuffle 
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import glob
from sklearn.model_selection import train_test_split

images = []
labels = []
for filename in glob.glob("/home/Share/Safak/MontgomerySet10GB/data/CXR_png/*.png"):
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))  # Resize to 64x64
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img / 255.0  # Normalize pixel values
    images.append(img)
    label = 1 if filename[-6:-4] == "00" else 0  # Determine label from filename
    labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = np.array(y_train)
y_train = np.expand_dims(y_train, axis=-1) # Add an extra dimension in the last axis.
y_train = y_train.reshape(640,1)


y_test = np.array(y_test)
y_test = np.expand_dims(y_test, axis=-1) # Add an extra dimension in the last axis.
y_test = y_test.reshape(160,1)


x_train = x_train.reshape(640, 4096)
x_test = x_test.reshape(160, 4096)

x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

def sigmoid(x):
    s = np.exp(x)/((1 + np.exp(x)))
    return s


#lojistik regrasyonda sınıfların olasılık değerleri tahmin edilir. 
#Verinin ait olduğu sınıfı tahmin ederken bu olasılık 1'den büyük çıkabilir.
#Örneğin boy ve kilo oranlarının çizdirildiği bir grafikte tüm kişilere ait boy-kilo endeksi <=1 ise
#(<0.5 ise zayıf >0.5 ise şişman sınıflandırması yapıldığını varsayalım) bu oranın 1'den büyük olduğu
#yeni bir veri geldiğinde model bunu hangi sınıfa atayacağını bilemez. Bu durumda tüm olasılıkları normalize eden bir fonksiyona
#ihtiyaç duyarız bu da Sigmoid fonksiyonu oluyor.
#Sigmoid fonksiyonu binary bir sınıflandırma için modelden alınan tahmin değerlerini normalize ederek verinin hangi sınıfa ait olduğu olasılığını hesaplar.
#Fonksiyondaki x değeri modelden elde edilen tahmin değeridir. Bu değerleri array haline getirip, fonksiyonun tek tek her bir değer için
#olasılık değeri hesaplamasını sağladık.
#Prediction kısmında bu fonksiyon kullanılarak modelden elde edilen tahmin değerleri normalize edildi yani sınıfların olasılık değerleri hesaplandı.


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

#weigtler ve biaslar modelin parametreleri. 
#Görüntü işlemede ağırlık matrisleri kullanarak görseldeki önemli ve önemsiz alanları belirleriz. Akciger görüntüsünü oluşturan piksellerin değerini arttırarak
#modelin bu kısımları iyi öğrenmesi gerektiğini, kemik vs gibi boş alanları oluşturan piksel değerlerini küçülterek de bu alanları belirleyici faktör olarak kullanmamasını sağlarız.
#y=wx+b burada y modelin çıktısıdır. Lojistik regrasyonda amaç w ve b parametrelerini bulmaktır. 
#Bu definitionda parametreler tanımlandı ve başlangıç şartı olarak ikiside 0 olarak seçildi.
#dim istenilen w vektörünün boyutudur ve (dim,1) w parametresini 2 boyutlu 0 vektörü olarak başlatmak anlamına gelir. 

def propagate(w, b, X, Y):
    #Y=Y.T
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(Y*np.log(A) + (1-Y) * np.log(1-A))
    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)
    grads = {"dw": dw, "db": db}
    return grads, cost


#Modelin parametreleri olan w ve b tanımlandıktan sonra bu parametrelerin değerlerini ileri ve geri yayılım ile hesaplıyoruz.
#Nokta çarpımı yapılmasının sebebi vektörlerle işlem yapıldığı için.
#m: giriş öznitelik vektörü X, input. Number training samples
#y:tahmin edilen etiket
#A:Modelden gelen y değerine sigmoid fonksiyonu uygulandı ve olasılık hesaplandı. (y=np.dot(w, X) + b) bu aynı zamanda lineer bir fonksiyondur bu lineerliği bozmak için sigmoid uygulanır)
#Tahmin y'lerin gerçek değer Y'ler ile olan yakınlığını bilmek için w ve b'yi bilmeliyiz. Bunun için modeli eğitiriz.
#cost fonksiyonu tahmin edilen değerle gerçek değer arasındaki farkı gösterir. Model parametrelerini bulmak için öncelikle cost fonksiyonu hesaplanır. Amaç cost'u minimum yapmak.
#Şimdiye kadar yaptığımız şey, tahminimiz ile temel gerçek arasındaki hatayı tanımlayan "ileriye yayılma" denen şeyi hesaplamaktır.
#Bu noktada, bu hatayı en aza indirmek istiyoruz, bu nedenle parametreleri ayarlamak için öğrendiklerimizi geri yayıyoruz.
#dw:Maliyet fonksiyonunun ağırlığa göre türevi, db:biasın ağırlığa göre türevi.
#Her bir parametrenin maliyet fonksiyonuna göre türevini hesaplamak istiyoruz. Bunu, her bir parametredeki değişikliğin nihai kaybı ne kadar etkilediğini hesaplamak için yapıyoruz.
#Son olarak, her döngüde türevleri ve öğrenme oranını göz önünde bulundurarak parametreleri güncellememize izin veren gradyan iniş algoritmasını hesaplıyoruz.
#Böylece, tahminimiz ile temel gerçek arasındaki hatayı en aza indiren w ve b parametrelerini buluyoruz.
####Ağırlık ve input matrisleri birbirinin shapeleri bakımından transpose'u olmalı.
#dw ve db gradyeni bulmak için yapılan backward propagation

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = [] 
    cost_plot = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y) 
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            cost_plot.append(cost)
            print ("Cost after iteration {0} : {1}".format(i, cost))

    params = {"w": w,"b": b}
    grads = {"dw": dw,"db": db}
    plt.plot(cost_plot)
    plt.xticks(rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    return params, grads, costs



#Bu fonksiyon bir gradyan iniş algoritması çalıştırarak w ve b'yi optimize eder.
#Propagate kısmında cost ve gradyanı hesaplamıştık bu kısımda w ve b parametrelerini gradyen iniş kullanarak güncelledik.
#params: w ve bias b ağırlıklarını içeren dictionary,
#grads:cost fonksiyonuna göre ağırlıkların ve biasların gradyanlarını içeren dictionary,
#costs: optimizasyon sırasında hesaplanan tüm costların listesi, bu, öğrenme eğrisini çizmek için kullanılıyor.

def predict(w, b, X):
            m = X.shape[1]
            Y_prediction = np.zeros((1, m))
            w = w.reshape(X.shape[0], 1)
            A = sigmoid(np.dot(w.T, X) + b)
            for i in range(A.shape[1]):
                if A[0, i] > 0.5:
                   Y_prediction[0, i] = 1
                else:
                   Y_prediction[0, i] = 0
            return Y_prediction
#Model parametrelerini(w ve b) optimize ettikten sonra bir tahmin işlemi yapılıyor. 
#A, görselin hasta/sağlıklı olma olasılığını tahmin eden bir vektör. 
#for döngüsü içerisinde tahmin olasılığını 0.5 eşik değerinden geçirerek sınıf ataması yapıyor.

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train" : Y_prediction_train,"w" : w,"b" : b,"learning_rate" : learning_rate,"num_iterations": num_iterations}
    return d

d = model(x_train, y_train, x_test, y_test, num_iterations = 2000, learning_rate = 0.000001)
