import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D, regularization
from keras.optimizers import Adam
from tensorflow.keras import initializers
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, roc_auc_score
import sys
import matplotlib

images=[]
labels=[]
images_directory = "/home/Share/Safak/MontgomerySet10GB/data/CXR_png"
for filename in glob.glob(images_directory+"/*.png"):
    image = cv2.imread(filename)  #dosyadan resimleri okur
    image = cv2.resize(image,(64,64)) #bu resimleri 64x64 olarak yeniden boyutlandırır. boyut aynı ratiolar?
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #resimlerin renk türünü belirler
    image = image/255 #resimlerin piksel değerlerini normalize eder
    images.append(image) #ön işlemden geçen resimler yeni halleriyle images listesine doldurulur
    label = 1 if filename[-6:-4] == "_1" else 0 #resimlerin etiketler
    labels.append(label) #etiketleri labels listesine doldurur


x = np.array(images)
y = np.array(labels)
print(labels)
print(x.shape)
print(y.shape)

#Split as train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

print("x_train shape : ",x_train.shape)
print("y_train shape : ",y_train.shape)
print("x_test shape : ",x_test.shape)
print("y_test shape : ",y_test.shape)

#Defining Convolutional Layer:
def conv_layer(convolutional_x, filters):
    convolutional_x = BatchNormalization()(convolutional_x)
    convolutional_x = Activation('relu')(convolutional_x)
    convolutional_x = Conv2D(filters, (3, 3),kernel_initializer='normal', padding='same', use_bias=False)(convolutional_x)
    convolutional_x = Dropout(0.2)(convolutional_x)

    return convolutional_x

#Defining Dense Layer:
def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters

#Defining Transition Layer:
def transition_block(transition_x, tran_filters):
    transition_x = BatchNormalization()(transition_x)
    transition_x = Activation('relu')(transition_x)
    transition_x = Conv2D(tran_filters, (1, 1), kernel_initializer='normal', padding='same', use_bias=False)(transition_x)
    transition_x = AveragePooling2D((2, 2), strides=(2, 2))(transition_x)

    return transition_x, tran_filters

#Create the model
def dense_net(filters, growth_rate, classes, dense_block_size, layers_in_block):
    input_image = Input(shape=(64, 64, 3))
    x = Conv2D(64, (3, 3), kernel_initializer='normal', padding='same', use_bias=False)(input_image)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)
    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)

    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(classes, activation='sigmoid')(dense_x)

    return Model(input_image, output)

#Model Parameters
dense_block_size = 4
growth_rate = 10
classes = 1
layers_in_block = 5
model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
model.summary()

epochs = 80
batch_size = len(x_train)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
history=model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size, shuffle=True)


print("Wait,Creating Plot")
sys.stdout.flush()
matplotlib.use("Agg")
matplotlib.pyplot.style.use("ggplot")
matplotlib.pyplot.figure()
N = epochs
matplotlib.pyplot.plot(np.arange(0, N), history.history["loss"], label="train_loss")
matplotlib.pyplot.plot(np.arange(0, N), history.history["accuracy"], label="train_accuracy")
matplotlib.pyplot.title("X-ray Image Classification")
matplotlib.pyplot.xlabel("Epoch #")
matplotlib.pyplot.ylabel("Loss/Accuracy")
matplotlib.pyplot.legend(loc="lower left")
matplotlib.pyplot.savefig("plot.png")

#Let make a prediction
y_predictions = model.predict(x_test)

#Create a binary result of per prediction
y_pred = y_predictions.copy()
y_pred[y_pred <= 0.4] = 0
y_pred[y_pred > 0.4] = 1
print("y_pred:", y_pred)
print("y_pred shape :",y_pred.shape)

#Visiluaize model metrics

#model accuracy vs loss
E = epochs
print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
matplotlib.pyplot.style.use("ggplot")
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(np.arange(0, E), history.history["loss"], label="train_loss")
matplotlib.pyplot.plot(np.arange(0, E), history.history["accuracy"], label="train_accuracy")
matplotlib.pyplot.title("X-ray Image Classification")
matplotlib.pyplot.xlabel("Epoch #")
matplotlib.pyplot.ylabel("Loss/Accuracy")
matplotlib.pyplot.legend(loc="lower left")
matplotlib.pyplot.savefig("plot.png")

#Create ROC Curve
fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_test, y_pred)
auc_cnn = auc(fpr_cnn, tpr_cnn)
print("y_pred shape :",y_pred.shape)
print("auc_cnn",auc_cnn)
print("y_pred", y_pred)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_cnn, tpr_cnn, label='area = {:.3f}'.format(auc_cnn))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#Model metrics report
print(classification_report(y_test, y_pred))
