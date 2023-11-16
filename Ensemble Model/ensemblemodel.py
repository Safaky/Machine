
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import cv2
import glob
import os
from tensorflow.python.framework.ops import Tensor
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.engine import training
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import History
from typing import Tuple, List
#from tensorflow.python.keras.models import Input
from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.applications import DenseNet201
from keras.regularizers import l1_l2
from sklearn.metrics import accuracy_score, precision_score ,recall_score ,f1_score ,roc_auc_score ,confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


####VERİYİ AL

images=[]
labels=[]
images_directory = "/home/syasun/HastaNormal401nocrop"
for filename in glob.glob(images_directory+"/*.png"):
    image = cv2.imread(filename)  #dosyadan resimleri okur
    image = cv2.resize(image,(224,224)) #bu resimleri 64x64 olarak yeniden boyutlandırır. boyut aynı ratiolar?
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #resimlerin renk türünü belirler
    image = image/255 #resimlerin piksel değerlerini normalize eder
    images.append(image) #ön işlemden geçen resimler yeni halleriyle images listesine doldurulur
    label = 0 if filename[-6:-4] == "_0" else 1 #resimlerin etiketler
    labels.append(label) #etiketleri labels listesine doldurur
    
print(labels)

x = np.array(images)
y = np.array(labels)

print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=1)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=1)

print("X_train shape : ",x_train.shape)
print("y_train shape : ",y_train.shape)
print("X_test shape : ",x_test.shape)
print("y_test shape : ",y_test.shape)

input_shape = x_train[0,:,:,:].shape
model_input = Input(shape=input_shape)

####ALEXNET MODEL
def alexnet_cnn(model_input: Tensor) -> training.Model:

    x = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu')(model_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(model_input, x, name='alexnet_cnn')

    return model

alexnet_cnn_model = alexnet_cnn(model_input)

####MODELLERİ EĞİTMEK İÇİN
NUM_EPOCHS = 10

def compile_and_train(model: training.Model, num_epochs: int) -> Tuple [History, str]:
    
    model.compile(loss=binary_crossentropy, optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=64)
    history = model.fit(x=x_train, y=y_train, batch_size=64,
                     epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.20)
    weight_files = glob.glob(os.path.join(os.getcwd(), 'weights/*'))
    weight_file = max(weight_files, key=os.path.getctime)
    return history, weight_file

_, alexnet_weight_file = compile_and_train(alexnet_cnn_model, NUM_EPOCHS)


####MODEL HATA PAYI HESAPLAMAK İÇİN
def evaluate_error(model: training.Model) -> np.float64:
    pred = model.predict(x_test, batch_size = 64)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
 
    return error

evaluate_error(alexnet_cnn_model)


####DENSENET201 MODEL

def densenet_cnn(model_input: Tensor) -> training.Model:
   
    x = (DenseNet201(input_shape=(224,224,3), include_top=False, pooling='max'))(model_input)
    x = (BatchNormalization())(x)
    x = (Dense(2048, activation='relu',  kernel_regularizer=l1_l2(0.01)))(x)
    x = (BatchNormalization())(x)
    x = (Dense(2048, activation='relu',  kernel_regularizer=l1_l2(0.01)))(x)
    x = (BatchNormalization())(x)
    x = (Dense(2048, activation='relu',  kernel_regularizer=l1_l2(0.01)))(x)
    x = (BatchNormalization())(x)
    x = (Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.01)))(x)
    x = (BatchNormalization())(x)
    x = (Dense(1, activation='sigmoid'))(x)

    model = Model(model_input, x, name = 'densenet_cnn')
    return model

densenet_cnn_model = densenet_cnn(model_input)
_, densenet_weight_file = compile_and_train(densenet_cnn_model, NUM_EPOCHS)

evaluate_error(densenet_cnn_model)


####WEIGHT SAVE
ALEXNET_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'alexnet_cnn.01-0.00.hdf5')
DENSENET_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'densenet_cnn.01-0.00.hdf5')

alexnet_cnn_model = alexnet_cnn(model_input)
densenet_cnn_model = densenet_cnn(model_input)

alexnet_cnn_model.load_weights(ALEXNET_WEIGHT_FILE)
densenet_cnn_model.load_weights(DENSENET_WEIGHT_FILE)

models = [alexnet_cnn_model, densenet_cnn_model]

def ensemble(models: List [training.Model], model_input: Tensor) -> training.Model:
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model

ensemble_model = ensemble(models, model_input)
evaluate_error(ensemble_model)

ensemble_error = evaluate_error(ensemble_model)
print("ensemble", ensemble_error)

alex_error = evaluate_error(alexnet_cnn_model)
print("alex", alex_error)

dense_error = evaluate_error(densenet_cnn_model)
print("dense", dense_error)




