import numpy as np
import pandas as pd
import glob
import os
import cv2
import tqdm
import matplotlib as plt
import pathlib
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.applications import DenseNet201
from keras.utils.np_utils import to_categorical
from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization


def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    #model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    return model

densenet = DenseNet201(
    weights=None,
    include_top=False,
    input_shape=(224,224,1)
)

model = build_model(densenet ,lr = 1e-4)
model.summary()

model.save('rapordense.h5')

