#The necessary TensorFlow libraries and basic machine learning libraries required to build the model are being loaded.

import os
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Input, AveragePooling2D
import keras.utils as image
from keras import models
from keras import layers


#The folder paths for training, validation, and test data are specified using train_path, val_path, and test_path


train_path = "/home/syasun/labelled_data/train"
val_path = "/home/syasun/labelled_data/validation"
test_path ="/home/syasun/labelled_data/test"

#The model parameters, including batch size and image dimensions, are defined. The batch size determines into how many iterations the dataset will be divided in each epoch.
batch_size = 8
img_height = 500
img_width = 500


#With the ImageDataGenerator library, we load the dataset in real-time. Data augmentation can also be performed in this section. 
#We use the training data to train the model and the validation data to gain insight into the model's performance on real-world data. This phase is an important step for tuning the model's hyperparameters. 
#Finally, the model's performance is evaluated on the test data, which it has never seen before.
#We could say that the test data and validation data are like cousins.

image_gen = ImageDataGenerator(rescale = 1./255)

test_data_gen = ImageDataGenerator(rescale = 1./255)

train = image_gen.flow_from_directory(
      train_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary',
      batch_size=batch_size,
      )

test = test_data_gen.flow_from_directory(
      test_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      shuffle=False, 
      class_mode='binary',
      batch_size=batch_size,
   
      )
valid = test_data_gen.flow_from_directory(
      val_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary', 
      batch_size=batch_size,
      
      )



#This section provides a visualization of the dataset through some sample examples.

plt.figure(figsize=(12, 12))
for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    for X_batch, Y_batch in train:
        image = X_batch[0]        
        dic = {0:'normal', 1:'Sick'}
        plt.title(dic.get(Y_batch[0]))
        plt.axis('off')
        plt.imshow(np.squeeze(image),cmap='gray',interpolation='nearest')
        break
plt.tight_layout()
plt.show()


#The model is defined as a Sequential structure. First, Conv2D and MaxPooling2D layers are added, and then feature maps are flattened using a Flatten layer. 
#The final part of the neural network is formed with a Dense layer, and the output layer provides a single-class output (binary classification) with sigmoid activation.
#In each convolutional layer, filters of sizes 32, 64, and 128 are applied sequentially, using a kernel size of 3x3.

#The Sequential function is a Keras function that allows layers to be connected in a sequential manner.

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation ='relu', input_shape =(500, 500, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation ='relu',input_shape =(500, 500, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation ='relu',input_shape =(500, 500, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation ='relu'))
model.add(layers.Dense(1, activation ="sigmoid"))

model.summary()

from keras import optimizers

#The model is compiled with the binary crossentropy loss function and the Adam optimization method.
#Parameter updates are made based on the accuracy metric.

#The model is trained using the train and valid datasets with model.fit(). It runs for 30 epochs in 20 steps.
model.compile(loss ="binary_crossentropy", optimizer = 'adam',
metrics =['accuracy'])

#The model is trained using the train and valid datasets with model.fit(). It runs for 30 epochs in 20 steps.


#The history variable stores the outputs obtained by the model during training.

history = model.fit(train, steps_per_epoch = 20, epochs = 30,
validation_data = valid, validation_steps = 10)


#This section is where the model results are visualized. 
#The plot_model_history function plots the accuracy and loss values obtained during the training and validation processes.

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    #axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    #axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
plot_model_history(history)


#The model's performance on the test data is calculated using model.evaluate(test), and the accuracy value is stored in the test_accu variable. 
#Predictions are obtained on the test data using model.predict(), and classifications are made based on probabilities.

test_accu = model.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')

#If the predicted probability for a class is greater than 50%, the prediction is assigned to that class.
preds = model.predict(test,verbose=1)
predictions = preds.copy()
predictions > 0.5).astype(int)

#A graph is created to display the model's predictions on the test data along with the actual labels.
####!
test.reset()
x=np.concatenate([test.next()[0] for i in range(test.__len__())])
y=np.concatenate([test.next()[1] for i in range(test.__len__())])
print(x.shape)
print(y.shape)
####!
dic = {0:'normal', 1:'Sick'}
plt.figure(figsize=(20,20))
for i in range(2, ):
  plt.subplot(2, 2, i+2)
  if preds[i, 0] >= 0.5: 
      out = ('{:.2%} probability of being sick case'.format(preds[i][0]))
      
      
  else: 
      out = ('{:.2%} probability of being Normal case'.format(1-preds[i][0]))
plt.title(out+"\n Actual case : "+ dic.get(y[i]))    
plt.imshow(np.squeeze(x[i]))
plt.axis('off')
plt.show()

