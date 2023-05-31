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
from keras.optimizers import Adam
import keras.utils as image



train_path = "/home/syasun/labelled_data/train"
val_path = "/home/syasun/labelled_data/validation"
test_path ="/home/syasun/labelled_data/test"


batch_size = 8
img_height = 500
img_width = 500

#veri oluşturma çoğaltma
from keras.preprocessing.image import ImageDataGenerator

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


from keras import models
from keras import layers

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

model.compile(loss ="binary_crossentropy", optimizer = 'adam',
metrics =['accuracy'])

history = model.fit(train, steps_per_epoch = 20, epochs = 30,
validation_data = valid, validation_steps = 10)



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


#from keras.preprocessing import image
import numpy as np
test_accu = model.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')

preds = model.predict(test,verbose=1)

predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1
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

