import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2 
import os 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import glob
from sklearn.model_selection import train_test_split

images = []
labels = []
for filename in glob.glob("/home/syasun/labelled_data/CXR_labelled/*.png"):
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img = img / 255.0 
    images.append(img)
    label = 1 if filename[-6:-4] == "11" else 0
    labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = np.array(y_train)
y_train = np.expand_dims(y_train, axis=-1) 
y_train = y_train.reshape(563,1)


y_test = np.array(y_test)
y_test = np.expand_dims(y_test, axis=-1) 
y_test = y_test.reshape(141,1)


x_train = x_train.reshape(563, 4096)
x_test = x_test.reshape(141, 4096)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print("y_train",y_train)
print("y_test", y_test)


learning_rate=0.01
hidden_layer_act='relu'
output_layer_act='sigmoid'
no_epochs=50
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

model = Sequential()
model.add(Dense(128, activation=hidden_layer_act))
model.add(Dense(64, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))

opti=optimizers.Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=opti, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = len(x_train), epochs=no_epochs, 
		verbose=1, validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

predictions = model.predict(x_test)
print("prediction:", (predictions[0]*100))
print("predictions shape :",predictions.shape)

y_predictions = predictions.copy()
y_predictions[y_predictions <= 0.5] = 0
y_predictions[y_predictions > 0.5] = 1
print("y_predictions:", y_predictions)
print("y_predictions shape :",y_predictions.shape)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, label='Training accuracu')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, roc_auc_score
y_pred = model.predict(x_test)

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

cmn = confusion_matrix(y_test, y_predictions)
print("confusion matrisi", cmn)


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


plot_confusion_matrix(cmn, classes= range(2))

print(classification_report(y_test, y_predictions))

print("y_predictions shape :",y_predictions.shape)
print("y_test shape:",y_test.shape)
print("x_test shape:",x_test.shape)
print("y_pred shape :",y_pred.shape)
print("fpr_cnn shape :",fpr_cnn.shape)
print("tpr_cnn shape :",tpr_cnn.shape)
print("thresholds_cnn shape :",thresholds_cnn.shape)

