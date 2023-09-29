import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, StratifiedKFold
import json
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import json

X_shape = 224

####################################################################################

model = load_model('/home/syasun/DenseModel2.h5')
model.summary()
image_path = '/home/Share/Safak/MontgomerySet10GB/data/CXR_png'

####################################################################################

images_montgomery = []  # To store images from the Montgomery dataset
labels_montgomery = []  # To store labels for the Montgomery dataset
images_china = []       # To store images from the ChinaCXR dataset
labels_china = []       # To store labels for the ChinaCXR dataset

# Function to load and preprocess images from a directory
def load_images_from_directory(directory, labels, dataset_label):
    image_files = os.listdir(directory)
    for file_name in image_files:
        if file_name.startswith('MCUCXR'):
            file_dataset_label = 'Montgomery'
        elif file_name.startswith('CHNCXR'):
            file_dataset_label = 'ChinaCXR'
        else:
            continue  # Skip files that don't belong to Montgomery or ChinaCXR
        
        if file_dataset_label == dataset_label:
            if file_name.endswith('0.png'):
                labels.append(0)  # Normal
            elif file_name.endswith('1.png'):
                labels.append(1)  # Abnormal
            else:
                continue  # Skip files that don't have labels

            # Load and preprocess the image
            image = cv2.resize(cv2.imread(os.path.join(directory, file_name)), (X_shape, X_shape))[:, :, 0]
            
            # Append the image to the appropriate dataset
            if dataset_label == 'Montgomery':
                images_montgomery.append(image)
            elif dataset_label == 'ChinaCXR':
                images_china.append(image)

# Load and preprocess images from the directory for both datasets
load_images_from_directory(image_path, labels_montgomery, 'Montgomery')
load_images_from_directory(image_path, labels_china, 'ChinaCXR')

# Convert the lists to NumPy arrays
images_montgomery = np.array(images_montgomery)
labels_montgomery = np.array(labels_montgomery)
images_china = np.array(images_china)
labels_china = np.array(labels_china)

# Count the number of normal (0) and abnormal (1) samples for both datasets
num_normal_montgomery = np.count_nonzero(labels_montgomery == 0)
num_abnormal_montgomery = np.count_nonzero(labels_montgomery == 1)
num_normal_china = np.count_nonzero(labels_china == 0)
num_abnormal_china = np.count_nonzero(labels_china == 1)

# Print the statistics for both datasets
print("Montgomery Dataset:")
print("Number of Normal (0) Samples:", num_normal_montgomery)
print("Number of Abnormal (1) Samples:", num_abnormal_montgomery)

print("\nChinaCXR Dataset:")
print("Number of Normal (0) Samples:", num_normal_china)
print("Number of Abnormal (1) Samples:", num_abnormal_china)

####################################################################################


def augment_data(images, labels, num_augmented_samples, save_dir=None):
    augmented_images = []
    augmented_labels = []

    for i in range(num_augmented_samples):
        # Randomly select an image from the original dataset
        idx = np.random.randint(0, len(images))
        image = images[idx]
        label = labels[idx]

        # Apply random augmentation techniques
        if np.random.choice([True, False]):
            # Flip horizontally with 50% probability
            image = np.fliplr(image)

        if np.random.choice([True, False]):
            # Rotate the image by a random angle between -15 and 15 degrees
            angle = np.random.uniform(-15, 15)
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if np.random.choice([True, False]):
            # Adjust brightness by scaling pixel values
            alpha = np.random.uniform(0.7, 1.3)  # Brightness factor
            image = cv2.multiply(image, np.array([alpha]))

        augmented_images.append(image)
        augmented_labels.append(label)

        # Save the augmented image if save_dir is provided
        if save_dir:
            file_name = f"augmented_{i}_{label}.png"
            cv2.imwrite(os.path.join(save_dir, file_name), image)

    return np.array(augmented_images), np.array(augmented_labels)

# Define the number of augmented samples you want to generate
num_augmented_samples = 600  # Adjust this number as needed

# Directory to save augmented data
save_dir = '/home/syasun/çalışma/preproses/augmented_data'

# Create the directory if it doesn't exist
#os.makedirs(save_dir, exist_ok=True)

# Perform data augmentation on the Montgomery dataset
augmented_images_montgomery, augmented_labels_montgomery = augment_data(images_montgomery, labels_montgomery, num_augmented_samples, save_dir)

# Calculate the counts of new healthy and unhealthy Montgomery data samples
num_new_normal_montgomery = np.count_nonzero(augmented_labels_montgomery == 0)
num_new_abnormal_montgomery = np.count_nonzero(augmented_labels_montgomery == 1)

print("Montgomery Dataset:")
print(f"Number of Normal (0) Samples Before Augmentation: {num_normal_montgomery}")
print(f"Number of Abnormal (1) Samples Before Augmentation: {num_abnormal_montgomery}")
print(f"Number of New Normal (0) Samples After Augmentation: {num_new_normal_montgomery}")
print(f"Number of New Abnormal (1) Samples After Augmentation: {num_new_abnormal_montgomery}")

# Now, augmented data is saved in '/kaggle/working/augmented_data'
X_combined = np.concatenate([images_montgomery, images_china, augmented_images_montgomery], axis=0)
y_combined = np.concatenate([labels_montgomery, labels_china, augmented_labels_montgomery], axis=0)

# Check the shape of X_combined and y_combined
print("Shape of X_combined:", X_combined.shape)
print("Shape of y_combined:", y_combined.shape)

# Check the size (total number of elements) of X_combined and y_combined
print("Size of X_combined:", X_combined.size)
print("Size of y_combined:", y_combined.size)



# Normalize the X (image) data
X_combined_normalized = X_combined / 255.0  # Scale to the [0, 1] range

# Normalize the Y (label) data (assuming it's binary)
Y_combined_normalized = y_combined  # No need to normalize labels for binary classification


# Check the shape of the normalized data
print("Shape of X_combined_normalized:", X_combined_normalized.shape)
print("Shape of Y_combined_normalized:", Y_combined_normalized.shape)

# Check the minimum and maximum pixel values of the normalized X data
min_pixel_value = np.min(X_combined_normalized)
max_pixel_value = np.max(X_combined_normalized)
print("Minimum Pixel Value:", min_pixel_value)
print("Maximum Pixel Value:", max_pixel_value)

####################################################################################


# Step 1: Split the data into initial training and test sets
test_percent = 0.2
X_train_initial, X_test, y_train_initial, y_test = train_test_split(
    X_combined_normalized, Y_combined_normalized, test_size=test_percent, random_state=42)

# Step 2: Apply cross-validation on the initial training set
k_folds = 5  
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize a list to store the test accuracies for each fold
test_accuracies = []
training_histories = []

# Define a list of colors for plotting
colors = ['blue', 'green', 'orange', 'purple', 'brown']

for fold, (train_index, val_index) in enumerate(skf.split(X_train_initial, y_train_initial), start=1):
    X_fold_train, y_fold_train = X_train_initial[train_index], y_train_initial[train_index]
    X_fold_val, y_fold_val = X_train_initial[val_index], y_train_initial[val_index]

####################################################################################

    history = model.fit(X_fold_train, y_fold_train, epochs=100, batch_size=16, validation_data=(X_fold_val, y_fold_val))
    
####################################################################################

    # Save the training history to a JSON file
    history_file_name = f'fold_{fold}_training_history.json'
    with open(history_file_name, 'w') as history_file:
        json.dump(history.history, history_file)

    # Evaluate the trained model on the test data (X_test, y_test)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Fold {fold} - Test Accuracy: {test_accuracy * 100:.2f}%")

    # Store the test accuracy for this fold
    test_accuracies.append(test_accuracy)

    # Store the training history for this fold
    training_histories.append(history)

# Calculate and print the average test accuracy across all folds
avg_test_accuracy = np.mean(test_accuracies)
print(f"Average Test Accuracy: {avg_test_accuracy * 100:.2f}%")

# Initialize lists to store aggregated training history
all_training_loss = []
all_training_accuracy = []
all_validation_loss = []
all_validation_accuracy = []

# Aggregate training history across folds
for history in training_histories:
    all_training_loss.append(history.history['loss'])
    all_training_accuracy.append(history.history['accuracy'])
    all_validation_loss.append(history.history['val_loss'])
    all_validation_accuracy.append(history.history['val_accuracy'])

# Calculate mean values for each epoch
mean_training_loss = np.mean(all_training_loss, axis=0)
mean_training_accuracy = np.mean(all_training_accuracy, axis=0)
mean_validation_loss = np.mean(all_validation_loss, axis=0)
mean_validation_accuracy = np.mean(all_validation_accuracy, axis=0)

####################################################################################


# Plot the mean training and validation curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(mean_training_accuracy, label='Mean Training Accuracy', color='blue')
plt.plot(mean_validation_accuracy, label='Mean Validation Accuracy', linestyle='--', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Mean Learning Curves - Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mean_training_loss, label='Mean Training Loss', color='blue')
plt.plot(mean_validation_loss, label='Mean Validation Loss', linestyle='--', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Learning Curves - Loss')
plt.legend()

plt.tight_layout()
plt.show()



# Plot training and validation curves for each fold with different colors
plt.figure(figsize=(12, 4))
# Define a list of colors for plotting
colors = ['blue', 'green', 'orange', 'purple', 'brown']
for fold in range(1, k_folds + 1):
    history_file_name = f'fold_{fold}_training_history.json'
    
    # Load the training history from a JSON file
    with open(history_file_name, 'r') as history_file:
        history = json.load(history_file)
    
    # Use a different color for each fold
    color = colors[fold - 1]

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label=f'Fold {fold} Training Accuracy', color=color)
    plt.plot(history['val_accuracy'], label=f'Fold {fold} Validation Accuracy', linestyle='--', color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Mean Learning Curves - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label=f'Fold {fold} Training Loss', color=color)
    plt.plot(history['val_loss'], label=f'Fold {fold} Validation Loss', linestyle='--', color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean Learning Curves - Loss')
    plt.legend()

# Save the figures for each fold
for fold in range(1, k_folds + 1):
    plt.figure(fold)
    plt.tight_layout()
    plt.savefig(f'fold_{fold}_training_plot.png')

# Show all the saved figures
plt.show()
y_pred = model.predict(X_test)
print(y_pred)
score = model.evaluate(X_test, y_test)
print("test loss:",score[0])
print("test accuracy:", score[1])

y_predictions = y_pred.copy()
y_predictions[y_predictions <= 0.5] = 0
y_predictions[y_predictions > 0.5] = 1

# Confusion matrix
cmn = confusion_matrix(y_test, y_predictions)
print("confusion matrisi", cmn)


import itertools
conf_matrix = confusion_matrix(y_test,y_predictions)

normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
plt.imshow(normalized_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
classes = ['Class 0', 'Class 1']  # Replace with your class labels
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = '.2f'
for i, j in itertools.product(range(normalized_conf_matrix.shape[0]), range(normalized_conf_matrix.shape[1])):
    plt.text(j, i, format(normalized_conf_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if normalized_conf_matrix[i, j] > 0.5 else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

plt.savefig('normalized_confusion_matrix.png')
plt.show()
from scipy import interp
from itertools import cycle


# Roc Curve
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
plt.title('Receiver operating characteristic example')
plt.legend(loc='best')
plt.show()


#Mertics
print(classification_report(y_test, y_predictions))
