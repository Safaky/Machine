import json
import matplotlib.pyplot as plt

# Load history from the JSON file, if you don't have json files you can implement plot section directly to your code.
with open('/home/syasun/Codes/fold_1_training_history.json', 'r') as file:
    history = json.load(file)

# Assuming the keys are 'loss' and 'val_loss', these names have adjusted by you.
loss = history['loss']
val_loss = history['val_loss']

# Plot training & validation loss values, normal one.
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Set the range which you want.

start_epoch = 30
end_epoch = 60
min_loss = 20
max_loss = 40

# Plot training & validation loss values, scaled one.
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim(min_loss, max_loss)
plt.xlim(start_epoch, end_epoch)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

