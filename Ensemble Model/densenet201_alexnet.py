import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_alexnet(input_shape=(224, 224, 1), num_classes=1):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 2
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 3
    model.add(Conv2D(384, (3, 3), activation='relu'))

   # Layer 4
    model.add(Conv2D(384, (3, 3), activation='relu'))


    # Layer 5
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flatten the layers
    model.add(Flatten())

    # Dense layers
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))  # Output layer for binary classification

    return model

# Create an instance of AlexNet
alexnet_model = create_alexnet()


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, GlobalAveragePooling2D

def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    bn_axis = 3  # Assuming channels_last data format
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    bn_axis = 3  # Assuming channels_last data format
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def create_densenet201(input_shape=(224, 224,1), num_classes=1):
    img_input = Input(shape=input_shape)

    bn_axis = 3  # Assuming channels_last data format

    x = Conv2D(64, 7, strides=2, use_bias=False, padding='same', name='conv1/conv')(img_input)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

    x = dense_block(x, 6, name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, 12, name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, 48, name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, 32, name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='sigmoid', name='fc1000')(x)

    model = tf.keras.models.Model(img_input, x, name='densenet201')
    
    return model

# Create an instance of DenseNet201
densenet201_model = create_densenet201()

# Create input layer
input_layer = Input(shape=(224, 224,1))

# Get output tensors from both models
alexnet_output = alexnet_model(input_layer)
densenet_output = densenet201_model(input_layer)
densenet_output = Flatten()(densenet_output)
# Concatenate the outputs
concatenated = concatenate([alexnet_output, densenet_output], axis=-1)

# Add some additional dense layers for combining features
x = Dense(256, activation='relu')(concatenated)
x = Dense(128, activation='relu')(x)

# Output layer for classification (adjust units based on your task)
output_layer = Dense(units=1, activation='sigmoid')(x)

# Create the ensemble model
ensemble_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the ensemble model
ensemble_model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Display the summary of the model
ensemble_model.summary()

ensemble_model.save('Dense_Alex_Ensemble4.h5')

