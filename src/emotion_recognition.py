# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras_preprocessing.image.image_data_generator import ImageDataGenerator
import os

# Initializing dataset directories
train_dir = 'data/train'
validate_dir = 'data/test'

batch_size = 64
num_epoch = 1

# Initializing image data generator and pre-processing images
train_data_generator = ImageDataGenerator(rescale=1./255)
validator_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
)

validator_generator = validator_data_generator.flow_from_directory(
    validate_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
)

# Building the convolutional neural network for the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# compiling and defining our optimizer and loss functions
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.0001, decay=1e-6),
    metrics=['accuracy']
)

# training our model
history = model.fit_generator(
    train_generator,
    epochs=num_epoch,
    validation_data=validator_generator,
)

df = pd.DataFrame(data=history.history)
loss = df['loss']
accuracy = df['accuracy']
epoch = history.epoch

# saving out model, (check how to get home path with os module)
model.save_weights(os.path.join(os.path.curdir + '/' + 'emotion_model.h5'))

# loading the saved model
saved_model = model.load_weights('emotion_model.h5')
model.summary()

# plotting the loss curves
style.use('seaborn')
plt.plot(epoch, loss, color='red', label='loss_curve')
plt.title('model loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

# plotting the accuracy curves
plt.plot(epoch, accuracy, color='green', label='accuracy_curve')
plt.title('model accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()


