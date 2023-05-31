import os
import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model

# 위노그라드 알고리즘 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

LossFunction = 'categorical_crossentropy'

NumberOfClass = 2

rootPath = 'E:\\datasets\\'
SaveModelPath = 'E:\\train code'

SaveModelPathForEarlyStopping = SaveModelPath +  ".hdf5"

def GetTopModel(model_input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=model_input_shape))
    model.add(Dense((NumberOfClass*2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NumberOfClass, activation='softmax'))
    model.compile(#optimizer='rmsprop',
                  optimizer='adam',
                  loss=LossFunction, metrics=['accuracy'])

    return model


epochs = 30
batchSize = 64

imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[.95, 1.30],
    horizontal_flip=False,
    vertical_flip=False,
    shear_range=0.1,
    zoom_range=0.1,
)

trainGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'training_set'),
    target_size=(64, 64),
    batch_size=batchSize,
    subset='training'
)

validationGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'validation_set'),
    target_size=(64, 64),
    batch_size=batchSize,
    subset='validation'
)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(ResNet50(include_top=True, weights=None, input_shape=None, classes=2))

model.compile(loss=LossFunction,
              optimizer='adam',
              metrics=['acc'])


history = model.fit_generator(
    trainGen, 
    epochs=epochs,
    steps_per_epoch=trainGen.samples / batchSize, 
    validation_data=validationGen,
    validation_steps=trainGen.samples / batchSize
)

import numpy as np
import matplotlib.pyplot as plt

y_vloss=history.history['loss']
y_acc=history.history['acc']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

plt.show()