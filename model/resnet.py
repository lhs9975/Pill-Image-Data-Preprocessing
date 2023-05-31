import os
import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

LossFunction = 'categorical_crossentropy'
img_width, img_height = 75, 75
NumberOfClass = 5

rootPath = 'E:\\data\\'

# def GetTopModel(model_input_shape):
#     model = Sequential()
#     model.add(Flatten(input_shape=model_input_shape))
#     model.add(Dense((NumberOfClass*2), activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(NumberOfClass, activation='softmax'))
#     model.compile(#optimizer='rmsprop',
#                   optimizer='adam',
#                   loss=LossFunction, metrics=['accuracy'])

#     return model


epochs = 100
batchSize = 32

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
    os.path.join(rootPath, 'train'),
    target_size=(img_width, img_height),
    batch_size=batchSize,
    subset='training'
)

validationGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'valid'),
    target_size=(img_width, img_height),
    batch_size=batchSize,
    subset='validation'
)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(ResNet50(include_top=True, weights=None, input_shape=None, classes=5))

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

from keras.models import load_model
from tensorflow.python.keras.models import load_model

model.save('E:\\model\\mnist_mlp_model_imprint.h5', save_format='h5')



import numpy as np
import matplotlib.pyplot as plt

acc= history.history['acc']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c="red", label='Trainset_acc')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()
