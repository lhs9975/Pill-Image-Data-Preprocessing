import os
import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
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
    os.path.join(rootPath, 'train\\imprint\\datasets'),
    target_size=(img_width, img_height),
    batch_size=batchSize,
    subset='training'
)

validationGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'valid\\imprint\\datasets'),
    target_size=(img_width, img_height),
    batch_size=batchSize,
    subset='validation'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss=LossFunction,
              optimizer='adam',
              metrics=['acc'])



from keras.models import load_model
from tensorflow.python.keras.models import load_model

model.save('E:\\model\\alexnet.h5', save_format='h5')


dir_name = 'E:\\test_log'

def make_Tensorboard_dir(dir_name):
    root_logdir = os.path.join(dir_name)
    return os.path.join(root_logdir)

lhs_log_dir = make_Tensorboard_dir(dir_name)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir = lhs_log_dir)

model.fit(trainGen,
          epochs=200,
          validation_data=validationGen,
          validation_freq=1,
          callbacks=[tensorboard_cb])