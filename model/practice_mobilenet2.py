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


epochs = 5
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

rootPath = 'E:\\data\\'

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

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential


base_model = tf.keras.applications.MobileNetV2(input_shape=(75, 75),
                                               include_top=False,
                                               weights='imagenet')



model.compile(loss=LossFunction,
              optimizer='adam',
              metrics=['acc'])


nb_train_samples = 2112
nb_validation_samples = 405

history = model.fit_generator(
    trainGen,
    steps_per_epoch= nb_train_samples // batchSize,
    epochs=epochs,
    validation_data=validationGen,
    validation_steps = nb_validation_samples// batchSize,
    #use_multiprocessing=,
    #workers=2
)

# history = model.fit_generator(
#     trainGen, 
#     epochs=epochs,
#     steps_per_epoch=trainGen.samples / batchSize, 
#     validation_data=validationGen,
#     validation_steps=trainGen.samples / batchSize
# )

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