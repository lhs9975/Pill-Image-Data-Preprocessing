import os
import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.regularizers import l2


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

image_width = 75
image_height = 75

def build_model(l2_reg=0, n_classes=5):
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=(image_width, image_height, 3),
        padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    alexnet.compile(optimizer='rmsprop', # 학습 옵티마이저
                  loss='categorical_crossentropy', # 손실 함수
                  metrics=['accuracy'])    # 모델이 평가할 항목
    
    
    return alexnet


model = build_model()

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