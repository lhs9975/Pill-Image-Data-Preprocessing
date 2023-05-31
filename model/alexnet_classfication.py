# import tensorflow.compat.v1 as tf 

# tf.disable_v2_behavior()

import tensorflow as tf

from tensorflow import keras

from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# 데이터셋 디렉터리 경로
dataset_dir = 'E:\\data'
train_dir = os.path.join(dataset_dir, 'train\\imprint\\datasets')
validation_dir = os.path.join(dataset_dir, 'valid\\imprint\\datasets')

# 사진 크기
image_width = 100
image_height = 100

# ImageDataGenerator 초기화
# 0~1 사이로 숫자값 변경
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 훈련셋 제너레이터
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_width, image_height),
    batch_size=30,
    class_mode='categorical')

# 검증셋 제너레이터
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_width, image_height),
    batch_size=30,
    class_mode='categorical')

# 테스트셋 제너레이터
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(image_width, image_height),
#     batch_size=30,
#     class_mode='categorical')

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

# 모델 생성
model = build_model()

# 훈련 시작
history = model.fit(train_generator,
                    steps_per_epoch=60,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=20)



# 모델 저장
model.save('E:\\model\\alexnet.h5', save_format='h5')

dir_name = 'E:\\test_log'

def make_Tensorboard_dir(dir_name):
    root_logdir = os.path.join(dir_name)
    return os.path.join(root_logdir)

lhs_log_dir = make_Tensorboard_dir(dir_name)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir = lhs_log_dir)