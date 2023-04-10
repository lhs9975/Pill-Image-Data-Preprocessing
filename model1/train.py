import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
import zipfile

import gdown

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_path = 'D:\\pill\\image\\EXK\\EXK8(rembg_his)'

# 예측할 클래스 수
classes = 3

# Input으로 사용될 크기와 채널수
height = 1080
width = 720
channels = 3

resnetv2 = tf.keras.applications.ResNet50V2(include_top=False,input_shape=(height,width,channels))

resnetv2.trainable=False

model = Sequential([
                 resnetv2,
                 Dense(512,activation='relu'),
                 BatchNormalization(),
                 GlobalAveragePooling2D(),
                 Dense(classes,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(image_data_train,batch_size=32,epochs=5,callbacks=[lrs],validation_data=(image_data_test),
          validation_steps =image_data_test.samples/image_data_test.batch_size)