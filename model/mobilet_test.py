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

rootPath = 'E:\\Original Data'
test_dir = os.path.join(rootPath, 'new')

img_width = 75
img_height = 75

batchSize = 64

test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(image_width, image_height),
#     batch_size=30,
#     class_mode='categorical')

testGen = test_datagen.flow_from_directory(
    os.path.join(rootPath, 'new'),
    target_size=(img_width, img_height),
    batch_size=batchSize,
    subset='test'
)

# 저장된 모델 파일 로드
model = models.load_model('E:\\model\\MobileNet_test1.h5')

# 테스트셋 평가
results = model.evaluate(testGen)
results