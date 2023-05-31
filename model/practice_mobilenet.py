import os
import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

LossFunction = 'categorical_crossentropy'
img_width, img_height = 75, 75

rootPath = 'E:\\data\\'

epochs = 5
batchSize = 64

train_datagen = ImageDataGenerator(
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

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

trainGen = train_datagen.flow_from_directory(
    os.path.join(rootPath, 'train'),
    target_size=(img_width, img_height),
    batch_size=batchSize,
    subset='training'
)

validationGen = validation_datagen.flow_from_directory(
    os.path.join(rootPath, 'valid'),
    target_size=(img_width, img_height),
    batch_size=batchSize,
    subset='validation'
)

testGen = test_datagen.flow_from_directory(
    os.path.join(rootPath, 'test'),
    target_size=(img_width, img_height),
    batch_size=batchSize
)

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential


model = Sequential()
model.add(MobileNet(include_top=True, weights=None, input_shape=(224, 224, 3), classes=27))

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

model.save('E:\\model\\MobileNet_test2.h5', save_format='h5')

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



# 저장된 모델 파일 로드
model = load_model('E:\\model\\MobileNet_test2.h5')

# 테스트셋 평가
results = model.evaluate(testGen)
results