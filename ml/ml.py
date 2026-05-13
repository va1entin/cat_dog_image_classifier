import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import random
import gc

train_dir = "input/train"
test_dir = "input/test"

# get dog images
train_dogs = [f'input/train/{i}' for i in os.listdir(train_dir) if 'dog' in i]

#get cat images
train_cats = [f'input/train/{i}' for i in os.listdir(train_dir) if 'cat' in i]

#get test images
test_images = [f'input/test/{i}' for i in os.listdir(test_dir)]

# slice dataset, use 2000 in each class
train_images = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_images)

del train_dogs
del train_cats
gc.collect()

# view images

import matplotlib.image as mpimg
for ima in train_images[0:3]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()



nrows = 150
ncolumns = 150
channels = 3 #1 for grayscale


def read_and_process_image(list_of_images):
    X = []
    y = []
    for image in list_of_images:
        # Read image
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
    return X, y

X, y = read_and_process_image(train_images)

X = np.array(X)
y = np.array(y)

print('Shape of train images: ', X.shape)
print('Shape of labels:', y.shape)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print('Shape of train images is:', X_train.shape)
print('Shape of validation images is:', X_val.shape)
print('Shape of train labels is:', y_train.shape)
print('Shape of validation labels is:', y_val.shape)

del X
del y
gc.collect()

ntrain = len(X_train)
nval = len(X_val)

batch_size = 32

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

history = model.fit_generator(train_generator, \
steps_per_epoch=ntrain // batch_size, \
epochs=64, \
validation_data = val_generator, \
validation_steps = nval // batch_size)

model.save_weights('model_weights.h5')
model.save('model_keras.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
