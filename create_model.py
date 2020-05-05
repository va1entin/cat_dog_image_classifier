#!/usr/bin/env python3

import cv2
import numpy as np

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import os
import random

# folders to look into relative to current directory
train_dir = "ml/input/train"
test_dir = "ml/input/test"


# get image file names
train_dog_images = [f'{train_dir}/{i}' for i in os.listdir(train_dir) if 'dog' in i]
train_cat_images = [f'{train_dir}/{i}' for i in os.listdir(train_dir) if 'cat' in i]
test_images = [f'{test_dir}/{i}' for i in os.listdir(test_dir)]

# slice and shuffle train_images
train_images = train_dog_images[:2000] + train_cat_images[:2000]
random.shuffle(train_images)

# define how the input images shall be resized
px_rows = 150
px_columns = 150

# fill X and y with images and tags
X = []
y = []

for image_file_name in train_images:
    # read from file
    image = cv2.resize(cv2.imread(image_file_name, cv2.IMREAD_COLOR), (px_rows, px_columns), interpolation=cv2.INTER_CUBIC)
    X.append(image)
    if 'cat' in image_file_name:
        y.append(0)
    elif 'dog' in image_file_name:
        y.append(1)

# convert to numpy arrays
X = np.array(X)
y = np.array(y)

# split into train and test, test being 20% of the dataset
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=2)

# get lengths
len_train = len(X_train)
len_validation = len(X_validation)

batch_size = 32

# create model
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

# compile model
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4))

# create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = validation_datagen.flow(X_validation, y_validation, batch_size=batch_size)

# train model
model.fit_generator(train_generator, steps_per_epoch=len_train // batch_size, epochs=64, validation_data=validation_generator, validation_steps=len_validation // batch_size)

# save model
model.save_weights('ml/model_weights.h5')
model.save('ml/model_keras.h5')
