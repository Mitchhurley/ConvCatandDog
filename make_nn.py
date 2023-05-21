import sys

import keras.layers
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import shutil
import tempfile
import os
import numpy as np

#take the directory as a param and creates a temp directory
#that can be used to generate the dataset for training and validation
if(os.path.exists(sys.argv[1])):
    #creating my tempdir to sort the photos into
    with tempfile.TemporaryDirectory() as temp_dir:
        #take the first arg as the directory name
        tempDirectory = sys.argv[1]
        cat_dir = os.path.join(temp_dir, 'cat')
        os.makedirs(cat_dir)
        dog_dir = os.path.join(temp_dir, 'dog')
        os.makedirs(dog_dir)

        # copy images to subdirectories
        for file in os.listdir(tempDirectory):
            if file.startswith('c'):
                shutil.copy(os.path.join(tempDirectory, file), os.path.join(cat_dir, file))
            elif file.startswith('d'):
                shutil.copy(os.path.join(tempDirectory, file), os.path.join(dog_dir, file))

        # construct the train dataset
        train = tf.keras.preprocessing.image_dataset_from_directory(
            temp_dir,
            validation_split=0.2,
            subset='training',
            seed=1109,
            image_size=(100, 100),
            batch_size=32)

        val = tf.keras.preprocessing.image_dataset_from_directory(
            temp_dir,
            validation_split=0.2,
            subset='validation',
            seed=1109,
            image_size=(100, 100),
            batch_size=8)


        data_aug = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ])


        model = models.Sequential([
            data_aug,
            layers.experimental.preprocessing.Rescaling(1./255), #does what the flatten method would do
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='tanh'),
            layers.Dense(1, activation='sigmoid')
        ])
        op = tf.keras.optimizers.Adam(
            learning_rate=0.0001, #tried other vals and this seems best
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            weight_decay=None)

        model.compile(optimizer=op,
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

        #do the actual training
        model.fit(train, validation_data=val, epochs=25)

        #save the model
    modelname = sys.argv[2] + ".h5"
    model.save(modelname, include_optimizer = False)
else:
    print(f'Error: Unable to find folder {sys.argv[1]}')
#Make sure our temp directory has been closed