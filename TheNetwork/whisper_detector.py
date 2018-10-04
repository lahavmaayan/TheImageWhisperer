
from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras import regularizers
import os

from TheNetwork.veggie import VeGGieModel

class WhisperDetector(object):
    """"""
    def __init__(self, path):
        self.model = None
        self.data_path = path

    def build(self):
        """Build the VeGGie architecture."""
        veggie_model = VeGGieModel()
        self.model = veggie_model.build_veggie_model()

    def json_file_to_array(self, filename):
        # filename doesn't end with .json
        with open(filename) as f:
            a = json.loads(f.read())
            a = np.array([[[pix for pix in row] for row in color] for color in a])
            a = a.transpose(1, 2, 0)
            return a

    def train(self):
        """
        Train the model with new data.

        This is where the Transfer Learning is happening - the VGG part of the network is already trained,
        and now we are exposing the model to a new data set -
        of CIFAR10 images that a random half of them were manipulated using various steganography algorithms.
        """
        # training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        def lr_scheduler(epoch):
             return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        train_datagen = ImageDataGenerator(preprocessing_function=self.json_file_to_array)
        train_generator = train_datagen.flow_from_directory(
            directory=os.path.join(self.data_path, "train"),
            target_size=(32, 32),
            color_mode="rgb",
            batch_size=32,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        validate_datagen = ImageDataGenerator(preprocessing_function=self.json_file_to_array)
        validate_generator = validate_datagen.flow_from_directory(
            directory=os.path.join(self.data_path, "validate"),
            target_size=(32, 32),
            color_mode="rgb",
            batch_size=32,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        test_datagen = ImageDataGenerator(preprocessing_function=self.json_file_to_array)
        test_generator = test_datagen.flow_from_directory(
            directory=os.path.join(self.data_path, "test"),
            target_size=(32, 32),
            color_mode="rgb",
            batch_size=1,
            class_mode=None,
            shuffle=False,
            seed=42
        )

        #data augmentation - only flip as we don't want to harm the stegged data
        datagen = ImageDataGenerator(
#            featurewise_center=False,  # set input mean to 0 over the dataset
#            samplewise_center=False,  # set each sample mean to 0
#            featurewise_std_normalization=False,  # divide inputs by std of the dataset
#            samplewise_std_normalization=False,  # divide each input by its std
#            zca_whitening=False,  # apply ZCA whitening
#            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
#            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
#        (std, mean, and principal components if ZCA whitening is applied).
#         datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        # historyvtemp = self.model.fit_generator(datagen.flow(x_train, y_train,
        #                                                     batch_size=batch_size),
        #                                        steps_per_epoch=x_train.shape[0] // batch_size,
        #                                        epochs=maxepoches,
        #                                        validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=validate_generator,
            validation_steps=800 // batch_size)
        self.model.save_weights('veggie.h5')

    def predict(self):
        pass

    def save_trained_model(self, h5_filename='veggie.h5'):
        self.model.save_weights(h5_filename)
