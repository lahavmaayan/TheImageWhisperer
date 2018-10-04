
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

from TheNetwork.veggie import VeGGieModel


class WhisperDetector(object):
    """"""
    def __init__(self):
        self.model = None

    def build(self):
        """Build the VeGGie architecture."""
        veggie_model = VeGGieModel()
        self.model = veggie_model.build_veggie_model()

    def train(self, x_train, y_train, x_test, y_test):
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

        #
        # def lr_scheduler(epoch):
        #     return learning_rate * (0.5 ** (epoch // lr_drop))
        # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        #
        # # data augmentation
        # datagen = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False)  # randomly flip images
        # # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(x_train)
        #
        # # optimization details
        # sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        # self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #
        # # training process in a for loop with learning rate drop every 25 epoches.
        #
        # historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
        #                                                     batch_size=batch_size),
        #                                        steps_per_epoch=x_train.shape[0] // batch_size,
        #                                        epochs=maxepoches,
        #                                        validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)
        # self.model.save_weights('veggie.h5')

    def predict(self):
        pass

    def save_trained_model(self, h5_filename='veggie.h5'):
        self.model.save_weights(h5_filename)
