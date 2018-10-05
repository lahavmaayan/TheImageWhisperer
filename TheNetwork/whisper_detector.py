
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import os
import json

from TheNetwork.veggie import VeGGieModel

class WhisperDetector(object):
    """"""
    def __init__(self, max_num_pics_per_category=None, epochs=250, batch_size=24):
        self.max_num_pics_per_category = max_num_pics_per_category or float('inf')
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build(self):
        """Build the VeGGie architecture."""
        veggie_model = VeGGieModel()
        self.model = veggie_model.build_veggie_model()

    def load_weights(self, h5_filename):
        """For fail-safe reasons, sometimes we train in separate epochs, and save weights between epochs."""
        print("Loading weights of previously trained model.")
        self.model.load_weights(h5_filename)

    def json_filename_to_array(self, json_filename):
        """Load .json filename into a numpy array that fits into VeGGie network."""
        a = json.load(open(json_filename))
        a = np.array([[[pix for pix in row] for row in color] for color in a])
        a = a.transpose(1, 2, 0)
        return a

    def folder_to_array(self, folder_path):
        """Load all images from a folder and put in a numpy array of one batch."""
        array_list = []
        for i, filename in enumerate(os.listdir(folder_path)):
            arr = self.json_filename_to_array(folder_path + "/" + filename)
            array_list.append(arr)
            if i > self.max_num_pics_per_category:
                break
        res = np.asarray(array_list)
        return res

    def unison_shuffled_copies(self, a, b):
        """Shuffle order of input photos in the batch."""
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def load_data(self):
        """
        Loads the data, split between train and test sets and shuffles it
        """
        train_path_stegged = 'C:/Users/Rey/Projects/TheImageWhisperer/Data/train/stegged'
        train_path_not_stegged = 'C:/Users/Rey/Projects/TheImageWhisperer/Data/train/not_stegged'
        test_path_stegged = 'C:/Users/Rey/Projects/TheImageWhisperer/Data/validate/stegged'
        test_path_not_stegged = 'C:/Users/Rey/Projects/TheImageWhisperer/Data/validate/not_stegged'

        x_train_stegged = self.folder_to_array(train_path_stegged)
        x_train_not_stegged = self.folder_to_array(train_path_not_stegged)
        x_test_stegged = self.folder_to_array(test_path_stegged)
        x_test_not_stegged = self.folder_to_array(test_path_not_stegged)
        x_train = np.concatenate((x_train_stegged, x_test_not_stegged), axis=0)
        x_test = np.concatenate((x_test_stegged, x_test_not_stegged), axis=0)
        y_train = np.zeros(len(x_train_stegged) + len(x_train_not_stegged))
        y_test = np.zeros(len(x_test_stegged) + len(x_test_not_stegged))
        y_train[:len(x_train_stegged)] = 1
        y_test[:len(x_test_stegged)] = 1

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        x_train, y_train = self.unison_shuffled_copies(x_train, y_train)
        x_test, y_test = self.unison_shuffled_copies(x_test, y_test)

        return (x_train, y_train), (x_test, y_test)

    def normalize(self, X_train, X_test):
        """
        this function normalize inputs for zero mean and unit variance
        it is used when training a model.
        Input: training set and test set
        Output: normalized training set and test set according to the trianing set statistics.
        """
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)

        return X_train, X_test

    def train(self):
        """
        Train the model with new data.

        This is where the Transfer Learning is happening - the VGG part of the network is already trained,
        and now we are exposing the model to a new data set -
        of CIFAR10 images that a random half of them were manipulated using various steganography algorithms.
        """
        # training parameters
        batch_size = self.batch_size
        maxepoches = self.epochs
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        (x_train, y_train), (x_test, y_test) = self.load_data()

        # data augmentation - only flip as we don't want to harm the stegged data
        datagen = ImageDataGenerator(
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.
        reduce_lr = self.reduce_lr(learning_rate, lr_drop)
        self.model.fit_generator(datagen.flow(x_train, y_train,
                                              batch_size=batch_size),
                                 steps_per_epoch=x_train.shape[0] // batch_size,
                                 epochs=maxepoches,
                                 validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)

        # self.model.fit_generator(datagen.flow(x_train, y_train,
        #                                       batch_size=batch_size),
        #                          steps_per_epoch=x_train.shape[0] // batch_size,
        #                          epochs=maxepoches,
        #                          validation_data=(x_test, y_test), verbose=2)


        self.save_trained_model('veggie.h5')

    def reduce_lr(self, learning_rate, lr_drop):
        """Keras callback to reduce learning rate as the learning progresses."""
        return keras.callbacks.LearningRateScheduler(
            lambda epoch: learning_rate * (0.5 ** (epoch // lr_drop)))

    def predict(self, json_file):
        arr = self.json_filename_to_array(json_file)
        arr = np.array(arr)
        return self.model.predict(arr)

    def save_trained_model(self, h5_filename='veggie.h5'):
        self.model.save_weights(h5_filename)
