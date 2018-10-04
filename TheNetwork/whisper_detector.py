
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
import json

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

    # def json_file_to_array(self, filename):
    #     # filename doesn't end with .json
    #     with open(filename) as f:
    #         a = json.loads(f.read())
    #         a = np.array([[[pix for pix in row] for row in color] for color in a])
    #         a = a.transpose(1, 2, 0)
    #         return a

    def json_filename_to_array(self, json_filename):
       a = json.load(open(json_filename))
       a = np.array([[[pix for pix in row] for row in color] for color in a])
       a = a.transpose(1, 2, 0)
       return a

    def folder_to_array(self, folder_path):
        array_list = []
        for i, filename in enumerate(os.listdir(folder_path)):
            arr = self.json_filename_to_array(folder_path + "/" + filename)
            array_list.append(arr)
        res = np.asarray(array_list)
        return res

    def load_data(self):
        num_classes = 1

        # The data, shuffled and split between train and test sets:
        train_path_stegged = r"C:\Users\Asya\Code\dataHack\data\train\stegged"
        train_path_not_stegged = r"C:\Users\Asya\Code\dataHack\data\train\not_stegged"
        test_path_stegged = r"C:\Users\Asya\Code\dataHack\data\test\stegged"
        test_path_not_stegged = r"C:\Users\Asya\Code\dataHack\data\test\not_stegged"

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

        # y_train = keras.utils.to_categorical(y_train, num_classes)
        #
        # y_test = keras.utils.to_categorical(y_test, num_classes)
        return (x_train, y_train), (x_test, y_test)

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
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
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        (x_train, y_train), (x_test, y_test) = self.load_data()

        def lr_scheduler(epoch):
             return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # train_datagen = ImageDataGenerator(preprocessing_function=self.json_file_to_array)
        # train_generator = train_datagen.flow_from_directory(
        #     directory=os.path.join(self.data_path, "train"),
        #     target_size=(32, 32),
        #     color_mode="rgb",
        #     batch_size=32,
        #     class_mode="categorical",
        #     shuffle=True,
        #     seed=42
        # )
        #
        # validate_datagen = ImageDataGenerator(preprocessing_function=self.json_file_to_array)
        # validate_generator = validate_datagen.flow_from_directory(
        #     directory=os.path.join(self.data_path, "validate"),
        #     target_size=(32, 32),
        #     color_mode="rgb",
        #     batch_size=32,
        #     class_mode="categorical",
        #     shuffle=True,
        #     seed=42
        # )
        #
        # test_datagen = ImageDataGenerator(preprocessing_function=self.json_file_to_array)
        # test_generator = test_datagen.flow_from_directory(
        #     directory=os.path.join(self.data_path, "test"),
        #     target_size=(32, 32),
        #     color_mode="rgb",
        #     batch_size=1,
        #     class_mode=None,
        #     shuffle=False,
        #     seed=42
        # )

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
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        # self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        historyvtemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                                                             batch_size=batch_size),
                                               steps_per_epoch=x_train.shape[0] // batch_size,
                                               epochs=maxepoches,
                                               validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=validate_generator,
            validation_steps=800 // batch_size)
        self.model.save_weights('veggie.h5')

    def predict(self, json_file):
        arr = self.json_filename_to_array(json_file)
        self.model.predict(arr)

    def save_trained_model(self, h5_filename='veggie.h5'):
        self.model.save_weights(h5_filename)
