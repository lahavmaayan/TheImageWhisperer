#!/usr/bin/env python
# coding: utf-8

import mxnet as mx
import numpy as np
import cPickle
import cv2
import random
import json
import os

text_dict = create_encryption_text()

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = cPickle.load(f)
    return dict['label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)

def saveCifarJPGImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to JPG file
    return cv2.imwrite(path+file+'.jpg', array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def saveCifarBMPImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to JPG file
    return cv2.imwrite(path+file+'.bmp', array)

def get_images_array(path, filename):
    imgarray, lblarray = extractImagesAndLabels(path, filename)
    return imgarray

def cifarImageToArray(path, file):
    srcBGR = cv2.imread(path + file)
    dest = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    array = dest.transpose(2, 0, 1)
    mx_ex_int_array = mx.nd.array(array)
    return mx_ex_int_array

def get_all_images(path, file_pattern):
    images = []
    for i in range(5):
        images.append(get_images_array(path, file_pattern + str(i+1)))
    return images

def create_encryption_text():
    text = open("./PRIDE_AND_PREJUDICE.txt", "r").read()    
    start_index = 0
    jump = 50
    read = 50
    read_dict = {}
    for i in xrange(5000):
        short_text = text[start_index:start_index+read]
        start_index = start_index + jump
        filename = "./short_text_" + str(i) + ".txt"
        file(filename, "w").write(short_text)
        read_dict[i] = filename

def encrypt_file(text_filename):
    passphrase = get_random_passphrase()
    print "Goint to encrypt text file " + text_filename
    encrypt_command = "steghide embed -cf ./big_batch/stegged_folder/encrypt_image.bmp -ef " + text_filename + " -p " + passphrase
    print encrypt_command
    os.system(encrypt_command)

def get_random_passphrase():
    return str(random.randint(1000, 9999))

def create_split_batches():
    path = "./cifar-10-batches-py/"
    images = get_all_images(path, "data_batch_")
    encryption_counter = 0
    #for batch in images:
    image = images[4]
    for image_index, image in enumerate(image):
        print image_index
        should_encrypt = random.randint(0,1)
        if should_encrypt:
            print "Going to encrypt..."
            saveCifarBMPImage(image, "./big_batch/stegged_folder/", "encrypt_image")
            encrypt_file("./short_text_" + str(encryption_counter % 5000) + ".txt")
            encrypted_array = cifarImageToArray("./big_batch/stegged_folder/", "encrypt_image.bmp")
            filename = "image" + str(image_index)
            encryption_counter += 1
            with open("./big_batch/stegged_folder/" + filename + ".json", "w") as f:
                json.dump(encrypted_array.asnumpy().tolist(), f)
            print "saved in stegged as " + filename
        else:
            print "Not going to encrypt...",
            filename = "image" + str(image_index)
            print type(image)
            with open("./big_batch/not_stegged_folder/" + filename + ".json", "w") as f:
                json.dump(image.asnumpy().tolist(), f)
            print "saved in not_stegged as " + filename

