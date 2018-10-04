
import cPickle
import numpy as np
import cv2
import mxnet as mx
import json


def main_flow_games():
    # path = "/Users/maayanl/Downloads/cifar-10-batches-py/data_batch_1"
    # image_list = get_images_array(path)
    # test_image = image_list[1]
    # saveCifarBMPImage(test_image, "../images/not_encrypted/test_image")
    with open("../image2400", "r") as f:
        data = json.load(f)

    saveCifarBMPImage(data, "../images/not_encrypted/test_image")
    # return test_image
    # with open("../images/not_encrypted/test_image", "w") as f:
    #     json.dump(test_image.asnumpy().tolist(), f)
    # print "3"
    # d = unpickle(path)
    # print d


def get_images_array(path):
    imgarray, lblarray = extractImagesAndLabels(path)
    return imgarray


def saveCifarBMPImage(array, path):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to JPG file
    return cv2.imwrite(path+'.bmp', array)


def extractImagesAndLabels(path):
    f = open(path, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray


def unpickle(file):
    with open(file, 'rb') as fo:
        d = cPickle.load(fo)
    return d


if __name__ == "__main__":
    main_flow_games()