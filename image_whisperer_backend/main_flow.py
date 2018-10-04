"""
flow:
    - get an image
    - check if it was encrypted
        - use either Asia's or Dalya's code.
    - return a number between 0 to 1, matching the probability
"""
import os
import json
import cv2
import mxnet as mx
import numpy as np
import sys
import logging
logging.basicConfig(filename='log.log', level=logging.INFO)


PATH_TO_IMAGE_DICT = "../images"
ENCRYPTED_IMAGES_DICT_NAME = "encrypted/"
REGULAR_IMAGES_DICT_NAME = "not_encrypted/"

BASE_TEXT = "Based on our *data science magic*, we conclude the image was {!s}!"
WAS_ENCRYPTED = 1
WAS_NOT_ENCRYPTED = 0

RESULT_TO_TEXT_DICT = {
    WAS_ENCRYPTED: "encrypted",
    WAS_NOT_ENCRYPTED: "was not encrypted"
}


def main_flow(image_file_name):
    logging.info("main flow, and this is the image file name - {!s}".format(image_file_name))
    path_to_image_file = _get_path_to_image_as_json(image_file_name)
    result = _check_if_image_was_encrypted(path_to_image_file)
    bmp_path = _transform_array_to_bmp_file(path_to_image_file)
    _present_result(result, bmp_path)


def _get_path_to_image_as_json(image_file_name):
    logging.info("_get_path_to_image_as_json, and this is the image file name - {!s}".format(image_file_name))
    dir_of_current_file = os.path.dirname(os.path.abspath(__file__))
    chosen_image_path = os.path.join(dir_of_current_file, "..", "image_whisperer_frontend", "uploads", image_file_name)
    image_represented_as_lists = _cifar_image_to_array(chosen_image_path)
    json_path = os.path.join(dir_of_current_file, "..", "images", image_file_name.split(".")[0] + ".json")
    with open(json_path, "w") as f:
        json.dump(image_represented_as_lists.asnumpy().tolist(), f)
    logging.info("this is the json path in _get_path_to_image_as_json - {!s}".format(json_path))
    return json_path


def _check_if_image_was_encrypted(image_represented_as_lists):
    logging.info("_check_if_image_was_encrypted, and this is the image_represented_as_lists - {!s}".format(
        image_represented_as_lists))
    return _asyas_code(image_represented_as_lists)


def _asyas_code(image):
    # @TODO - should call the relevant code
    return 1


def _json_filename_to_array(json_filename):
    a = json.load(open(json_filename))
    a = np.array([[[pix for pix in row] for row in color] for color in a])
    a = a.transpose(1, 2, 0)
    return a


def _cifar_image_to_array(path):
    logging.info("run this function - _cifar_image_to_array, and this is the path - {!s}".format(path))
    srcBGR = cv2.imread(path)
    dest = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    array = dest.transpose(2, 0, 1)
    mx_ex_int_array = mx.nd.array(array)
    return mx_ex_int_array


def _transform_array_to_bmp_file(array_path):
    logging.info("_transform_array_to_bmp_file, and this is the array path - {!s}".format(array_path))
    array = _json_filename_to_array(array_path)

    file_name = os.path.splitext(array_path)[0]
    bmp_path = file_name + ".bmp"

    cv2.imwrite(bmp_path, array)
    logging.info("this is the bmp_path - {!s}".format(bmp_path))
    return bmp_path


def _present_result(result, image_path):
    logging.info("_present result, this is the image_path - {!s}".format(image_path))
    # the STDOUT is reported to the front-end.
    print 1


if __name__ == "__main__":
    logging.info("we were called!")
    path = sys.argv[1]
    main_flow(path)
    logging.info("done!")
