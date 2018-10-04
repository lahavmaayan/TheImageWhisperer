"""
flow:
    - get an image
    - check if it was encrypted
        - use either Asia's or Dalya's code.
    - return a number between 0 to 1, matching the probability
"""
import random
import os
from PIL import Image
import json
import cv2
import mxnet as mx
import numpy as np


PATH_TO_IMAGE_DICT = "../images"
ENCRYPTED_IMAGES_DICT_NAME = "encrypted/"
REGULAR_IMAGES_DICT_NAME = "not_encrypted/"

SHOULD_RUN_ASYAS_CODE = True

BASE_TEXT = "Based on our *data science magic*, we conclude the image was {!s}!"
WAS_ENCRYPTED = 1
WAS_NOT_ENCRYPTED = 0

RESULT_TO_TEXT_DICT = {
    WAS_ENCRYPTED: "encrypted",
    WAS_NOT_ENCRYPTED: "was not encrypted"
}


def main_flow():
    chosen_image_filename, was_image_encrypted = _get_random_image_file()
    # image_represented_as_lists = _convert_image_file_to_lists(chosen_image_filename)
    result = _check_if_image_was_encrypted(chosen_image_filename)
    bmp_path = _transform_array_to_bmp_file(chosen_image_filename)
    _present_result(result, bmp_path, was_image_encrypted)


def _get_random_image_file():
    # @TODO - for testing reasons, only choose no encrypted.
    should_take_encrypted_image = random.randint(0, 1)
    chosen_dir_name = ENCRYPTED_IMAGES_DICT_NAME if should_take_encrypted_image else REGULAR_IMAGES_DICT_NAME
    dir_of_current_file = os.path.dirname(os.path.abspath(__file__))
    path_to_image_dir = os.path.join(dir_of_current_file, PATH_TO_IMAGE_DICT, chosen_dir_name)
    all_images_in_chosen_dir = os.listdir(path_to_image_dir)
    image_to_check = random.choice(all_images_in_chosen_dir)
    print "this is the image we chose - {!s}".format(image_to_check)
    return os.path.join(path_to_image_dir, image_to_check), should_take_encrypted_image


def _convert_image_file_to_matrix(chosen_image_filename):
    # return main_flow_games()
    with open(chosen_image_filename, "w"):
        # @TODO - read the image nested lists and convert to numpy
        pass
    return [[], [], []]


def _check_if_image_was_encrypted(image_represented_as_lists):
    if SHOULD_RUN_ASYAS_CODE:
        return _asyas_code(image_represented_as_lists)
    return _dalyas_code(image_represented_as_lists)


def _dalyas_code(image):
    return 0


def _asyas_code(image):
    return 1


def _json_filename_to_array(json_filename):
    a = json.load(open(json_filename))
    a = np.array([[[pix for pix in row] for row in color] for color in a])
    a = a.transpose(1, 2, 0)
    return a


def _cifar_image_to_array(path):
    srcBGR = cv2.imread(path)
    dest = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    array = dest.transpose(2, 0, 1)
    mx_ex_int_array = mx.nd.array(array)
    return mx_ex_int_array


def _transform_array_to_bmp_file(array_path):
    array = _json_filename_to_array(array_path)

    file_name = os.path.splitext(array_path)[0]
    bmp_path = file_name + ".bmp"
    print bmp_path

    cv2.imwrite(bmp_path, array)
    return bmp_path


def _present_result(result, image_path, was_image_encrypted):
    print image_path
    img = Image.open(image_path)
    img.show()
    print "The image was {!s}".format(RESULT_TO_TEXT_DICT[was_image_encrypted])
    result_text = BASE_TEXT.format(RESULT_TO_TEXT_DICT[result])
    print result_text


# if __name__ == "__main__":
    # dir_of_current_file = os.path.dirname(os.path.abspath(__file__))
    # _present_result(1, os.path.join(dir_of_current_file, "../images/encrypted/encrypt_image2_2.bmp"), 1)
main_flow()
