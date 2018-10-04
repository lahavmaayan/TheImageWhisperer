import json
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageUtils(object):
    IMAGES_DIR = 'C:/Users/Rey/Projects/TheImageWhisperer/Data'

    def __init__(self):
        self.images_dir = ImageUtils.IMAGES_DIR

    def json_filename_to_array(self, json_filename):
        """Load .json filename into a numpy array that fits as input to VeGGie."""
        a = json.load(open(json_filename))
        a = np.array([[[pix for pix in row] for row in color] for color in a])
        a = a.transpose(1, 2, 0)
        return a

    def show_img_array_in_notebook(self, a):
        """When in jupyter notebook, show img inline. requires %matplotlib inline"""
        cv2.imwrite('color_img.jpg', a)
        img = cv2.imread("color_img.jpg", 3)
        plt.imshow(img, cmap='gray')

    # def load_images(self):
    #     stagged_dir = '{}/stegged'.format(self.images_dir)
    #     not_stagged_dir = '{}/not_stegged'.format(self.images_dir)

