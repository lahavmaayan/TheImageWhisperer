import json
import numpy as np

class DataUtils(object):
    IMAGES_DIR = 'C:/Users/Rey/Projects/TheImageWhisperer/Data'

    def __init__(self):
        self.images_dir = DataUtils.IMAGES_DIR

    def json_filename_to_array(self, json_filename):
        a = json.load(open(json_filename))
        a = np.array([[[pix for pix in row] for row in color] for color in a])
        a = a.transpose(1, 2, 0)
        return a

    # def load_images(self):
    #     stagged_dir = '{}/stegged'.format(self.images_dir)
    #     not_stagged_dir = '{}/not_stegged'.format(self.images_dir)

