from TheNetwork.pre_trained_model import PreTrainedModel

from keras.layers import Dense
from keras.models import Model


class VeGGieModel(object):
    """
    The VeGGie model is a variation on the familiar VGG architecture.
    It is built by popping the last two layers of https://github.com/geifmany/cifar-vgg
        and adding two dense layers (10+relu => 1+sigmoid)
        to make this architecture suitable for binary classification tasks.
    """
    MIDDLE_LAYER_SIZE = 10
    WEIGHT_DECAY = 0.0005
    X_SHAPE = [32, 32, 3]

    def __init__(self):
        self.middle_layer_size = VeGGieModel.MIDDLE_LAYER_SIZE
        self.weight_decay = VeGGieModel.WEIGHT_DECAY
        self.x_shape = VeGGieModel.X_SHAPE

        self.model = None

    def build_veggie_model(self):
        """Build pre-trained network and use transfer learning architecture to build VeGGie."""
        pre_trained_model = PreTrainedModel()
        pre_trained_model.build_pre_trained_model()

        pre_trained_model.model.layers.pop()
        pre_trained_model.model.layers.pop()

        new_dense_layer = Dense(10, activation='relu', name='new_dense_layer')
        new_binary_classification_layer = Dense(1, activation='sigmoid', name='new_binary_classification_layer')

        inp = pre_trained_model.model.input
        middle = new_dense_layer(pre_trained_model.model.layers[-1].output)
        out = new_binary_classification_layer(middle)

        veggie = Model(inp, out)
        self.model = veggie
