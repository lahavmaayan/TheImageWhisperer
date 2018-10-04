
from optparse import OptionParser

from TheNetwork.whisper_detector import WhisperDetector



def Main():
    parser = OptionParser()
    parser.add_option("-m", "--max_num_pics_per_category",
                      help="For testing purposes, specify upper limit for images to load. "
                           "This value should be larger than the batch size.")
    parser.add_option("-e", "--epochs",
                      help="Number of times to train on the whole dataset.")
    parser.add_option("-b", "--batch_size",
                      help="Size of one batch of training. keep small for the large VeGGie network.")
    parser.add_option("-w", "--weights_filename",
                      help="When, for fail-safe reasons, training in separate batches, load weights of previous train.")
    options, _ = parser.parse_args()

    m = int(options.max_num_pics_per_category) if options.max_num_pics_per_category is not None else None
    e = int(options.epochs)
    b = int(options.batch_size)
    whisper_detector = WhisperDetector(m, e, b)

    whisper_detector.build()
    if options.weights_filename is not None:
        whisper_detector.load_weights(options.weights_filename)
    whisper_detector.train()

if __name__ == '__main__':
    Main()
