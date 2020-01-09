
""" Utility functions for retreiving a prediction from the trained model"""

import _pickle as pickle
import numpy as np

INPUT_FEATURES = 784
CLASSES = 10


def vectorize(image):
    image_vector = np.ones(INPUT_FEATURES)
    for i in range(784):
        image_vector[i] = image[i // 28][i % 28]

    return image_vector


def get_digit(image):

    image_vector = vectorize(image)
    binfile = open('predictor_bin', 'rb')
    ann = pickle.load(binfile)

    outputs = ann.predict(image_vector)
    max_prob, max_prob_dig = 0, 0
    for i in range(CLASSES):
        print("Probability of digit to be %i = %f" % (i, outputs[i]))
        if outputs[i] > max_prob:
            max_prob = outputs[i]
            max_prob_dig = i

    return max_prob_dig if max_prob > 0.3 else -1




