
""" Utility functions for interacting with the dataset """

def get_training_image_vector(image_number):
    file = open('Dataset/train-images-idx3-ubyte', 'rb')
    file.seek(16 + (image_number - 1) * 784, 0)

    image_vector = []
    for i in range(0, 784):
        byte = int.from_bytes(file.read(1), byteorder='big')
        image_vector.append(byte)

    file.close()
    return image_vector


def get_training_image_label(image_number):
    file = open('Dataset/train-labels-idx1-ubyte', 'rb')
    file.seek(8 + (image_number - 1), 0)

    byte = int.from_bytes(file.read(1), byteorder='big')
    image_label = byte

    file.close()
    return image_label

def get_test_image_vector(image_number):
    file = open('Dataset/t10k-images-idx3-ubyte', 'rb')
    file.seek(16 + (image_number - 1) * 784, 0)

    image_vector = []
    for i in range(0, 784):
        byte = int.from_bytes(file.read(1), byteorder='big')
        image_vector.append(byte)

    file.close()
    return image_vector


def get_test_image_label(image_number):
    file = open('Dataset/t10k-labels-idx1-ubyte', 'rb')
    file.seek(8 + (image_number - 1), 0)

    byte = int.from_bytes(file.read(1), byteorder='big')
    image_label = byte

    file.close()
    return image_label
