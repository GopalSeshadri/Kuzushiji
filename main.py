import numpy as np
import pandas as pd
import os
from models import Models
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array

DATA_NAME = 'kmnist'
MODEL_NAME = 'vgg'
classes_dict = {'kmnist' : 10, 'k49' : 49}
BATCH_SIZE = 128
EPOCHS = 2

## Loading Data
def readData(data_name):

    # data_name can be either kmnist or k49
    train_x = np.load('Data/{}/{}-train-imgs.npz'.format(data_name.upper(), data_name.lower()))['arr_0']
    train_y = np.load('Data/{}/{}-train-labels.npz'.format(data_name.upper(), data_name.lower()))['arr_0']

    test_x = np.load('Data/{}/{}-test-imgs.npz'.format(data_name.upper(), data_name.lower()))['arr_0']
    test_y = np.load('Data/{}/{}-test-labels.npz'.format(data_name.upper(), data_name.lower()))['arr_0']

    return train_x, train_y, test_x, test_y

def oneHot(label, data_name, classes_dict):
    label_onehot = to_categorical(label, num_classes = classes_dict[data_name])
    return label_onehot

def reshapeResize(img_array, model_name):
    if model_name == 'lenet':
        img_array = img_array.reshape(-1, 28, 28, 1)
    else:
        img_array = img_array.reshape(-1, 28, 28, 1)
        img_array = np.asarray([img_to_array(array_to_img(img, scale = False).resize((98, 98))) for img in img_array])
        img_array = img_array.reshape(-1, 98, 98, 1)

    img_array = img_array / 255
    return img_array


train_x, train_y, test_x, test_y = readData(DATA_NAME)

train_x = reshapeResize(train_x, MODEL_NAME)
test_x = reshapeResize(test_x, MODEL_NAME)

train_y_onehot = oneHot(train_y, DATA_NAME, classes_dict)
test_y_onehot = oneHot(test_y, DATA_NAME, classes_dict)

if MODEL_NAME == 'lenet':
    lenet5 = Models.leNet5(classes_dict[DATA_NAME])
    lenet5.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE)
    print(lenet5.evaluate(test_x, test_y_onehot))
elif MODEL_NAME == 'vgg':
    vgg16 = Models.vgg16(classes_dict[DATA_NAME])
    vgg16.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE)
    print(vgg16.evaluate(test_x, test_y_onehot))
