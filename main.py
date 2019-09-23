import numpy as np
import pandas as pd
import os
from models import Models
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array

DATA_NAME = 'k49'
classes_dict = {'kmnist' : 10, 'k49' : 49}

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

def reshapeResize(img_array, data_name):
    if data_name == 'kmnist':
        img_array = img_array.reshape(-1, 28, 28, 1)
    else:
        img_array = img_array.reshape(-1, 48, 48, 1)
        img_array = np.asarray([img_to_array(array_to_img(img, scale = False).resize((48, 48, 1))) for img in img_array])

    return img_array


train_x, train_y, test_x, test_y = readData(DATA_NAME)

train_x = reshapeResize(train_x, DATA_NAME)
test_x = reshapeResize(train_x, DATA_NAME)

train_y_onehot = oneHot(train_y, DATA_NAME, classes_dict)
test_y_onehot = oneHot(test_y, DATA_NAME, classes_dict)

print(train_x.shape)

lenet5 = Models.leNet5(classes_dict[DATA_NAME])
lenet5.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = 10, batch_size = 128)
print(lenet5.evaluate(test_x, test_y_onehot))
