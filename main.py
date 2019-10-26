import numpy as np
import pandas as pd
import os
from models import Models
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array
from utilities import Utilities

DATA_NAME = 'k49'
MODEL_NAME = 'lenet'
classes_dict = {'kmnist' : 10, 'k49' : 49, 'kkanji' : 1000}
BATCH_SIZE = 128
EPOCHS = 50

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
    elif model_name == 'vgg16':
        img_array = img_array.reshape(-1, 28, 28, 1)
        img_array = np.asarray([img_to_array(array_to_img(img, scale = True).resize((92, 92))) for img in img_array])
        img_array = img_array.reshape(-1, 92, 92, 1)
    elif model_name == 'seven':
        img_array = img_array.reshape(-1, 28, 28, 1)
        img_array = np.asarray([img_to_array(array_to_img(img, scale = True).resize((64, 64))) for img in img_array])
        img_array = img_array.reshape(-1, 64, 64, 1)
    elif model_name == 'seven_stacked':
        img_array = img_array.reshape(-1, 28, 28, 1)
        img_array = np.asarray([img_to_array(array_to_img(img, scale = True).resize((64, 64))) for img in img_array])
        img_array = img_array.reshape(-1, 64, 64, 1)


    img_array = img_array / 255
    return img_array

def generateImages(img_array):
    img_array = img_array.reshape(-1, 28, 28, 1)
    for idx, img in enumerate(img_array):
        image_from_array = array_to_img(img, scale = True).resize((100, 100))
        image_from_array.save('assets/{}.png'.format(idx))



train_x, train_y, test_x, test_y = readData(DATA_NAME)
# generateImages(test_x)
train_x = reshapeResize(train_x, MODEL_NAME)
test_x = reshapeResize(test_x, MODEL_NAME)

train_y_onehot = oneHot(train_y, DATA_NAME, classes_dict)
test_y_onehot = oneHot(test_y, DATA_NAME, classes_dict)

if MODEL_NAME == 'lenet':
    lenet5 = Models.leNet5(classes_dict[DATA_NAME])
    lenet5.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE)
    print(lenet5.evaluate(test_x, test_y_onehot))
    Utilities.saveModel(lenet5, 'lenet')
elif MODEL_NAME == 'vgg16':
    vgg16 = Models.vgg16(classes_dict[DATA_NAME])
    vgg16.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE)
    print(vgg16.evaluate(test_x, test_y_onehot))
    Utilities.saveModel(vgg16, 'vgg16')
elif MODEL_NAME == 'seven':
    seven = Models.seven(classes_dict[DATA_NAME])
    seven.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE)
    print(seven.evaluate(test_x, test_y_onehot))
    Utilities.saveModel(seven, 'seven')
elif MODEL_NAME == 'seven_stacked':
    seven_stacked = Models.seven_stacked(classes_dict[DATA_NAME])
    seven_stacked.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE)
    print(seven_stacked.evaluate(test_x, test_y_onehot))
    Utilities.saveModel(seven_stacked, 'seven_stacked')
