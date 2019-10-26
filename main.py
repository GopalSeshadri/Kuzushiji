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
    '''
    This function reads the data and returns the train and test data and labels.

    Parameters:
    data_name (str) : The name of the data, it can be either kminst or k49

    Returns:
    train_x (numpy array) : The array of normalised images from training set
    train_y (numpy array) : The array of image labels from training set. Labels can be from 0 to 48
    test_x (numpy array) : The array of normalised images from test set.
    test_y (numpy array) : The array of image label from the test set
    '''

    # data_name can be either kmnist or k49
    train_x = np.load('Data/{}/{}-train-imgs.npz'.format(data_name.upper(), data_name.lower()))['arr_0']
    train_y = np.load('Data/{}/{}-train-labels.npz'.format(data_name.upper(), data_name.lower()))['arr_0']

    test_x = np.load('Data/{}/{}-test-imgs.npz'.format(data_name.upper(), data_name.lower()))['arr_0']
    test_y = np.load('Data/{}/{}-test-labels.npz'.format(data_name.upper(), data_name.lower()))['arr_0']

    return train_x, train_y, test_x, test_y

def oneHot(label, data_name, classes_dict):
    '''
    It takes an array of labels, data set name and the classes dictionary and returns the one-hot representation
    oof the array of labels.

    Parameters:
    label (numpy array) : The array of labels, it can be either training or testing data.
    data_name (string) : The name of the dataset, it can be kmnist or k49
    classes_dict (dict) : A dictionary of dataname and classes counts

    Returns:
    label_onehot (numpy array) : The array of one hot representation of labels.
    '''
    label_onehot = to_categorical(label, num_classes = classes_dict[data_name])
    return label_onehot

def reshapeResize(img_array, model_name):
    '''
    It takes the input array and it reshapes the array based on the model name. It returns the reshaped array.

    Parameters:
    img_array (numpy array) : The array of normalised images.
    model_name (string) : The name of the model

    Returns:
    img_array(numpy array) : The reshaped array
    '''
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
    '''
    This function takes as input an array representation of normalised images and generates the image as a PNG file
    in the assets directory.

    Parameters:
    img_array (numpy array) : An array of normalised images
    '''
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
