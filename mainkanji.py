import numpy as np
import pandas as pd
import os
from models import Models
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array
from utilities import Utilities
from matplotlib.image import imread
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

def findtop1000():
    path = 'Data/KKANJI/'
    count = [len([f for f in os.listdir(path + dir)]) for dir in os.listdir(path)]
    top1000 = np.argpartition(count, -1000)[-1000:]
    return top1000

def readDataforKanji(top1000):
    kanji_data = []
    i, idx = 0, 0
    path = 'Data/KKANJI/'
    idx2word = {}

    for dir in os.listdir(path):
        if i in top1000:
            dir_path = path + dir
            for f in os.listdir(dir_path):
                image_path = dir_path + '/' + f
                image = imread(image_path)
                kanji_data.append((image, idx))
            idx2word[idx] =  dir
            idx += 1
        i += 1

    return kanji_data, idx2word

def reshapeResize(img_array, model_name):
    if model_name == 'lenet':
        img_array = img_array.reshape(-1, 64, 64, 1)
        img_array = np.asarray([img_to_array(array_to_img(img, scale = True).resize((28, 28))) for img in img_array])
        img_array = img_array.reshape(-1, 28, 28, 1)
    elif model_name == 'vgg16':
        img_array = img_array.reshape(-1, 64, 64, 1)
        img_array = np.asarray([img_to_array(array_to_img(img, scale = True).resize((92, 92))) for img in img_array])
        img_array = img_array.reshape(-1, 92, 92, 1)
    elif model_name == 'seven':
        img_array = img_array.reshape(-1, 64, 64, 1)
    elif model_name == 'seven_stacked':
        img_array = img_array.reshape(-1, 64, 64, 1)

    return img_array

DATA_NAME = 'kkanji'
MODEL_NAME = 'vgg16'
classes_dict = {'kmnist' : 10, 'k49' : 49, 'kkanji' : 1000}
BATCH_SIZE = 32
EPOCHS = 20

## Finding top 1000 Classes and loading only their data
# top1000 = findtop1000()
# print(top1000)
#
# kanji_data, idx2word = readDataforKanji(top1000)
# print(idx2word)

# Utilities.saveData(kanji_data, 'kanji_data')

kanji_data = Utilities.loadData('kanji_data')
## Creating Kanji Dataframe from the loaded 1000 class Kanji Data
kanji_df = pd.DataFrame(kanji_data, columns = ['Image', 'Class'])


## Sampling 50 examples for each classes
kanji_sampled_data = []
for i in range(1000):
    oneclass_df = kanji_df[kanji_df['Class'] == i]
    indices = np.random.randint(low = 0, high = len(oneclass_df), size = 100)
    kanji_sampled_data.extend([tuple(oneclass_df.iloc[idx].values) for idx in indices])

train_x = np.array([row[0] for row in kanji_sampled_data])
train_y = np.array([row[1] for row in kanji_sampled_data])

train_x_reshaped = reshapeResize(train_x, MODEL_NAME)
train_y_onehot = to_categorical(train_y, num_classes = classes_dict[DATA_NAME])

print(train_x_reshaped.shape)
print(train_y_onehot.shape)

vgg = VGG16(input_shape = (64, 64), weights = 'imagenet', include_top = False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
out = Dense(classes_dict[DATA_NAME], activation = 'softmax')(x)

model = Model(inputs = vgg.input, outputs = out)
model.compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])
model.fit(train_x, train_y_onehot, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE)
