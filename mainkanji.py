import numpy as np
import pandas as pd
import os
from models import Models
from keras.utils import to_categorical
from utilities import Utilities
from matplotlib.image import imread

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

top1000 = findtop1000()
print(top1000)

kanji_data, idx2word = readDataforKanji(top1000)
print(idx2word)
