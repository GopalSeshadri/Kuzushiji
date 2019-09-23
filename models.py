import keras
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from keras.models import Model

class Models:

    def leNet5(num_classes):
        inp = Input(shape = (28, 28, 1))
        x = Conv2D(filters = 6, kernel_size = (3, 3), activation = 'relu')(inp)
        x = AvgPool2D()(x)
        x = Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu')(x)
        x = AvgPool2D()(x)
        x = Flatten()(x)
        x = Dense(120, activation = 'relu')(x)
        x = Dense(84, activation = 'relu')(x)
        out = Dense(num_classes, activation = 'softmax')(x)

        lenet_model = Model(input = inp, output = out)
        lenet_model.compile(loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])

        return lenet_model
