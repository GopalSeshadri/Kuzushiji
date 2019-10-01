import keras
from keras.layers import Input, Dense, Flatten, Dropout, Concatenate
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
        x = Dropout(0.2)(x)
        x = Dense(84, activation = 'relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(num_classes, activation = 'softmax')(x)

        lenet_model = Model(input = inp, output = out)
        lenet_model.compile(loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])

        return lenet_model

    def seven(num_classes):
        inp = Input(shape = (64, 64, 1))
        x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(inp)
        x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024, activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation = 'relu')(x)
        out = Dense(num_classes, activation = 'softmax')(x)

        seven_model = Model(input = inp, output = out)
        seven_model.compile(loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])

        return seven_model

    def seven_stacked(num_classes):
        inp = Input(shape = (64, 64, 1))
        x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(inp)
        x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(x)

        x1 = MaxPool2D(pool_size = (2, 2))(x)
        x1 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x1)
        x1 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x1)
        x1 = MaxPool2D(pool_size = (2, 2))(x1)
        x1 = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x1)
        x1 = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x1)
        x1 = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x1)
        x1 = MaxPool2D(pool_size = (2, 2))(x1)
        x1 = Flatten()(x1)

        x2 = MaxPool2D(pool_size = (2, 2))(x)
        x2 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x2)
        x2 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x2)
        x2 = MaxPool2D(pool_size = (2, 2))(x2)
        x2 = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x2)
        x2 = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x2)
        x2 = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x2)
        x2 = MaxPool2D(pool_size = (2, 2))(x2)
        x2 = Flatten()(x2)

        print(x1.shape)
        print(x2.shape)
        x = Concatenate(axis = 1)([x1, x2])
        print(x.shape)

        x = Dense(1024, activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation = 'relu')(x)
        out = Dense(num_classes, activation = 'softmax')(x)

        vgg16_model = Model(input = inp, output = out)
        vgg16_model.compile(loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])

        return vgg16_model

    def vgg16(num_classes):
        inp = Input(shape = (92, 92, 1))
        x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(inp)
        x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu')(x)
        x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu')(x)
        x = MaxPool2D(pool_size = (2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024, activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation = 'relu')(x)
        out = Dense(num_classes, activation = 'softmax')(x)

        vgg16_model = Model(input = inp, output = out)
        vgg16_model.compile(loss = 'categorical_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])

        return vgg16_model
