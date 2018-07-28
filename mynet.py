# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K


def AlexNet(width, height, depth, classes):
    inputShape = (height, width, depth)
    # if we are using "channels last", update the input shape
    if K.image_data_format() == "channels_first":  # for tensorflow
        inputShape = (depth, height, width)
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=inputShape, padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model


def LeNet(width, height, depth, classes):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)
    # if we are using "channels last", update the input shape
    if K.image_data_format() == "channels_first":   #for tensorflow
        inputShape = (depth, height, width)
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


def LeNet2(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    # if we are using "channels last", update the input shape
    if K.image_data_format() == "channels_first":   #for tensorflow
        inputShape = (depth, height, width)
    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=inputShape,padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Flatten())  
    model.add(Dense(100,activation='relu'))  
    model.add(Dense(classes,activation='softmax'))  
    return model