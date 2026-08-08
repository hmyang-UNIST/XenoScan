import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras import models


def model_GAP(IMG_HEIGHT,IMG_WIDTH,num_class):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.Conv2D(8, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (2, 2), padding='same'))
    model.add(layers.Conv2D(128, (2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(num_class,activation='softmax'))
    
    return model


def model_DENSE(IMG_HEIGHT,IMG_WIDTH,num_class):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.Conv2D(8, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (2, 2), padding='same'))
    model.add(layers.Conv2D(128, (2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (2, 2), padding='same'))
    model.add(layers.Conv2D(128, (2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten() )
    
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(num_class,activation='softmax'))
    
    return model


def model_GAP_GRAY(IMG_HEIGHT,IMG_WIDTH,num_class):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (2, 2), padding='same'))
    model.add(layers.Conv2D(256, (2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(num_class,activation='softmax'))
    
    return model