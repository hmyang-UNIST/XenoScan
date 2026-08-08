## Imports
import os
import sys
import time
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)

from _train_modules import *

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"]="5"
    
    size_y = 208*4
    size_x = 128*4
    EPOCH = 10000

    #--- model setup
    mname = f'UNET_XENOPUS_{size_y}x{size_x}'

    mpath = './models'
    cpath = './xx_chpt'
    lpath = './xx_logs'
    os.makedirs(mpath,exist_ok=True)
    os.makedirs(cpath,exist_ok=True)
    os.makedirs(lpath,exist_ok=True)

    logs_fpath = os.path.join(lpath,f'{mname}.csv')
    chpt_fpath = os.path.join(cpath,f'{mname}.hdf5')
    model_path = os.path.join(mpath,f'{mname}.h5')
    
    model = UNet_rgb(size_y,size_x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_checkpointer = keras.callbacks.ModelCheckpoint(
        filepath = chpt_fpath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose = 1
    )
    callbacks = [
            model_checkpointer,
            keras.callbacks.EarlyStopping(
                patience=50*3,
                monitor='val_loss',
                mode='min',
                verbose=1
                ),
    ]


    #--- load dataset
    roi_path = f'./__dataset/__ds-trainValid-{size_y}x{size_x}/__roi/'
    msk_path = f'./__dataset/__ds-trainValid-{size_y}x{size_x}/__msk/'
    
    seed = 909
    roi_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    msk_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

    roi_train = \
        roi_datagen.flow_from_directory( 
            roi_path,
            target_size=(size_y, size_x),
            class_mode=None, 
            seed=seed,
            subset='training'
        )

    msk_train = \
        msk_datagen.flow_from_directory( 
            msk_path,
            target_size=(size_y, size_x),
            class_mode=None,
            color_mode='grayscale',
            seed=seed,
            subset='training'
        )
    
    roi_valid = \
        roi_datagen.flow_from_directory( 
            roi_path,
            target_size=(size_y, size_x),
            class_mode=None, 
            seed=seed,
            subset='validation'
        )
    msk_valid = \
        msk_datagen.flow_from_directory( 
            msk_path,
            target_size=(size_y, size_x),
            class_mode=None,
            color_mode='grayscale',
            seed=seed,
            subset='validation'
        )

    train_ds = zip(roi_train, msk_train)
    valid_ds = zip(roi_valid, msk_valid)
    
    
    #--- train start
    print('Start')
    stime = time.time()

    history = model.fit(
            train_ds,
            validation_data=valid_ds,
            validation_steps=15,
            batch_size=32,
            steps_per_epoch=50,
            epochs=EPOCH,
            callbacks=callbacks
        )

    print('Finished')
    ftime = time.time()
    time_consumed = ftime - stime
    print('Time consumed : ', np.round(time_consumed/60,3),'-mins' )

    model.save(
        model_path, overwrite=True, include_optimizer=True, save_format=None,
        signatures=None, options=None
    )
    print('model saved', model_path)

    #--- training curve
    acc        = list(history.history['accuracy'])
    val_acc    = list(history.history['val_accuracy'])
    loss       = list(history.history['loss'])
    val_loss   = list(history.history['val_loss'])
    epochs_log = list(range(1, len(acc) + 1))

    dict_train = {
        'acc' : acc     ,
        'val_acc' : val_acc ,
        'loss' : loss    ,
        'val_loss' : val_loss,
        'steps' : epochs_log  
    }

    df_train = pd.DataFrame.from_dict(dict_train)
    df_train.to_csv(logs_fpath,index=False)
    print(logs_fpath)