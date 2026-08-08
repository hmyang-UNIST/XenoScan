import os
import cv2
import numpy as np

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

def load_ds_rgb(paths,size_y,size_x,N_data):
    
    ds_path_roi = paths[0]
    ds_path_msk = paths[1]
    
    print('Available data')
    print(ds_path_roi,len(os.listdir(ds_path_roi)))
    print(ds_path_msk,len(os.listdir(ds_path_msk)))
    
    print(f'Using data : {N_data}')
    
    #--- ds SET
    fnames_ds = os.listdir(ds_path_roi)
    fnames_ds.sort()
    if fnames_ds[0][0] == '.':
        fnames_ds = fnames_ds[1:]

    # X_ds = np.zeros( # of images, # Y size, X size, # channels ), dtype=np.float32 )
    # Ex input : color
    #   -- X_ds = np.zeros( (len(fnames_ds), target_size, target_size, 3 ), dtype=np.float32 )
    
    X_ds = np.zeros( (len(fnames_ds), size_y, size_x, 3), dtype=np.float32 )
    Y_ds = np.zeros( (len(fnames_ds), size_y, size_x, 1), dtype=np.bool  )

    # just using for-loop
    for i in tqdm(range(N_data)):
        ifname = f'{i:05d}.jpg'
        i_ds_roi_path = os.path.join(ds_path_roi, ifname)
        roi_bgr  = cv2.imread(i_ds_roi_path)
        roi_rgb  = cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2RGB)
        # roi_rgb  = cv2.resize(roi_rgb,dsize=(target_size,target_size),interpolation=cv2.INTER_LINEAR)
        # msk  = cv2.resize(msk,dsize=(target_size,target_size),interpolation=cv2.INTER_LINEAR)
        # scaling [0-255] ---> [0,1.0], float32
        roi = np.round(roi_rgb/255,5).astype(np.float32)
        X_ds[i] = roi
        
        # msk = np.round(msk/255,1).astype(np.bool)
        ifname = f'{i:05d}.png'
        i_ds_msk_path = os.path.join(ds_path_msk,ifname)
        msk = cv2.imread(i_ds_msk_path,cv2.IMREAD_GRAYSCALE)
        msk = (msk//255).astype(np.bool)
        #--- dimension control | Ex roi : color, no need extra-channel ---> keep the shape
        # roi  = np.expand_dims(roi,axis=-1)
        #--- dimension control | if roi : gray, 2d matrix, need one more channel --> expand_dims
        # roi  = np.expand_dims(roi,axis=-1)
        #--- dimension control | msk : gray 2d matrix, need one more channel --> expand_dims
        msk = np.expand_dims(msk,axis=-1)

        
        Y_ds[i] = msk

    return X_ds, Y_ds


'''
    model build
'''

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPooling2D((2, 2),padding=padding)(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet_rgb(size_y,size_x):
    input_shape = (size_y, size_x, 3)
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input(input_shape)
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model