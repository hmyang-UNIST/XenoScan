import os
import cv2
import copy
import numpy as np
from tqdm import tqdm

import time

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import glob
import random


def load_ds_gray(ds_path, IMG_HEIGHT, IMG_WIDTH, shuffle=True):
    
    imgpaths = list(glob.glob(ds_path + '/*/*.jpg'))
    N_ds     = len(imgpaths)
    # N_ds     = 100
    imgpaths = random.sample(imgpaths,N_ds)
    
    labels   = os.listdir(ds_path)
    N_labels = len(labels)
    labels.sort()

    dict_labelcode = {}
    label_code = 0
    for i_label in labels:
        dict_labelcode[i_label] = label_code
        label_code += 1

    print( labels, len(imgpaths) )
    
    X_ds = np.zeros( (N_ds, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32 )
    Y_ds = np.zeros( (N_ds, 1),                        dtype=np.uint8   )

    # just using for-loop
    for i in tqdm(range(N_ds)):
        
        i_fpath = imgpaths[i]
        i_label = i_fpath.split('/')[-2]
        
        roi = cv2.imread(i_fpath,cv2.IMREAD_GRAYSCALE)
        roi = cv2.resize(roi,(IMG_WIDTH,IMG_HEIGHT),interpolation=cv2.INTER_CUBIC)
        roi = np.round(roi/255,5).astype(np.float32)
        roi = np.expand_dims(roi,axis=-1)
        
        X_ds[i] = roi
        
        i_labelcode = dict_labelcode[i_label]
        Y_ds[i] = i_labelcode

    # i_seed = int(time.time())
    # np.random.seed(i_seed)
    # np.random.shuffle(X_ds)
    # np.random.seed(i_seed)
    # np.random.shuffle(Y_ds)

    return X_ds,Y_ds


def load_ds_rgb(ds_path, IMG_HEIGHT, IMG_WIDTH, shuffle=True):
    
    imgpaths = list(glob.glob(ds_path + '/*/*.jpg'))
    N_ds     = len(imgpaths)
    
    labels   = os.listdir(ds_path)
    N_labels = len(labels)
    labels.sort()

    dict_labelcode = {}
    label_code = 0
    for i_label in labels:
        dict_labelcode[i_label] = label_code
        label_code += 1

    print( labels, len(imgpaths) )
    
    X_ds = np.zeros( (N_ds, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32 )
    Y_ds = np.zeros( (N_ds, 1),                        dtype=np.uint8   )

    # just using for-loop
    for i in tqdm(range(N_ds)):
        
        i_fpath = imgpaths[i]
        i_label = i_fpath.split('/')[-2]
        
        roi = cv2.imread(i_fpath)
        roi = cv2.resize(roi,(IMG_WIDTH,IMG_HEIGHT),interpolation=cv2.INTER_CUBIC)
        roi = np.round(roi/255,5).astype(np.float32)
        # roi = np.expand_dims(roi,axis=-1)
        
        X_ds[i] = roi
        
        i_labelcode = dict_labelcode[i_label]
        Y_ds[i] = i_labelcode

    i_seed = int(time.time())
    np.random.seed(i_seed)
    np.random.shuffle(X_ds)
    np.random.seed(i_seed)
    np.random.shuffle(Y_ds)

    return X_ds,Y_ds


class ds_gen(keras.utils.Sequence):
    
    def __init__(self, ds_path, IMG_HEIGHT, IMG_WIDTH, batch_size, N_ds, mode='rgb', shuffle=True):
        
        # ds_path  = './data/DL_01_dataset/train/TB-{timeblock:02d}/'
        # labels   = os.listdir(ds_path)
        # imgpaths = glob.glob(ds_path + '/*/*.jpg')
        
        # self.N_ds     = len(self.imgpaths)
        # self.N_ds     = N_ds
        
        self.imgpaths = list( glob.glob(ds_path + '/*/*.jpg') )
        self.N_ds     = len(self.imgpaths)
        
        self.labels   = os.listdir(ds_path)
        self.N_labels = len(self.labels)
        self.labels.sort()

        self.dict_labelcode = {}
        label_code = 0
        for i_label in self.labels:
            self.dict_labelcode[i_label] = label_code
            label_code += 1

        print( self.labels, len(self.imgpaths) )

        self.mode = mode
        if mode == 'rgb':
            self.X_ds = np.zeros( (self.N_ds, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32 )
        elif mode == 'gray':
            self.X_ds = np.zeros( (self.N_ds, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32 )
        
        self.Y_ds = np.zeros( (self.N_ds, 1), dtype=np.uint8    )

        # just using for-loop
        for i in tqdm(range(self.N_ds)):
            
            i_fpath = self.imgpaths[i]
            i_label = i_fpath.split('/')[-2]
            
            if mode == 'gray':
                roi = cv2.imread(i_fpath,cv2.IMREAD_GRAYSCALE)
            else:
                roi = cv2.imread(i_fpath)
                
            roi = cv2.resize(roi,(IMG_WIDTH,IMG_HEIGHT),interpolation=cv2.INTER_CUBIC)

            # scaling [0-255] ---> [0,1.0], float32
            roi = np.round(roi/255,5).astype(np.float32)
            
            if mode == 'gray':
                roi = np.expand_dims(roi,axis=-1)
            
            self.X_ds[i] = roi
            
            i_labelcode = self.dict_labelcode[i_label]
            self.Y_ds[i] = i_labelcode
        
        self.batch_size = batch_size
        self.shuffle    = shuffle
        
        if self.shuffle:
            i_seed = int(time.time())
            np.random.seed(i_seed)
            np.random.shuffle(self.X_ds)
            np.random.seed(i_seed)
            np.random.shuffle(self.Y_ds)
    
    def __len__(self):
        return self.N_ds // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            i_seed = int(time.time())
            np.random.seed(i_seed)
            np.random.shuffle(self.X_ds)
            np.random.seed(i_seed)
            np.random.shuffle(self.Y_ds)
    
    def __getitem__(self, idx):
        i = idx*self.batch_size
        X = self.X_ds[i:i+self.batch_size]
        Y = self.Y_ds[i:i+self.batch_size]
        return X, Y