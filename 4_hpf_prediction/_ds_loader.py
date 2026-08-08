import cv2
import glob
import copy
import random
import numpy as np
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


class ds_generator(keras.utils.Sequence):
    
    def __init__(self,ds_path,batch_size,shuffle=True):
        
        self.paths = glob.glob(ds_path+'*')
        self.paths.sort()
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        img_0 = cv2.imread(self.paths[0])
        size_y, size_x, size_c = img_0.shape
        self.size_y = size_y
        self.size_x = size_x
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.paths) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.paths)
    
    def __getitem__(self, idx):
        i = idx*self.batch_size
        i_batch_paths = self.paths[i:i+self.batch_size]
        
        X = np.zeros( (self.batch_size, self.size_y, self.size_x, 3), dtype=np.float32 )
        Y = np.zeros( (self.batch_size, 1), dtype=np.float32  )

        # just using for-loop
        for idx in range(len(i_batch_paths)):
            i_fpath = i_batch_paths[idx]
            img_bgr = cv2.imread(i_fpath)
            img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

            # scaling [0-255] ---> [0,1.0], float32
            img = np.round(img_rgb/255,5).astype(np.float32)        
            X[idx] = img

            # get HPF value
            i_frame = int( i_fpath.split('/')[-1].split('.')[0].split('_')[0] )
            i_HPF   = np.round(i_frame*25/60,2)
            Y[idx]  = i_HPF

        return X, Y