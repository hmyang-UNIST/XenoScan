## Imports
import os
import sys
import time
import random
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)

from _train_modules import *
import glob

def get_img_batch(paths_832x512):
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    X_832x512 = np.zeros( (len(paths_832x512), 832, 512, 3), dtype=np.float32 )
    X_208x128 = np.zeros( (len(paths_832x512), 208, 128, 3), dtype=np.float32 )
    for i in range(len(paths_832x512)):
        ipath = paths_832x512[i]
        ibgr = cv2.imread(ipath)
        # get tensor
        irgb = cv2.cvtColor(ibgr.copy(),cv2.COLOR_BGR2RGB)
        for ic in range(3):
            irgb[:,:,ic] = CLAHE.apply(irgb[:,:,ic])
        irgb_208x128 = cv2.resize(irgb,(128,208),cv2.INTER_AREA)
        
        irgb = irgb.astype(np.float32)
        irgb = irgb/255
        X_832x512[i] = irgb
        
        irgb_208x128 = irgb_208x128.astype(np.float32)
        irgb_208x128 = irgb_208x128/255
        X_208x128[i] = irgb_208x128
    return X_832x512, X_208x128, paths_832x512


def main(args):
    gpuid = int(args[1])
    NGPU = 5
    os.environ["CUDA_VISIBLE_DEVICES"]=f'{gpuid}'
    
    dict_labelcode = {
        'CONTROL' : 1,
        'AG1'     : 2,
        'BIO'     : 3,
        'C59'     : 4,
        'IVER'    : 5,
        'IWR'     : 6,
    }
    
    # load models
    mpath = f'./models/UNET_XENOPUS_{832}x{512}.h5'
    model_832x512 = keras.models.load_model(mpath)
    mpath = f'./models/UNET_XENOPUS_{208}x{128}.h5'
    model_208x128 = keras.models.load_model(mpath)
    
    # get roi paths
    rawpath = './data/__rawdata/__v2023-04/__rois'
    srcpaths = glob.glob(f'{rawpath}/*/*.jpg')
    srcpaths.sort()
    if gpuid == 4:
        srcpaths = srcpaths[ int(len(srcpaths)/NGPU*(gpuid)) : ]
    else:
        srcpaths = srcpaths[ int(len(srcpaths)/NGPU*(gpuid)) : int(len(srcpaths)/NGPU*(gpuid+1)) ]
    
    # setup dstpaths
    dstpath_832x512 = './data/__rawdata/__v2023-04/__predictions/__msk_832x512'
    os.makedirs(dstpath_832x512,exist_ok=True)
    dstpath_208x128 = './data/__rawdata/__v2023-04/__predictions/__msk_208x128'
    os.makedirs(dstpath_208x128,exist_ok=True)
    
    
    dict_info = {}
    dict_info['sid'] = []
    dict_info['exp'] = []
    dict_info['plt'] = []
    dict_info['plt_id'] = []
    dict_info['label'] = []
    dict_info['labelcode'] = []
    dict_info['frame'] = []
    dict_info['roipath'] = []
    dict_info['mskpath_832x512'] = []
    dict_info['mskpath_208x128'] = []
    
    N_BATCH = 32
    N = int( len(srcpaths) / N_BATCH )
    
    for idx in tqdm(range(N+1)):
        if idx < N:
            srcs = srcpaths[idx*N_BATCH:(idx+1)*N_BATCH]
            i_N_BATCH = N_BATCH
        else:
            srcs = srcpaths[idx*N_BATCH:]
            i_N_BATCH = len(srcs)

        X_832x512, X_208x128, paths = get_img_batch(srcs)
        batch_pred_832x512 = model_832x512.predict(X_832x512,verbose=0)
        batch_pred_208x128 = model_208x128.predict(X_208x128,verbose=0)
        
        for idx in range(i_N_BATCH):
            # get info
            iroipath = paths[idx]
            info     = iroipath.split('/')[-2].split('_')
            ifname   = iroipath.split('/')[-1].split('.')[0]
            
            isid = int(info[0])
            iexp = info[1]
            iplt = info[2]
            iplt_id = info[3]
            ilabel = info[-1]
            ilabelcode = dict_labelcode[ilabel]
            iframe = int(ifname.split('_')[-1])
            
            i_msk_prob_832x512 = batch_pred_832x512[idx,:,:,0]
            i_msk_prob_832x512 = i_msk_prob_832x512*255
            i_msk_prob_832x512 = i_msk_prob_832x512.astype(np.uint8)
            imskpath_832x512 = os.path.join(dstpath_832x512,f'{ifname}.png')
            cv2.imwrite(imskpath_832x512,i_msk_prob_832x512)

            i_msk_prob_208x128 = batch_pred_208x128[idx,:,:,0]
            i_msk_prob_208x128 = i_msk_prob_208x128*255
            i_msk_prob_208x128 = i_msk_prob_208x128.astype(np.uint8)
            imskpath_208x128 = os.path.join(dstpath_208x128,f'{ifname}.png')
            cv2.imwrite(imskpath_208x128,i_msk_prob_208x128)


            dict_info['sid'].append(isid)
            dict_info['exp'].append(iexp)
            dict_info['plt'].append(iplt)
            dict_info['plt_id'].append(iplt_id)
            dict_info['label'].append(ilabel)
            dict_info['labelcode'].append(ilabelcode)
            dict_info['frame'].append(iframe)
            dict_info['roipath'].append(iroipath)
            dict_info['mskpath_832x512'].append(imskpath_832x512)
            dict_info['mskpath_208x128'].append(imskpath_208x128)
    df = pd.DataFrame.from_dict(dict_info)
    csvpath = f'./__predictions-gpuid{gpuid:03d}.csv'
    df.to_csv(csvpath,index=False)

    return None

if __name__ == "__main__":
    main(sys.argv)
